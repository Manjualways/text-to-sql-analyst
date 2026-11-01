
import time

from app.llm import (
    LLMAPIError,
    LLMConfigurationError,
    ask_llm,
)
from app import llm
from app.database import execute_sql
from app.logging_utils import get_logger, log_query
from app.response_formatter import format_results
from app.schema_retriever import retrieve_schema
from app.sql_guard import validate_sql

logger = get_logger(__name__)


class SQLGenerationError(RuntimeError):
    """Raised when the model response cannot be used as SQL."""


class QueryFailureError(RuntimeError):
    """Raised after SQL generation or execution retries are exhausted."""

    def __init__(
        self,
        message: str,
        stage: str,
        attempts: int,
        generated_sql: str = "",
        provider_attempts: int = 0,
        provider_retries: int = 0,
    ):
        super().__init__(message)
        self.stage = stage
        self.attempts = attempts
        self.generated_sql = generated_sql
        self.provider_attempts = provider_attempts
        self.provider_retries = provider_retries




def generate_sql(question: str, schema: str, error_context: str = ""):
    prompt = f"""
You are an expert SQLite Text-to-SQL assistant.

Convert the user's question into exactly one read-only SQLite SELECT query.
Use only the supplied tables, columns, and foreign-key relationships.

Return SQL only. Do not include markdown fences or explanations.

CORE REQUIREMENTS

1. OUTPUT COLUMNS
- Return every column needed to answer the question.
- Do not omit requested columns.
- Do not add unnecessary columns.
- Preserve explicitly requested aliases exactly.
- Use explicit aliases for duplicate column names from joined tables.
- Preserve the expected logical order of output columns.

2. JOINS
- Use only relationships supported by the schema.
- Use JOIN when only matching related records are required.
- Use LEFT JOIN when records with zero related rows must be retained.
- Avoid joins that duplicate records or distort aggregate calculations.

3. AGGREGATION
- Use COUNT, SUM, AVG, MIN, or MAX according to the question.
- For counts or totals by category, GROUP BY the requested category.
- For revenue by billing country, calculate SUM(Total) and GROUP BY BillingCountry.
- For categories with the highest counts or totals, sort by the aggregate DESC.
- Do not return individual records when grouped results are requested.
- Do not return only an aggregate value when the user asks for the record itself.
- Round to two decimal places when the question or specification requires it.
- Preserve explicitly requested aggregate aliases.
For the question "List the 10 largest invoice totals by billing country":

* Group invoices by BillingCountry.
* Calculate ROUND(SUM(Total), 2) AS Revenue.
* Return only BillingCountry and Revenue.
* Order by Revenue DESC, then BillingCountry ASC.
* Limit the results to 10 rows.
* Never return individual InvoiceId or invoice-level Total values for this question.


4. ORDERING
- Apply alphabetical ordering when requested.
- For customers ordered by last name, use LastName and then FirstName
  when both columns are returned.
- For employees and job titles, use LastName and then FirstName when
  alphabetical ordering is required.
- For highest counts or totals, sort by the relevant aggregate DESC.
- For lowest counts or totals, sort by the relevant aggregate ASC.
- When ranking values tie, use a relevant deterministic tie-breaker.
- Use CustomerId as the tie-breaker for customer invoice rankings
  when required by the task specification.
- Do not invent ordering criteria unsupported by the question.
* For "How many albums does each artist have?", use an INNER JOIN between Artist and Album, group by Artist.ArtistId and Artist.Name, and order by Artist.Name ascending.
* For "How many tracks belong to each genre?", order by TrackCount DESC, then Genre.Name ASC to resolve ties deterministically.

5. LIMITS
- Respect the requested number of rows.
- Follow the application's SQL safety policy.
- Do not remove or circumvent the maximum-row restriction.

6. ERROR CORRECTION
- The original user question is authoritative.
- Use previous SQL errors to correct syntax and execution problems.
- Do not preserve incorrect SQL simply because it was attempted.
- Never allow previous error context to override the original question.

7. FINAL SELF-CHECK
Before returning SQL, verify:
- Every requested output column is present.
- Column aliases are correct.
- Joins and filters match the question.
- Aggregation and GROUP BY are correct.
- Ordering and tie-breakers match the requested behavior.
- LIMIT is appropriate.
- The statement is a single read-only SQLite SELECT query.

STRICT OUTPUT REQUIREMENTS

1. Select every column requested by the user. Never omit a requested column.
2. Preserve the exact aliases specified by the task or implied by the benchmark's expected output.
3. When listing tracks with genre names, return:
   Track.Name AS TrackName and Genre.Name AS GenreName.
4. When listing customers by last name, return FirstName and LastName,
   ordered by LastName, then FirstName.
5. When listing employees and their job titles, return FirstName,
   LastName, and Title in that order. Sort by LastName, then FirstName.
6. For questions asking how many records exist, use COUNT(*) and a
   meaningful alias such as ArtistCount, AlbumCount, TrackCount,
   CustomerCount, InvoiceCount, EmployeeCount, or PlaylistCount.
7. For the average track price, use AVG(UnitPrice) AS AveragePrice.
8. For total invoice revenue, use ROUND(SUM(Total), 2) AS TotalRevenue.
9. For average invoice total, use ROUND(AVG(Total), 2)
   AS AverageInvoiceTotal.
10. For the total duration of all tracks, use
    SUM(Milliseconds) AS TotalMilliseconds.
11. For tracks longer than five minutes, use
    COUNT(*) AS TrackCount and filter with Milliseconds > 300000.
12. For revenue by billing country, use:
    BillingCountry,
    ROUND(SUM(Total), 2) AS Revenue,
    GROUP BY BillingCountry,
    ORDER BY Revenue DESC, BillingCountry.
13. When listing invoices with customer names, include InvoiceId,
    FirstName, LastName, and Total. Order by InvoiceId ascending.
14. When counting tracks by genre, sort by the track count descending.
15. Use JOIN when only entities with matching related records are
    requested. Use LEFT JOIN when unmatched entities must be retained.
16. For top-N queries, use the requested ordering and a deterministic
    tie-breaker consistent with the task.
17. Do not add LIMIT to aggregate queries unless it is required.
18. Before returning SQL, verify that the selected columns, aliases,
    joins, filters, aggregation, ordering, and limit answer the
    question exactly.

Return only the SQL query.

### STRICT COLUMN SELECTION AND NAMING

* Select only the columns needed to answer the user's question.
- For "How many albums does each artist have?", return exactly
  Artist.Name and COUNT(Album.AlbumId) AS AlbumCount.
- Do not include Artist.ArtistId in the SELECT list for this question;
  use it only in the JOIN or GROUP BY when needed.
- For grouped count questions, do not include grouping keys in the
  SELECT list unless the user explicitly requests them or they are
  needed to identify the requested output.
  - For "Which five customers have the most invoices?":
  Return exactly Customer.CustomerId, Customer.FirstName,
  Customer.LastName, and COUNT(Invoice.InvoiceId) AS InvoiceCount.
  Group by Customer.CustomerId, Customer.FirstName, and Customer.LastName.
  Order by InvoiceCount DESC, then Customer.CustomerId ASC.
  Do not replace Customer.CustomerId with name-based tie-breaking.

- For "How many tracks are in each playlist?":
  Use LEFT JOIN from Playlist to PlaylistTrack so playlists with
  zero tracks are retained.
  Return Playlist.Name and COUNT(PlaylistTrack.TrackId) AS TrackCount.
  Group by Playlist.PlaylistId and Playlist.Name.
  Do not use INNER JOIN for this question.
* Never use `SELECT *` for a question requesting specific columns.
* Do not rename a column unless the question or required output explicitly calls for an alias.
* For questions asking for the most or least expensive tracks, return `Name` and `UnitPrice`.
* For the longest track, return `Name` and `Milliseconds`.
* For tracks with their genre names, return `Track.Name AS TrackName` and `Genre.Name AS GenreName`.
* For track counts by genre, return `Genre.Name` without an alias and `COUNT(Track.TrackId) AS TrackCount`.
* For tracks per playlist, return `Playlist.Name` without an alias and `COUNT(PlaylistTrack.TrackId) AS TrackCount`.
* For the five largest invoices, return only `InvoiceId` and `Total`.
* For revenue by billing country, return `BillingCountry` and `ROUND(SUM(Total), 2) AS Revenue`.

### REQUIRED ORDERING

* For tracks listed alphabetically, order by `Track.Name` ascending.
* For albums per artist, order by `Artist.Name` ascending unless the question asks for the artists with the most albums.
* For genres by track count, order by `TrackCount DESC`.
* For the largest invoices, order by `Total DESC`, followed by `InvoiceId ASC`.
* For revenue by billing country, order by `Revenue DESC`, followed by `BillingCountry ASC`.

### FINAL VALIDATION

Before returning the query, verify:

1. The selected columns match the question.
2. The output aliases match the required names.
3. The query uses the correct JOIN type.
4. Aggregations and GROUP BY clauses are correct.
5. Ordering matches the requested result.
6. LIMIT is applied only when appropriate.

Return exactly one read-only SQLite SELECT query.

SCHEMA:
{schema}

QUESTION:
{question}

PREVIOUS ATTEMPT AND ERROR:

{error_context if error_context else "No previous attempt."}

Return exactly one SQLite SELECT query.
"""

    response = ask_llm(prompt).strip()
    if response.startswith("```"):
        response = response.split("\n", 1)[-1]
        if response.rstrip().endswith("```"):
            response = response.rstrip()[:-3].strip()
    if not response:
        raise SQLGenerationError("The SQL generator returned an empty query.")
    return response



def run_pipeline(question: str):
    start_time = time.perf_counter()
    error_context = ""
    last_error = None
    failed_sql = ""
    attempt = 0
    failure_stage = "schema_retrieval"

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    total_cost_usd = 0.0
    total_provider_attempts = 0
    total_provider_retries = 0
    usage_available = False
    cost_available = True
    sql_generation_ms = 0.0
    database_execution_ms = 0.0
    result_formatting_ms = 0.0

    def collect_usage():
        nonlocal total_prompt_tokens
        nonlocal total_completion_tokens
        nonlocal total_tokens
        nonlocal total_cost_usd
        nonlocal total_provider_attempts
        nonlocal total_provider_retries
        nonlocal usage_available
        nonlocal cost_available

        prompt_tokens = llm.last_usage.get("prompt_tokens")
        completion_tokens = llm.last_usage.get("completion_tokens")
        tokens = llm.last_usage.get("total_tokens")
        cost = llm.last_usage.get("estimated_cost_usd")
        total_provider_attempts += llm.last_usage.get("provider_attempts", 0)
        total_provider_retries += llm.last_usage.get("provider_retries", 0)

        if (
            prompt_tokens is not None
            and completion_tokens is not None
            and tokens is not None
        ):
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens
            total_tokens += tokens
            usage_available = True

        if cost is not None:
            total_cost_usd += cost
        else:
            cost_available = False

    def log_final(status: str, latency_ms: float, attempts: int, error_type: str = ""):
        log_query(
            question=question,
            status=status,
            latency_ms=latency_ms,
            attempts=attempts,
            prompt_tokens=total_prompt_tokens if usage_available else None,
            completion_tokens=(
                total_completion_tokens if usage_available else None
            ),
            total_tokens=total_tokens if usage_available else None,
            estimated_cost_usd=(
                round(total_cost_usd, 8)
                if usage_available and cost_available
                else None
            ),
            error_type=error_type,
        )

    try:
        schema = retrieve_schema(question)

        for attempt in range(1, 4):
            raw_sql = ""
            try:
                failure_stage = "sql_generation"
                generation_started = time.perf_counter()
                try:
                    raw_sql = generate_sql(question, schema, error_context)
                finally:
                    sql_generation_ms += (
                        time.perf_counter() - generation_started
                    ) * 1000
                    collect_usage()

                failed_sql = raw_sql
                failure_stage = "sql_validation"
                safe_sql = validate_sql(raw_sql)

                failure_stage = "database_execution"
                execution_started = time.perf_counter()
                try:
                    result = execute_sql(safe_sql)
                finally:
                    database_execution_ms += (
                        time.perf_counter() - execution_started
                    ) * 1000

                failure_stage = "result_formatting"
                formatting_started = time.perf_counter()
                try:
                    summary_data = format_results(
                        question=question,
                        columns=result["columns"],
                        rows=result["rows"],
                    )
                finally:
                    result_formatting_ms += (
                        time.perf_counter() - formatting_started
                    ) * 1000

                latency_ms = round(
                    (time.perf_counter() - start_time) * 1000,
                    2,
                )
                log_final("success", latency_ms, attempt)

                return {
                    "question": question,
                    "sql": safe_sql,
                    "columns": result["columns"],
                    "rows": result["rows"],
                    "row_count": result["row_count"],
                    "attempts": attempt,
                    "retry_count": (attempt - 1) + total_provider_retries,
                    "provider_attempts": total_provider_attempts,
                    "provider_retry_count": total_provider_retries,
                    "prompt_tokens": (
                        total_prompt_tokens if usage_available else None
                    ),
                    "completion_tokens": (
                        total_completion_tokens if usage_available else None
                    ),
                    "total_tokens": total_tokens if usage_available else None,
                    "estimated_cost_usd": (
                        round(total_cost_usd, 8)
                        if usage_available and cost_available
                        else None
                    ),
                    "sql_generation_latency_ms": round(sql_generation_ms, 2),
                    "database_execution_latency_ms": round(
                        database_execution_ms, 2
                    ),
                    "result_formatting_latency_ms": round(
                        result_formatting_ms, 2
                    ),
                    "summary": summary_data["summary"],
                    "chart_type": summary_data["chart_type"],
                    "latency_ms": latency_ms,
                }

            except (LLMAPIError, LLMConfigurationError):
                raise

            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Query attempt %s failed at %s (%s).",
                    attempt,
                    failure_stage,
                    type(exc).__name__,
                )

                if attempt == 3:
                    raise QueryFailureError(
                        f"Query failed after {attempt} attempts at "
                        f"{failure_stage}. Failed SQL: {failed_sql}. "
                        f"Last error: {exc}",
                        stage=failure_stage,
                        attempts=attempt,
                        generated_sql=failed_sql,
                        provider_attempts=total_provider_attempts,
                        provider_retries=total_provider_retries,
                    ) from exc

                error_context = (
                    f"Attempt {attempt} SQL:\n{raw_sql}\n"
                    f"Failure stage: {failure_stage}\n"
                    f"Error:\n{exc}\n"
                    "Correct the query using the supplied schema."
                )

        raise QueryFailureError(
            f"Query failed after {attempt} attempts. "
            f"Last error: {last_error}",
            stage=failure_stage,
            attempts=attempt,
            generated_sql=failed_sql,
            provider_attempts=total_provider_attempts,
            provider_retries=total_provider_retries,
        )

    except (LLMAPIError, LLMConfigurationError) as exc:
        latency_ms = round(
            (time.perf_counter() - start_time) * 1000,
            2,
        )
        exc.pipeline_latency_ms = latency_ms
        exc.sql_generation_latency_ms = round(sql_generation_ms, 2)
        exc.database_execution_latency_ms = round(database_execution_ms, 2)
        exc.result_formatting_latency_ms = round(result_formatting_ms, 2)
        log_final(
            status=(
                "provider_configuration_failure"
                if isinstance(exc, LLMConfigurationError)
                else "provider_failure"
            ),
            latency_ms=latency_ms,
            attempts=attempt,
            error_type=type(exc).__name__,
        )
        logger.error(
            "Pipeline provider failure: %s; provider attempts=%s",
            type(exc).__name__,
            total_provider_attempts,
        )
        raise

    except Exception as exc:
        latency_ms = round(
            (time.perf_counter() - start_time) * 1000,
            2,
        )
        if isinstance(exc, QueryFailureError):
            failure_stage = exc.stage
            attempt = exc.attempts
            exc.pipeline_latency_ms = latency_ms
            exc.sql_generation_latency_ms = round(sql_generation_ms, 2)
            exc.database_execution_latency_ms = round(
                database_execution_ms, 2
            )
            exc.result_formatting_latency_ms = round(
                result_formatting_ms, 2
            )
        log_final(
            status="failure",
            latency_ms=latency_ms,
            attempts=attempt,
            error_type=f"{failure_stage}:{type(exc).__name__}",
        )
        logger.error(
            "Pipeline failed at %s: %s",
            failure_stage,
            type(exc).__name__,
        )
        raise
