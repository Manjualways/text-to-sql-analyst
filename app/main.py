from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.database import DatabaseNotFoundError
from app.llm import (
    LLMAPIError,
    LLMConfigurationError,
    LLMQuotaExceededError,
    LLMRateLimitError,
)
from app.logging_utils import get_logger
from app.pipeline import run_pipeline
from app.schema_retriever import SchemaIndexError

logger = get_logger(__name__)

app = FastAPI(
    title="Text-to-SQL Analyst Agent",
    version="0.1.0",
    description="Ask questions about the Chinook database in plain English.",
)

class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


@app.get("/")
def root():
    return {
        "message": "Text-to-SQL Analyst Agent is running!",
        "status": "success",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/query")
def query_database(request: QueryRequest):
    try:
        return run_pipeline(request.question)
    except LLMConfigurationError as exc:
        logger.warning("Query rejected because the LLM API key is not configured")
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except DatabaseNotFoundError as exc:
        logger.exception("Configured database is unavailable")
        raise HTTPException(
            status_code=503,
            detail="The configured database is unavailable. Check DATABASE_PATH.",
        ) from exc
    except SchemaIndexError as exc:
        logger.exception("Query service dependency is unavailable")
        raise HTTPException(
            status_code=503,
            detail="The schema retrieval service is unavailable.",
        ) from exc
    except LLMQuotaExceededError as exc:
        logger.warning("LLM provider quota is exhausted")
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc
    except LLMRateLimitError as exc:
        logger.warning("LLM provider is rate-limiting requests")
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc
    except LLMAPIError as exc:
        logger.exception("LLM provider failed while processing a query")
        raise HTTPException(
            status_code=502,
            detail="The language model provider could not complete the request.",
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error while processing a query")
        raise HTTPException(
            status_code=500,
            detail="The query could not be completed. "
                   "Check the question and server logs.",
        ) from exc