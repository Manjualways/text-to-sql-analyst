
import chromadb

from app.config import CHROMA_PATH
from app.database import DatabaseNotFoundError, get_schema
from app.logging_utils import get_logger

logger = get_logger(__name__)


class SchemaIndexError(RuntimeError):
    """Raised when the schema vector index cannot be initialized or queried."""


_collection = None


def _get_collection():
    global _collection

    if _collection is None:
        try:
            CHROMA_PATH.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(CHROMA_PATH))
            _collection = client.get_or_create_collection(name="chinook_schema")
        except Exception as exc:
            logger.exception("Failed to initialize the ChromaDB schema index")
            raise SchemaIndexError("Failed to initialize ChromaDB schema index") from exc

    return _collection

def index_schema():
    collection = _get_collection()

    try:
        schema = get_schema()
        if not schema:
            raise RuntimeError("No database tables found")

        existing = collection.get(include=["documents"])
        existing_documents = dict(
            zip(existing.get("ids", []), existing.get("documents") or [])
        )
        current_documents = {
            item["table"]: item["description"]
            for item in schema
        }

        if existing_documents == current_documents:
            return

        collection.upsert(
            ids=[item["table"] for item in schema],
            documents=[item["description"] for item in schema],
            metadatas=[{"table": item["table"]} for item in schema],
        )
        stale_ids = set(existing_documents) - set(current_documents)
        if stale_ids:
            collection.delete(ids=sorted(stale_ids))
    except DatabaseNotFoundError:
        raise
    except SchemaIndexError:
        raise
    except Exception as exc:
        logger.exception("Failed to index the database schema in ChromaDB")
        raise SchemaIndexError("Failed to index the database schema") from exc


def retrieve_schema(question: str, top_k: int = 4):
    index_schema()
    collection = _get_collection()

    try:
        count = collection.count()
        if count == 0:
            raise RuntimeError("Schema index is empty")

        results = collection.query(
            query_texts=[question],
            n_results=min(top_k, count),
        )
        selected_documents = (
            results["documents"][0] if results["documents"] else []
        )
        if not selected_documents:
            raise RuntimeError("No relevant database tables were retrieved")

        selected_tables = list(dict.fromkeys(
            metadata["table"]
            for metadata in (
                results["metadatas"][0] if results["metadatas"] else []
            )
            if metadata and metadata.get("table")
        ))
        schema_by_table = {
            item["table"]: item
            for item in get_schema()
        }
        related_tables = {}
        for item in schema_by_table.values():
            for relationship in item["relationships"]:
                referenced_table = relationship.split(" -> ", 1)[1].split(
                    ".", 1
                )[0]
                related_tables.setdefault(item["table"], set()).add(
                    referenced_table
                )
                related_tables.setdefault(referenced_table, set()).add(
                    item["table"]
                )

        relevant_tables = set(selected_tables)
        for table in selected_tables:
            relevant_tables.update(related_tables.get(table, set()))

        ordered_tables = list(selected_tables)
        ordered_tables.extend(
            sorted(relevant_tables - set(selected_tables))
        )
        return "\n".join(
            schema_by_table[table]["description"]
            for table in ordered_tables
            if table in schema_by_table
        )
    except Exception as exc:
        logger.exception("Failed to retrieve schema context from ChromaDB")
        raise SchemaIndexError("Failed to retrieve schema context") from exc