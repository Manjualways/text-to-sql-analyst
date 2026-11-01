import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

API_KEY = os.getenv("LLM_API_KEY", "")
BASE_URL = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
MODEL = os.getenv("LLM_MODEL", "openai/gpt-4.1-mini")

DB_PATH = ROOT_DIR / os.getenv("DATABASE_PATH", "data/Chinook_Sqlite.sqlite")
CHROMA_PATH = ROOT_DIR / os.getenv("CHROMA_PATH", "chroma_db")
