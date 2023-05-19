import os
from dotenv import load_dotenv
from chromadb.config import Settings

load_dotenv()

PERSISTENT_DIR = os.environ.get("PERSIST_DIR")

CHROMA_SETTINGS = Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=PERSISTENT_DIR,
    anonymized_telemetry=False,
)
