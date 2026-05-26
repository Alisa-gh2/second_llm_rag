import os
from dotenv import load_dotenv

load_dotenv()

# --- Chunking ---
CHUNK_SIZE = 512
OVERLAP = 256

# --- Dense Embedding Model ---
EMBEDDING_MODEL = 'paraphrase-multilingual-MiniLM-L12-v2'

# --- Hybrid Search ---
TOP_K = 50          # initial retrieved candidates
RRF_K = 30          # reciprocal rank fusion constant

# --- Reranking ---
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_TOP_N = 5    # number of chunks after reranking

# --- Generation (Ollama) ---
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:3b")

# --- Paths ---
DATA_DIR = os.getenv("DATA_DIR", "./data")
CLEAN_DOCS_DIR = os.getenv("CLEAN_DOCS_DIR", "./data/clean_docs")
INDEX_DIR = os.getenv("INDEX_DIR", "./data/faiss_index")
LOG_DIR = os.getenv("LOG_DIR", "./logs")

# --- Random Seed for reproducibility ---
SEED = 42

# --- API Settings ---
API_HOST = "0.0.0.0"
API_PORT = 8000
