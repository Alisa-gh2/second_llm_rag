import os
from dotenv import load_dotenv

load_dotenv()

# настройки чанкинга
CHUNK_SIZE = 512
OVERLAP = 256

# модель для плотных эмбеддингов
EMBEDDING_MODEL = 'paraphrase-multilingual-MiniLM-L12-v2'

# гибридный поиск
TOP_K = 50
RRF_K = 30

# переранжирование
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_TOP_N = 5

# генерация (ollama)
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:3b")

# пути к данным
DATA_DIR = os.getenv("DATA_DIR", "./data")
CLEAN_DOCS_DIR = os.getenv("CLEAN_DOCS_DIR", "./data/clean_docs")
INDEX_DIR = os.getenv("INDEX_DIR", "./data/faiss_index")
LOG_DIR = os.getenv("LOG_DIR", "./logs")

# случайное зерно для воспроизводимости
SEED = 42

# настройки api
API_HOST = "0.0.0.0"
API_PORT = 8000
