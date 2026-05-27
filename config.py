# конфигурация проекта

import os
from dotenv import load_dotenv

load_dotenv()

# openrouter
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "z-ai/glm-4.5-air:free"

# пути
CLEAN_DOCS_DIR = "./clean_docs"
INDEX_DIR = "./faiss_index"
LOG_DIR = "./logs"

# чанкинг
CHUNK_SIZE = 512
OVERLAP = 256

# поиск
TOP_K = 15
RRF_K = 30
RERANK_TOP_N = 5

# эмбеддинги
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# воспроизводимость
SEED = 42

# api
API_HOST = "0.0.0.0"
API_PORT = 8000
