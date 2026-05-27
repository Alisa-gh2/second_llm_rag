# основной сервер fastapi с эндпоинтами /ask и /health

import os
import json
import time
import logging
from contextlib import asynccontextmanager
from typing import List, Dict

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openrouter import OpenRouter
from dotenv import load_dotenv

from config import *

# загружаем .env
load_dotenv()

# логи
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# проверка API-ключа
API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
if not API_KEY:
    raise ValueError("Переменная окружения OPENROUTER_API_KEY не задана! Создайте файл .env")

# глобальные переменные
embedder = None
index = None
chunks = []


def load_models_and_index():
    global embedder, index, chunks
    logger.info("загрузка модели эмбеддингов...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    index_path = os.path.join(INDEX_DIR, "faiss.index")
    chunks_path = os.path.join(INDEX_DIR, "chunks.json")

    if not (os.path.exists(index_path) and os.path.exists(chunks_path)):
        raise RuntimeError(f"индексы не найдены в {INDEX_DIR}. сначала запустите python scripts/index_data.py")

    logger.info("загрузка faiss индекса...")
    index = faiss.read_index(index_path)
    logger.info("загрузка чанков...")
    with open(chunks_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    logger.info(f"загружено {len(chunks)} чанков")


def retrieve(query: str, top_k: int = TOP_K) -> List[Dict]:
    """поиск по faiss, возвращает список чанков"""
    q_emb = embedder.encode([query])
    q_emb = np.array(q_emb).astype('float32')
    distances, indices = index.search(q_emb, top_k)
    faiss.normalize_L2(q_emb)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx != -1 and idx < len(chunks):
            results.append(chunks[idx])
    return results


def build_prompt(query: str, context_chunks: List[Dict]) -> str:
    """формирует промпт для llm"""
    context_parts = []
    for i, ch in enumerate(context_chunks):
        context_parts.append(f"[Фрагмент {i+1}]\n{ch['text']}")
    context = "\n\n".join(context_parts)
    system_msg = (
        "Ты – ассистент, отвечающий строго по предоставленному контексту.\n"
        "1. Прочитай контекст.\n"
        "2. Найди факты, относящиеся к вопросу.\n"
        "3. Дай краткий ответ, используя только эти факты.\n"
        "4. Если информации недостаточно, напиши: 'Информации в предоставленных документах недостаточно.'\n"
        "Не добавляй ничего от себя."
    )
    return f"{system_msg}\n\nКонтекст:\n{context}\n\nВопрос: {query}\nОтвет:"


def generate_answer(query: str, context_chunks: List[Dict]) -> str:
    """отправляет запрос в openrouter и возвращает ответ"""
    if not context_chunks:
        return "не найдено релевантных фрагментов."
    prompt = build_prompt(query, context_chunks)
    try:
        with OpenRouter(api_key=API_KEY) as client:
            response = client.chat.send(
                model=OPENROUTER_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=0.05,
            )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"ошибка llm: {e}")
        return f"ошибка генерации: {e}"


# lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("инициализация сервиса...")
    try:
        load_models_and_index()
        logger.info("сервис готов")
    except Exception as e:
        logger.error(f"ошибка инициализации: {e}")
        raise
    yield
    logger.info("сервис остановлен")


app = FastAPI(title="RAG Service", lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str
    top_k: int = TOP_K
    include_sources: bool = True


class QueryResponse(BaseModel):
    answer: str
    sources: List[str] = []
    time_taken: float


@app.post("/ask", response_model=QueryResponse)
async def ask(request: QueryRequest):
    start = time.time()
    try:
        best_chunks = retrieve(request.question, top_k=request.top_k)
        answer = generate_answer(request.question, best_chunks)
        sources = [c["text"] for c in best_chunks] if request.include_sources else []
        return QueryResponse(answer=answer, sources=sources, time_taken=time.time() - start)
    except Exception as e:
        logger.error(f"ошибка: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": embedder is not None,
        "index_loaded": index is not None,
        "num_chunks": len(chunks) if chunks else 0,
        "openrouter_model": OPENROUTER_MODEL,
    }
