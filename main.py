# основной сервер fastapi с эндпоинтами /ask и /health

import os
import time
import json
import pickle
import requests
from contextlib import asynccontextmanager
from typing import List, Dict, Tuple

import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openrouter import OpenRouter

from config import *
from utils import setup_logger

# глобальные переменные (инициализируются при старте)
dense_model = None
index = None
bm25 = None
chunks = []
rerank_tokenizer = None
rerank_model = None
logger = None
openrouter_client = None


# загрузка моделей и индексов
def load_dense_model():
    """загружает модель для плотных эмбеддингов (sentence‑transformer)."""
    return SentenceTransformer(EMBEDDING_MODEL)


def load_rerank_model():
    """загружает токенизатор и модель cross-encoder для переранжирования."""
    tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
    model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
    model.eval()
    return tokenizer, model


def load_index_and_bm25():
    """загружает сохранённые faiss индекс, bm25 и список чанков из папки INDEX_DIR."""
    index_path = os.path.join(INDEX_DIR, "faiss.index")
    chunks_path = os.path.join(INDEX_DIR, "chunks.json")
    bm25_path = os.path.join(INDEX_DIR, "bm25.pkl")

    if not (os.path.exists(index_path) and os.path.exists(chunks_path) and os.path.exists(bm25_path)):
        raise RuntimeError(f"индексы не найдены в {INDEX_DIR}. сначала запустите python scripts/index_data.py")

    idx = faiss.read_index(index_path)
    with open(chunks_path, 'r', encoding='utf-8') as f:
        chunk_list = json.load(f)
    with open(bm25_path, 'rb') as f:
        bm25_obj = pickle.load(f)
    return idx, bm25_obj, chunk_list


# поиск и ранжирование
def hybrid_search_rrf(query: str, top_k: int = TOP_K, k_rrf: int = RRF_K) -> List[Tuple[Dict, float]]:
    """
    гибридный поиск: объединяет результаты faiss (dense) и bm25 (sparse) с помощью rrf.
    возвращает список кортежей (chunk, rrf_score).
    """
    # плотный поиск (faiss)
    q_emb = dense_model.encode([query])
    faiss.normalize_L2(q_emb)
    dense_scores, dense_indices = index.search(q_emb, top_k)
    dense_ranks = {int(idx): rank+1 for rank, idx in enumerate(dense_indices[0]) if idx != -1}

    # поиск bm25
    bm25_scores = bm25.get_scores(query.split())
    bm25_top_indices = np.argsort(bm25_scores)[-top_k:][::-1]
    bm25_ranks = {int(idx): rank+1 for rank, idx in enumerate(bm25_top_indices)}

    # rrf слияние
    rrf_scores = {}
    for idx, rank in dense_ranks.items():
        rrf_scores[idx] = 1 / (k_rrf + rank)
    for idx, rank in bm25_ranks.items():
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (k_rrf + rank)

    sorted_indices = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)
    return [(chunks[idx], rrf_scores[idx]) for idx in sorted_indices[:top_k]]


def rerank(query: str, candidates_with_scores: List[Tuple], top_n: int = RERANK_TOP_N) -> List[Dict]:
    """
    переранжирование кандидатов с помощью cross-encoder.
    возвращает топ‑n чанков.
    """
    if not candidates_with_scores:
        return []
    pairs = [(query, chunk["text"]) for chunk, _ in candidates_with_scores]
    inputs = rerank_tokenizer(pairs, padding=True, truncation=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        scores = rerank_model(**inputs).logits.squeeze().tolist()
    if isinstance(scores, float):
        scores = [scores]
    reranked = sorted(zip(candidates_with_scores, scores), key=lambda x: x[1], reverse=True)
    return [chunk for (chunk, _), _ in reranked[:top_n]]


# генерация ответа через openrouter
def generate_answer(query: str, context_chunks: List[Dict]) -> str:
    """отправляет промпт с контекстом в openrouter и возвращает ответ модели."""
    if not context_chunks:
        return "не найдено релевантных фрагментов."
    context_text = "\n\n".join([c["text"] for c in context_chunks])
    prompt = f"""ты — ассистент, отвечающий строго на основе предоставленного контекста.
контекст состоит из фрагментов документов. твои шаги:
1. прочитай внимательно каждый фрагмент.
2. найди конкретные факты, связанные с вопросом.
3. составь краткий ответ, используя только эти факты (можно цитировать).
4. если в контексте недостаточно информации для ответа, напиши ровно: 'информации в предоставленных документах недостаточно.'
не добавляй ничего от себя.

**контекст:**
{context_text}

**вопрос:** {query}

**ответ:"""
    try:
        response = openrouter_client.chat.send(
            model=OPENROUTER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.05,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"ошибка: {e}"


# lifespan — инициализация при старте
@asynccontextmanager
async def lifespan(app: FastAPI):
    global dense_model, index, bm25, chunks, rerank_tokenizer, rerank_model, logger, openrouter_client
    logger = setup_logger(LOG_DIR)
    logger.info("запуск rag сервиса...")
    logger.info("загрузка модели эмбеддингов...")
    dense_model = load_dense_model()
    logger.info("загрузка модели переранжирования...")
    rerank_tokenizer, rerank_model = load_rerank_model()
    logger.info("инициализация клиента openrouter...")
    openrouter_client = OpenRouter(api_key=OPENROUTER_API_KEY)
    logger.info("загрузка индексов...")
    index, bm25, chunks = load_index_and_bm25()
    logger.info(f"загружено {len(chunks)} чанков")
    logger.info("сервис готов")
    yield
    logger.info("остановка сервиса")


# создание fastapi приложения
app = FastAPI(title="RAG Service", lifespan=lifespan)


# pydantic модели для запросов/ответов
class QueryRequest(BaseModel):
    question: str
    top_k: int = TOP_K
    include_sources: bool = True


class QueryResponse(BaseModel):
    answer: str
    sources: List[str] = []
    time_taken: float


# эндпоинты
@app.post("/ask", response_model=QueryResponse)
async def ask(request: QueryRequest):
    """
    основной эндпоинт: принимает вопрос, возвращает ответ и источники.
    """
    start_time = time.time()
    try:
        candidates = hybrid_search_rrf(request.question, top_k=request.top_k)
        best_chunks = rerank(request.question, candidates, top_n=RERANK_TOP_N)
        answer = generate_answer(request.question, best_chunks)
        sources = [c["text"] for c in best_chunks] if request.include_sources else []
        logger.info(f"q: {request.question[:50]}... | time: {time.time()-start_time:.2f}s")
        return QueryResponse(answer=answer, sources=sources, time_taken=time.time() - start_time)
    except Exception as e:
        logger.error(f"ошибка: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """
    эндпоинт для проверки состояния сервера.
    возвращает статус и информацию о загруженных компонентах.
    """
    return {
        "status": "ok",
        "models_loaded": {
            "dense_model": dense_model is not None,
            "rerank_model": rerank_model is not None,
            "faiss_index": index is not None,
            "bm25": bm25 is not None,
        },
        "num_chunks": len(chunks) if chunks else 0,
        "openrouter_model": OPENROUTER_MODEL,
    }
