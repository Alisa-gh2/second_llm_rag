import os
import time
import json
import pickle
from contextlib import asynccontextmanager
from typing import List, Dict, Any

import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from config import *
from utils import clean_text, split_into_chunks, setup_logger

load_dotenv()

# глобальные переменные для загруженных моделей
dense_model = None
index = None
bm25 = None
chunks = []
rerank_tokenizer = None
rerank_model = None
logger = None

# вспомогательные функции rag-пайплайна

def load_dense_model():
    """загружает модель для получения плотных эмбеддингов."""
    return SentenceTransformer(EMBEDDING_MODEL)

def load_rerank_model():
    """загружает токенизатор и модель cross-encoder для переранжирования."""
    tokenizer = AutoTokenizer.from_pretrained(RERANK_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(RERANK_MODEL)
    model.eval()
    return tokenizer, model

def load_index_and_bm25():
    """загружает сохранённые faiss индекс, bm25 и список чанков из index_dir."""
    index_path = os.path.join(INDEX_DIR, "faiss.index")
    chunks_path = os.path.join(INDEX_DIR, "chunks.json")
    bm25_path = os.path.join(INDEX_DIR, "bm25.pkl")

    if not os.path.exists(index_path) or not os.path.exists(chunks_path) or not os.path.exists(bm25_path):
        raise RuntimeError("индексы не найдены. сначала запустите scripts/index_data.py")

    idx = faiss.read_index(index_path)
    with open(chunks_path, 'r', encoding='utf-8') as f:
        chunk_list = json.load(f)
    with open(bm25_path, 'rb') as f:
        bm25_obj = pickle.load(f)
    return idx, bm25_obj, chunk_list

def hybrid_search_rrf(query: str, top_k: int = TOP_K, k_rrf: int = RRF_K):
    """
    гибридный поиск: dense (faiss) + bm25 с объединением через reciprocal rank fusion.
    возвращает список кортежей (chunk, score).
    """
    # плотный поиск
    q_emb = dense_model.encode([query])
    faiss.normalize_L2(q_emb)
    dense_scores, dense_indices = index.search(q_emb, top_k)
    dense_ranks = {int(idx): rank+1 for rank, idx in enumerate(dense_indices[0]) if idx != -1}

    # bm25 поиск
    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)
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

def rerank(query: str, candidates_with_scores: List[tuple], top_n: int = RERANK_TOP_N):
    """
    переранжирование кандидатов с помощью cross-encoder.
    возвращает список лучших чанков.
    """
    pairs = [(query, chunk["text"]) for chunk, _ in candidates_with_scores]
    inputs = rerank_tokenizer(pairs, padding=True, truncation=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        scores = rerank_model(**inputs).logits.squeeze().tolist()
    if isinstance(scores, float):
        scores = [scores]
    reranked = sorted(zip(candidates_with_scores, scores), key=lambda x: x[1], reverse=True)
    return [chunk for (chunk, _), _ in reranked[:top_n]]

def generate_answer(query: str, chunks_list: List[Dict], model_name: str = LLM_MODEL) -> str:
    """
    генерирует ответ на основе найденных чанков с помощью ollama.
    """
    if not chunks_list:
        return "не найдено релевантных фрагментов."

    context_text = "\n\n".join([c["text"] for c in chunks_list])
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
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": model_name, "prompt": prompt, "stream": False, "options": {"temperature": 0.0}},
            timeout=120
        )
        if response.status_code == 200:
            return response.json()["response"].strip()
        else:
            return f"ошибка api: {response.status_code}"
    except Exception as e:
        return f"ошибка: {e}"

# lifespan для fastapi (загрузка моделей при старте)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global dense_model, index, bm25, chunks, rerank_tokenizer, rerank_model, logger
    # загрузка при старте
    logger = setup_logger(LOG_DIR)
    logger.info("запуск rag сервиса...")
    logger.info("загрузка модели плотных эмбеддингов...")
    dense_model = load_dense_model()
    logger.info("загрузка модели переранжирования...")
    rerank_tokenizer, rerank_model = load_rerank_model()
    logger.info("загрузка faiss индекса и bm25...")
    index, bm25, chunks = load_index_and_bm25()
    logger.info("сервис готов.")
    yield
    logger.info("остановка сервиса.")

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
        initial_results = hybrid_search_rrf(request.question, top_k=request.top_k)
        best_chunks = rerank(request.question, initial_results, top_n=RERANK_TOP_N)
        answer = generate_answer(request.question, best_chunks)
        sources = [c["text"] for c in best_chunks] if request.include_sources else []
        logger.info(f"q: {request.question} | a: {answer[:200]} | time: {time.time()-start_time:.2f}s")
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
    status = {
        "status": "ok",
        "models_loaded": {
            "dense_model": dense_model is not None,
            "rerank_model": rerank_model is not None,
            "faiss_index": index is not None,
            "bm25": bm25 is not None,
        },
        "num_chunks": len(chunks) if chunks else 0,
        "ollama_url": OLLAMA_URL,
        "llm_model": LLM_MODEL,
    }
    # проверим доступность ollama
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        status["ollama_reachable"] = resp.status_code == 200
    except:
        status["ollama_reachable"] = False
    return status
