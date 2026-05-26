#!/usr/bin/env python3
"""
скрипт для предварительной индексации документов.
запускается один раз для создания faiss индекса, bm25 и сохранения чанков.
"""

import os
import sys
import json
import pickle
from pathlib import Path

# добавляем путь к корню проекта для импорта конфигурации
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CLEAN_DOCS_DIR, INDEX_DIR, CHUNK_SIZE, OVERLAP, EMBEDDING_MODEL
from utils import clean_text, split_into_chunks
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from rank_bm25 import BM25Okapi

def main():
    print("загрузка модели плотных эмбеддингов...")
    dense_model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"чтение документов из {CLEAN_DOCS_DIR}...")
    docs_dir = Path(CLEAN_DOCS_DIR)
    if not docs_dir.exists():
        print(f"ошибка: папка {CLEAN_DOCS_DIR} не существует.")
        sys.exit(1)

    all_chunks = []
    for txt_file in docs_dir.glob("*.txt"):
        with open(txt_file, 'r', encoding='utf-8') as f:
            text = f.read()
        text = clean_text(text)
        chunks = split_into_chunks(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP)
        for i, ch in enumerate(chunks):
            all_chunks.append({"id": f"{txt_file.stem}-{i}", "text": ch})
    print(f"всего создано чанков: {len(all_chunks)}")

    # плотные эмбеддинги
    chunk_texts = [c["text"] for c in all_chunks]
    embeddings = dense_model.encode(chunk_texts, show_progress_bar=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    print(f"faiss индекс создан, размерность {dim}")

    # bm25
    tokenized_chunks = [text.split() for text in chunk_texts]
    bm25 = BM25Okapi(tokenized_chunks)
    print("bm25 индекс создан.")

    # сохраняем артефакты
    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(index, os.path.join(INDEX_DIR, "faiss.index"))
    with open(os.path.join(INDEX_DIR, "chunks.json"), 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    with open(os.path.join(INDEX_DIR, "bm25.pkl"), 'wb') as f:
        pickle.dump(bm25, f)
    print(f"все артефакты сохранены в {INDEX_DIR}")

if __name__ == "__main__":
    main()
