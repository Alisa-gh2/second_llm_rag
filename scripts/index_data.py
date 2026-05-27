#!/usr/bin/env python3
"""
однократная индексация документов.
создаёт faiss.index и chunks.json в папке faiss_index/
"""

import os
import sys
import json
import logging
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CLEAN_DOCS_DIR, INDEX_DIR, CHUNK_SIZE, OVERLAP, EMBEDDING_MODEL
from utils import clean_text, chunk_text
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    logger.info("загрузка модели эмбеддингов: %s", EMBEDDING_MODEL)
    dense_model = SentenceTransformer(EMBEDDING_MODEL)

    docs_dir = Path(CLEAN_DOCS_DIR)
    if not docs_dir.exists():
        logger.error("папка %s не существует", CLEAN_DOCS_DIR)
        sys.exit(1)

    # создаём все чанки
    all_chunks = []
    for txt_file in docs_dir.glob("*.txt"):
        with open(txt_file, 'r', encoding='utf-8') as f:
            text = f.read()
        text = clean_text(text)
        chunks = chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP)
        for i, ch in enumerate(chunks):
            all_chunks.append({"id": f"{txt_file.stem}-{i}", "text": ch})
    logger.info("всего чанков: %d", len(all_chunks))

    if not all_chunks:
        logger.warning("нет чанков, индексация прервана")
        sys.exit(0)

    # определяем размерность эмбеддинга на первом чанке
    sample_embedding = dense_model.encode([all_chunks[0]["text"]])[0]
    dim = sample_embedding.shape[0]
    index = faiss.IndexFlatL2(dim)  # используем L2
    logger.info("размерность эмбеддинга: %d", dim)

    # кодируем чанки батчами
    batch_size = 32
    total_batches = (len(all_chunks) + batch_size - 1) // batch_size
    logger.info("кодирование чанков батчами по %d (всего батчей: %d)", batch_size, total_batches)

    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i+batch_size]
        batch_texts = [c["text"] for c in batch]
        logger.debug("обработка батча %d/%d", i//batch_size + 1, total_batches)
        embeddings = dense_model.encode(batch_texts, show_progress_bar=False)
        embeddings = np.array(embeddings).astype('float32')
        index.add(embeddings)
        del embeddings

    logger.info("faiss индекс создан, добавлено %d векторов", index.ntotal)

    # сохраняем артефакты
    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(index, os.path.join(INDEX_DIR, "faiss.index"))
    with open(os.path.join(INDEX_DIR, "chunks.json"), 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    logger.info("артефакты сохранены в %s", INDEX_DIR)


if __name__ == "__main__":
    main()
