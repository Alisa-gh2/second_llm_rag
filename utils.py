# вспомогательные функции

import re
import random
import numpy as np
import torch
import logging
import os
from config import SEED

# фиксируем seed для воспроизводимости
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def clean_text(text: str, lower: bool = False) -> str:
    """очистка текста: удаление номеров страниц, исправление переносов, нормализация пробелов."""
    patterns = [
        r'-\s*\d+\s*-',
        r'page\s*\d+',
        r'стр\.?\s*\d+',
        r'с\.\s*\d+',
        r'\[\s*\d+\s*\]',
        r'\b\d+\s*/\s*\d+\b',
    ]
    for pat in patterns:
        text = re.sub(pat, ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    if lower:
        text = text.lower()
    return text


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    """простое разбиение на чанки с перекрытием (как в рабочем коде)"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def setup_logger(log_dir: str, name: str = "rag_api"):
    """настройка логгера"""
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(f"{log_dir}/requests.log")
    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(fh)
    return logger
