import re
import random
import numpy as np
import torch
from config import SEED

# фиксируем seed для воспроизводимости
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def clean_text(text: str, lower: bool = False) -> str:
    """
    очистка текста: удаление номеров страниц, исправление переносов, нормализация пробелов.
    """
    # паттерны для удаления номеров страниц и подобного
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
    
    # склеиваем слова, разорванные переносом строки
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    
    # убираем лишние пробелы
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    if lower:
        text = text.lower()
    return text

def split_into_chunks(text: str, chunk_size: int = 512, overlap: int = 256) -> list:
    """
    разбивает текст на перекрывающиеся чанки, стараясь резать по границам предложений.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            # ищем последнюю точку в пределах чанка
            last_period = text.rfind('.', start, end)
            if last_period != -1 and last_period > start:
                end = last_period + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        # следующий старт с перекрытием
        start = end - overlap if end < len(text) else end
    return chunks

def setup_logger(log_dir: str, name: str = "rag_api"):
    """настройка логгера для записи запросов и ответов."""
    import logging
    import os
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(f"{log_dir}/requests.log")
    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(fh)
    return logger
