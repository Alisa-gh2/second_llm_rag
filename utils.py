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
    """очистка текста: удаление номеров страниц, исправление переносов"""
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

def recursive_split(text: str, chunk_size: int = 512, chunk_overlap: int = 256, separators: list = None) -> list:
    """
    разбивает текст на перекрывающиеся чанки, ища разделители в порядке приоритета
    """
    if separators is None:
        separators = ["\n\n", "\n", " ", ""]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        split_pos = -1
        for sep in separators:
            pos = text.rfind(sep, start, end + 1)
            if pos != -1:
                split_pos = pos
                break
        if split_pos == -1:
            split_pos = end
        chunk = text[start:split_pos].strip()
        if chunk:
            chunks.append(chunk)
        next_start = split_pos - chunk_overlap
        if next_start <= start:
            next_start = split_pos
        start = next_start
        if start >= len(text):
            break
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

def simple_stem(word):
    """простой стеммер для русского языка — удаляет типичные окончания"""
    word = word.lower()
    suffixes = ['ая', 'яя', 'ые', 'ие', 'ой', 'ей', 'ую', 'юю', 'ого', 'ему', 'ым', 'им', 'ом', 'ем',
                'ая', 'яя', 'ие', 'ые', 'ое', 'а', 'я', 'о', 'е', 'и', 'ы', 'у', 'ю', 'ь', 'й']
    for suffix in suffixes:
        if word.endswith(suffix):
            word = word[:-len(suffix)]
            break
    if len(word) < 3:
        return word
    return word

def normalize_text(text):
    """извлекает все русские слова из текста и приводит их к стемам"""
    words = re.findall(r'\b[а-яё]+\b', text.lower())
    stems = [simple_stem(w) for w in words]
    return set(stems)
