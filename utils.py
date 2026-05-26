import re
import random
import numpy as np
import torch
from config import SEED

# Fix random seeds
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def clean_text(text: str, lower: bool = False) -> str:
    """Remove page numbers, fix hyphenation, normalize whitespace."""
    patterns = [
        r'-\s*\d+\s*-',           # - 1 -
        r'page\s*\d+',            # page 1
        r'стр\.?\s*\d+',          # стр. 1
        r'с\.\s*\d+',             # с. 1
        r'\[\s*\d+\s*\]',         # [1]
        r'\b\d+\s*/\s*\d+\b',     # 1/3
    ]
    for pat in patterns:
        text = re.sub(pat, ' ', text, flags=re.IGNORECASE)
    # Fix hyphenated line breaks
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    if lower:
        text = text.lower()
    return text

def split_into_chunks(text: str, chunk_size: int = 512, overlap: int = 256) -> list:
    """Split text into overlapping chunks, trying to cut at sentence boundaries."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            # find last period within chunk
            last_period = text.rfind('.', start, end)
            if last_period != -1 and last_period > start:
                end = last_period + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap if end < len(text) else end
    return chunks

def setup_logger(log_dir: str, name: str = "rag_api"):
    import logging
    import os
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(f"{log_dir}/requests.log")
    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(fh)
    return logger
