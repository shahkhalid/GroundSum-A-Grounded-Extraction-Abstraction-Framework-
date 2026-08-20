"""
src/utils.py
============
Centralised configuration, logging setup, and general-purpose helper utilities
shared across the entire hallucination-mitigation pipeline.

All path constants are relative to a user-supplied ``base_dir`` that is
resolved at runtime, making the repository environment-agnostic (local disk,
Google Drive, cloud storage mount, etc.).
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import tiktoken


# 1.  Dataset / training hyper-parameters
# Split sizes used throughout baseline, Longformer, and abstractive pipelines.
TRAIN_SIZE: int = 20000
VAL_SIZE: int = 2500
TEST_SIZE: int = 2500

# Number of sentences extracted in each extractive summary.
SUMMARY_LENGTH: int = 5

# DistilBERT sentence-classifier settings (baseline.py)
THRESHOLD: float = 0.5
MAX_LEN: int = 250
BATCH_SIZE: int = 16
EPOCHS: int = 3

# Keys to strip from raw JSONL records before building HuggingFace Datasets.
DROP_KEYS: frozenset[str] = frozenset({"indices", "sorted_indices", "score"})

# OpenAI / GPT settings
GPT_MODEL: str = "gpt-4o-mini"
GPT_TEMPERATURE: float = 0.0
GPT_TEMPERATURE_BASELINE: float = 0.5

# Pipeline-2 chunking
CHUNK_TOKEN_SIZE: int = 250

# Max files processed in the abstractive rewrite pipeline
MAX_FILES: int = 2_500

# Datasets targeted by the pipeline (can be extended to include "pubmed")
DATASETS: list[str] = ["arxiv_enriched"]


# 2.  Path helpers

def build_paths(base_dir: str) -> dict[str, str]:
    """
    Construct all project-level directory paths from a single *base_dir* root.

    Parameters
    ----------
    base_dir : str
        Root directory of the project (e.g. ``/data/thesis_project``).

    Returns
    -------
    dict[str, str]
        Mapping of logical path names → absolute path strings.
        All directories are created on disk if they do not already exist.
    """
    paths: dict[str, str] = {
        "base":        base_dir,
        "dataset":     os.path.join(base_dir, "dataset"),
        "arxiv":       os.path.join(base_dir, "dataset", "arxiv"),
        "pubmed":      os.path.join(base_dir, "dataset", "pubmed"),
        "arxiv_enr":   os.path.join(base_dir, "dataset", "arxiv_enriched"),
        "pubmed_enr":  os.path.join(base_dir, "dataset", "pubmed_enriched"),
        "summaries":   os.path.join(base_dir, "summaries"),
        "abs_summaries": os.path.join(base_dir, "abs_summaries"),
        "baselines":   os.path.join(base_dir, "baselines"),
        "extractive":  os.path.join(base_dir, "extractive"),
        "models":      os.path.join(base_dir, "models"),
        "checkpoints": os.path.join(base_dir, "checkpoints"),
        "results":     os.path.join(base_dir, "results"),
        "logs":        os.path.join(base_dir, "logs"),
    }
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    return paths


# 3.  Logging

def setup_logging(log_dir: str, filename: str = "pipeline.log") -> logging.Logger:
    """
    Configure the root logger to write to both a rotating file and stdout.

    Parameters
    ----------
    log_dir : str
        Directory in which the log file will be written.
    filename : str
        Log file name (default: ``pipeline.log``).

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, filename)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s — %(levelname)s — %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)
    logger.info("Logging initialised → %s", log_path)
    return logger


# 4.  JSON / JSONL I/O helpers

def save_json(data: dict[str, Any], path: str) -> None:
    """
    Serialise *data* to a pretty-printed JSON file at *path*.

    Parameters
    ----------
    data : dict
        Serialisable Python dictionary.
    path : str
        Destination file path (parent directories are created automatically).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=4)


def save_jsonl(records: list[dict[str, Any]], path: str) -> None:
    """
    Write *records* as newline-delimited JSON to *path*.

    Parameters
    ----------
    records : list[dict]
        List of serialisable Python dictionaries.
    path : str
        Destination file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl(path: str) -> list[dict[str, Any]]:
    """
    Load every line of a JSONL file and return a list of parsed dictionaries.

    Parameters
    ----------
    path : str
        Path to the ``.jsonl`` file.

    Returns
    -------
    list[dict]
        Parsed records.
    """
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            records.append(json.loads(line))
    return records


# 5.  Text / token helpers

def chunk_by_tokens(
    text: str,
    max_tokens: int = CHUNK_TOKEN_SIZE,
    model: str = GPT_MODEL,
) -> list[str]:
    """
    Split *text* into chunks of approximately *max_tokens* using the tiktoken
    encoder for *model*, preserving sentence boundaries where possible.

    The function encodes the entire text, then iteratively selects a token
    window and back-tracks to the last sentence-ending punctuation mark so
    that each chunk does not cut mid-sentence.

    Parameters
    ----------
    text : str
        Input text to chunk.
    max_tokens : int
        Approximate upper bound on tokens per chunk.
    model : str
        OpenAI model name used to select the correct tiktoken encoding.

    Returns
    -------
    list[str]
        Ordered list of text chunks.
    """
    enc = tiktoken.encoding_for_model(model)
    token_ids = enc.encode(text)
    chunks: list[str] = []

    i = 0
    while i < len(token_ids):
        j = min(i + max_tokens, len(token_ids))
        chunk = enc.decode(token_ids[i:j])

        # Attempt to break at the last sentence boundary.
        last_period = max(chunk.rfind("."), chunk.rfind("?"), chunk.rfind("!"))
        if last_period > 0 and j < len(token_ids):
            chunk = chunk[: last_period + 1]
            j = i + len(enc.encode(chunk))

        chunks.append(chunk.strip())
        i = j

    return chunks


def safe_batch_decode(
    preds: "np.ndarray | list",
    tokenizer: Any,
    skip_special_tokens: bool = True,
) -> list[str]:
    """
    Robustly convert logits or token-ID arrays into decoded Python strings.

    This utility handles the full conversion pipeline:

    1. Cast *preds* to ``np.ndarray`` if it is not already.
    2. If *preds* has 3 dimensions (batch × seq_len × vocab), apply argmax
       over the last axis to obtain token IDs.
    3. Cast to ``int32`` and convert to a nested Python list.
    4. Clamp each token ID to the valid vocabulary range ``[0, vocab_size)``.
    5. Call ``tokenizer.batch_decode`` with ``clean_up_tokenization_spaces=True``.

    Parameters
    ----------
    preds : np.ndarray or list
        Raw logit array (shape ``[B, L, V]``) *or* token-ID array
        (shape ``[B, L]`` or ``[L]``).
    tokenizer : PreTrainedTokenizer
        HuggingFace tokenizer used for decoding.
    skip_special_tokens : bool
        Whether to strip special tokens from decoded strings (default: True).

    Returns
    -------
    list[str]
        Decoded text sequences, one per batch element.
    """
    if not isinstance(preds, np.ndarray):
        preds = np.array(preds)

    if preds.ndim == 3:
        preds = preds.argmax(axis=-1)

    preds = preds.astype(np.int32)
    id_seqs = preds.tolist()

    if preds.ndim == 1:
        id_seqs = [id_seqs]

    vocab_size: int = (
        getattr(tokenizer, "vocab_size", None) or tokenizer.model_max_length
    )

    clamped: list[list[int]] = [
        [tok for tok in seq if 0 <= tok < vocab_size] for seq in id_seqs
    ]

    return tokenizer.batch_decode(
        clamped,
        skip_special_tokens=skip_special_tokens,
        clean_up_tokenization_spaces=True,
    )


# 6.  Miscellaneous

def inspect_jsonl(path: str, n: int = 3) -> None:
    """
    Print a brief structural summary of a JSONL file to stdout.

    Prints the file path, total record count, keys found in the first record,
    and the first *n* raw records.

    Parameters
    ----------
    path : str
        Path to the ``.jsonl`` file to inspect.
    n : int
        Number of sample records to display (default: 3).
    """
    records = load_jsonl(path)
    print(f"File: {path}")
    print(f"Total records: {len(records)}")
    if records:
        print("Keys per record:", list(records[0].keys()))
    print(f"Sample ({min(n, len(records))} entries):")
    for rec in records[:n]:
        print(rec)
    print()


def timer(func):  # type: ignore[no-untyped-def]
    """
    Simple decorator that logs the wall-clock execution time of a function.

    Parameters
    ----------
    func : callable
        The function to wrap.

    Returns
    -------
    callable
        Wrapped function that prints elapsed time after each call.
    """
    def wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"[timer] {func.__name__} completed in {elapsed:.2f}s")
        return result
    return wrapper
