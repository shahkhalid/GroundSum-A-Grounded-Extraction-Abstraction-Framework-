"""
src/data_processing.py
=======================
All dataset-related logic for the hallucination-mitigation summarization
pipeline:

* Downloading the ArXiv and PubMed long-document extractive summarization
  datasets from Hugging Face.
* TF-IDF-based extractive oracle summary generation.
* Enriched dataset serialisation (JSONL) with oracle annotations.
* BERTScore-based semantic similarity evaluation between TF-IDF oracle
  summaries and human-authored gold abstracts.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import wget
from bert_score import score as bert_score_fn
from datasets import (
    Dataset,
    DatasetDict,
    Features,
    Sequence,
    Value,
    load_dataset,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from src.utils import TRAIN_SIZE, VAL_SIZE, TEST_SIZE, load_jsonl, save_jsonl


# 1. Feature schema (shared across all dataset loaders)

DATASET_FEATURES: Features = Features(
    {
        "text": Sequence(Value("string")),
        "summary": Sequence(Value("string")),
        "indices": Sequence(Sequence(Value("int64"))),
        "score": Sequence(Value("float64")),
        "sorted_indices": Sequence(Value("int64")),
    }
)


# 2. Dataset downloading

def download_datasets(dataset_path: str) -> None:
    """
    Download the ArXiv and PubMed long-document summarization splits from the
    Hugging Face Hub via direct HTTPS URLs.

    Each dataset has three splits (``train``, ``val``, ``test``), each saved
    as a ``.jsonl`` file inside ``<dataset_path>/<dataset_name>/``.

    Temporary files produced during download are removed automatically.

    Parameters
    ----------
    dataset_path : str
        Root directory under which ``arxiv/`` and ``pubmed/`` sub-directories
        will be created and populated.
    """
    dataset_names: list[str] = ["arxiv", "pubmed"]
    splits: list[str] = ["train", "val", "test"]

    for dataset_name in dataset_names:
        dataset_dir = os.path.join(dataset_path, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)
        print(f"Downloading {dataset_name} …")

        for split in splits:
            url = (
                f"https://huggingface.co/datasets/nianlong/"
                f"long-doc-extractive-summarization-{dataset_name}/"
                f"resolve/main/{split}.jsonl"
            )
            wget.download(url, out=dataset_dir)

        # Remove any stale temporary files left by wget.
        for tmp_file in Path(dataset_dir).glob("*.tmp"):
            tmp_file.unlink()

    print(f"\nDownload complete. Files located in: {dataset_path}")


# 3. Dataset loading

def load_custom_dataset(dataset_name: str, dataset_path: str) -> DatasetDict:
    """
    Load a raw ArXiv or PubMed JSONL dataset into a HuggingFace
    :class:`~datasets.DatasetDict` with the fixed feature schema.

    The ``val.jsonl`` file is mapped to the ``validation`` split key so that
    it aligns with the HuggingFace convention used by Trainer.

    Parameters
    ----------
    dataset_name : str
        One of ``"arxiv"`` or ``"pubmed"``.
    dataset_path : str
        Root directory that contains ``<dataset_name>/train.jsonl`` etc.

    Returns
    -------
    DatasetDict
        Dictionary with keys ``"train"``, ``"validation"``, ``"test"``.
    """
    base = os.path.join(dataset_path, dataset_name)
    return load_dataset(
        "json",
        data_files={
            "train":      os.path.join(base, "train.jsonl"),
            "validation": os.path.join(base, "val.jsonl"),
            "test":       os.path.join(base, "test.jsonl"),
        },
        features=DATASET_FEATURES,
    )


# 4. TF-IDF oracle extraction

def tfidf_oracle(text_sentences: list[str], top_k: int = 5) -> list[str]:
    """
    Select the *top_k* most informative sentences from *text_sentences* using
    TF-IDF term weighting.

    Each sentence is treated as a "document" and scored by summing the TF-IDF
    weights of all tokens it contains.  The top-scoring sentences are returned
    in descending score order (the caller may re-order them to restore document
    order if required).

    If the input has fewer than *top_k* sentences, all sentences are returned
    unchanged to avoid an empty oracle.

    Parameters
    ----------
    text_sentences : list[str]
        Ordered list of sentences from the source document.
    top_k : int
        Number of sentences to select (default: 5).

    Returns
    -------
    list[str]
        Selected oracle sentences (up to *top_k*).
    """
    if not text_sentences or len(text_sentences) < top_k:
        return text_sentences

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(text_sentences)
    sentence_scores: np.ndarray = np.asarray(tfidf_matrix.sum(axis=1)).ravel()
    top_indices: np.ndarray = sentence_scores.argsort()[::-1][:top_k]
    return [text_sentences[i] for i in top_indices]


def generate_oracle_summary(text: list[str] | str, top_k: int = 5) -> list[str]:
    """
    Convenience wrapper that accepts either a list of sentences or a raw string
    and returns the TF-IDF oracle summary as a list of sentence strings.

    Parameters
    ----------
    text : list[str] | str
        Source document as a list of sentences or a single concatenated string.
    top_k : int
        Number of sentences to extract (default: 5).

    Returns
    -------
    list[str]
        Oracle summary sentences.
    """
    if isinstance(text, str):
        # Treat as a single-element list to avoid character-level TF-IDF.
        sentences: list[str] = [text]
    else:
        sentences = text
    return tfidf_oracle(sentences, top_k=top_k)


# 5. Enriched dataset generation (oracle annotation + serialisation)

def generate_tfidf_oracle_dataset(
    dataset_dict: DatasetDict,
    top_k: int = 5,
) -> DatasetDict:
    """
    Annotate every split in *dataset_dict* with a TF-IDF oracle summary column
    and return the enriched :class:`~datasets.DatasetDict`.

    This in-memory version is suitable for small experiments.  For full-scale
    runs, use :func:`generate_and_save_trimmed_split` to write JSONL files.

    Parameters
    ----------
    dataset_dict : DatasetDict
        Raw dataset with ``"text"`` and ``"summary"`` columns.
    top_k : int
        Number of oracle sentences per document (default: 5).

    Returns
    -------
    DatasetDict
        Enriched dataset with an added ``"oracle_summary"`` column in every
        split.
    """
    enriched_splits: dict[str, Dataset] = {}
    for split_name in ["train", "validation", "test"]:
        split_data = dataset_dict[split_name]
        oracle_summaries: list[list[str]] = [
            tfidf_oracle(example["text"], top_k=top_k)
            for example in split_data
        ]
        split_data = split_data.add_column("oracle_summary", oracle_summaries)
        print(
            f"Added TF-IDF oracle_summary to '{split_name}' "
            f"({len(split_data)} examples)"
        )
        enriched_splits[split_name] = split_data

    return DatasetDict(enriched_splits)


def generate_and_save_trimmed_split(
    dataset_split: Dataset,
    save_path: str,
    split_name: str,
    max_samples: int,
    top_k: int = 5,
) -> None:
    """
    Trim *dataset_split* to *max_samples* examples, annotate each with a
    TF-IDF oracle summary, and persist the enriched records to
    ``<save_path>/<split_name>.jsonl``.

    This is the main entry point for preparing the enriched JSONL files
    consumed by the Longformer and baseline training pipelines.

    Parameters
    ----------
    dataset_split : Dataset
        A single HuggingFace dataset split (e.g. the ``"train"`` slice).
    save_path : str
        Output directory (created automatically if absent).
    split_name : str
        Logical split name used as the output filename stem
        (e.g. ``"train"`` → ``train.jsonl``).
    max_samples : int
        Maximum number of examples to include (trimmed from the front).
    top_k : int
        Number of oracle sentences per document (default: 5).
    """
    os.makedirs(save_path, exist_ok=True)
    trimmed = dataset_split.select(range(min(max_samples, len(dataset_split))))

    enriched: list[dict[str, Any]] = []
    for example in tqdm(trimmed, desc=f"Generating oracle for '{split_name}'"):
        ex_dict: dict[str, Any] = {k: example[k] for k in example.keys()}
        oracle = generate_oracle_summary(ex_dict["text"], top_k=top_k)
        ex_dict["oracle_summary"] = oracle if isinstance(oracle, list) else []
        enriched.append(ex_dict)

    output_path = os.path.join(save_path, f"{split_name}.jsonl")
    save_jsonl(enriched, output_path)
    print(f"Saved {len(enriched)} examples → {output_path}")


def build_enriched_datasets(
    dataset_path: str,
    top_k: int = 5,
) -> None:
    """
    End-to-end pipeline that loads the raw ArXiv and PubMed datasets, trims
    each split to the canonical sizes defined in ``utils.py``, generates TF-IDF
    oracle annotations, and writes the enriched JSONL files to disk.

    Outputs are written to:
    * ``<dataset_path>/pubmed_enriched/{train,val,test}.jsonl``
    * ``<dataset_path>/arxiv_enriched/{train,val,test}.jsonl``

    Parameters
    ----------
    dataset_path : str
        Root directory containing the raw ``arxiv/`` and ``pubmed/``
        sub-directories (i.e. the output of :func:`download_datasets`).
    top_k : int
        Number of oracle sentences per document (default: 5).
    """
    pubmed_path = os.path.join(dataset_path, "pubmed_enriched")
    arxiv_path = os.path.join(dataset_path, "arxiv_enriched")

    pubmed_raw = load_custom_dataset("pubmed", dataset_path)
    arxiv_raw  = load_custom_dataset("arxiv",  dataset_path)

    split_config: list[tuple[Dataset, str, str, int]] = [
        (pubmed_raw["train"],      pubmed_path, "train", TRAIN_SIZE),
        (pubmed_raw["validation"], pubmed_path, "val",   VAL_SIZE),
        (pubmed_raw["test"],       pubmed_path, "test",  TEST_SIZE),
        (arxiv_raw["train"],       arxiv_path,  "train", TRAIN_SIZE),
        (arxiv_raw["validation"],  arxiv_path,  "val",   VAL_SIZE),
        (arxiv_raw["test"],        arxiv_path,  "test",  TEST_SIZE),
    ]

    for split_ds, out_dir, split_name, max_n in split_config:
        generate_and_save_trimmed_split(split_ds, out_dir, split_name, max_n, top_k)


# 6. Cleaned dataset loading (for BERTScore evaluation)

def load_cleaned_enriched_split(jsonl_path: str) -> Dataset:
    """
    Load an enriched JSONL split, drop the auxiliary fields that cause type
    conflicts (``sorted_indices``, ``indices``, ``score``), and return only
    examples that have both ``oracle_summary`` and ``summary`` populated.

    This cleaned view is used for BERTScore oracle-vs-gold comparison.

    Parameters
    ----------
    jsonl_path : str
        Path to an enriched ``*.jsonl`` file.

    Returns
    -------
    Dataset
        Cleaned HuggingFace Dataset.
    """
    cleaned: list[dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for line in fh:
            ex = json.loads(line)
            ex.pop("sorted_indices", None)
            ex.pop("indices",        None)
            ex.pop("score",          None)
            if ex.get("oracle_summary") and ex.get("summary"):
                cleaned.append(ex)

    dataset = Dataset.from_list(cleaned)
    print(f"Loaded {len(dataset)} examples from {jsonl_path}")
    print(f"Columns: {dataset.column_names}")
    return dataset


# 7. BERTScore oracle-vs-gold evaluation

def bertscore_pairwise(
    oracle: list[str] | str,
    gold: list[str] | str,
) -> dict[str, float]:
    """
    Compute BERTScore precision, recall, and F1 for a single oracle–gold pair.

    Both inputs are joined into a single string before scoring so that the
    BERT model processes the complete summary as one unit.

    Parameters
    ----------
    oracle : list[str] | str
        TF-IDF extractive oracle summary (sentence list or concatenated string).
    gold : list[str] | str
        Human-authored abstractive gold summary.

    Returns
    -------
    dict[str, float]
        Dictionary with keys ``"Precision"``, ``"Recall"``, ``"F1"``.
    """
    pred = " ".join(oracle) if isinstance(oracle, list) else oracle
    ref  = " ".join(gold)   if isinstance(gold,   list) else gold
    precision, recall, f1 = bert_score_fn([pred], [ref], lang="en", verbose=False)
    return {
        "Precision": precision.item(),
        "Recall":    recall.item(),
        "F1":        f1.item(),
    }


def evaluate_bert_similarity(
    dataset_dict: DatasetDict,
    split_name: str,
    n_samples: int = 20,
) -> list[dict[str, float]]:
    """
    Evaluate TF-IDF oracle quality against gold summaries for the first
    *n_samples* examples in *split_name* using BERTScore.

    Parameters
    ----------
    dataset_dict : DatasetDict
        Enriched dataset containing both ``"oracle_summary"`` and
        ``"summary"`` columns.
    split_name : str
        Which split to evaluate (``"train"``, ``"validation"``, or ``"test"``).
    n_samples : int
        Number of examples to evaluate (default: 20).

    Returns
    -------
    list[dict[str, float]]
        Per-example BERTScore results.
    """
    print(f"🔍  Evaluating semantic similarity for '{split_name}' split …")
    split = dataset_dict[split_name]
    results: list[dict[str, float]] = []

    for i in range(min(n_samples, len(split))):
        scores = bertscore_pairwise(split[i]["oracle_summary"], split[i]["summary"])
        results.append(scores)

    return results


def summarize_bertscore_results(
    results: list[dict[str, float]],
    label: str,
) -> dict[str, float]:
    """
    Aggregate per-example BERTScore results into mean scores and print a
    formatted summary.

    Parameters
    ----------
    results : list[dict[str, float]]
        Output of :func:`evaluate_bert_similarity`.
    label : str
        Human-readable label printed in the console output.

    Returns
    -------
    dict[str, float]
        Mean ``{"Precision": …, "Recall": …, "F1": …}`` scores.
    """
    df = pd.DataFrame(results)
    summary: dict[str, float] = df.mean().to_dict()
    print(f"\n📈  {label} — BERTScore Mean:")
    for metric, value in summary.items():
        print(f"    {metric}: {value:.4f}")
    return summary


def run_bertscore_evaluation(dataset_path: str) -> None:
    """
    Run the full BERTScore oracle-vs-gold evaluation over all splits of both
    ArXiv and PubMed enriched datasets and print aggregated results.

    This function first loads the in-memory oracle datasets (via
    :func:`generate_tfidf_oracle_dataset`) and then evaluates them split by
    split.

    Parameters
    ----------
    dataset_path : str
        Root directory containing raw ``arxiv/`` and ``pubmed/`` data.
    """
    arxiv_raw  = load_custom_dataset("arxiv",  dataset_path)
    pubmed_raw = load_custom_dataset("pubmed", dataset_path)

    arxiv_oracle  = generate_tfidf_oracle_dataset(arxiv_raw,  top_k=5)
    pubmed_oracle = generate_tfidf_oracle_dataset(pubmed_raw, top_k=5)

    split_names = ["train", "validation", "test"]
    dataset_label_pairs = [
        (arxiv_oracle,  "ArXiv"),
        (pubmed_oracle, "PubMed"),
    ]

    for dataset, ds_label in dataset_label_pairs:
        for split in split_names:
            scores = evaluate_bert_similarity(dataset, split, n_samples=20)
            summarize_bertscore_results(scores, f"{ds_label} {split.capitalize()}")
