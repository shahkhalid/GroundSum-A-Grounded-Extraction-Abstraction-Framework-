"""
main.py
=======
Single entry point for the hallucination-mitigation long-document
summarization pipeline.

Usage
-----
Run the full end-to-end pipeline::

    python main.py --base_dir /data/thesis_project --stages all

Run individual stages::

    python main.py --base_dir /data/thesis_project --stages download
    python main.py --base_dir /data/thesis_project --stages build_datasets
    python main.py --base_dir /data/thesis_project --stages longformer
    python main.py --base_dir /data/thesis_project --stages baseline_distilbert
    python main.py --base_dir /data/thesis_project --stages baseline_t5
    python main.py --base_dir /data/thesis_project --stages rewrite
    python main.py --base_dir /data/thesis_project --stages evaluate

All stage names can be combined (comma-separated)::

    python main.py --base_dir /data/thesis_project --stages longformer,evaluate
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any

import pandas as pd

#  ensure src/ is importable when running from repo root 
sys.path.insert(0, os.path.dirname(__file__))

from src.utils import (
    DATASETS,
    TRAIN_SIZE,
    VAL_SIZE,
    TEST_SIZE,
    build_paths,
    inspect_jsonl,
    save_json,
    setup_logging,
)
from src.data_processing import (
    build_enriched_datasets,
    download_datasets,
    load_custom_dataset,
    run_bertscore_evaluation,
)
from src.models import (
    LongformerExtractor,
    SummaryGenerator,
    run_rewrite_pipeline,
    run_t5_abstractive_pipeline,
    train_distilbert_baselines,
)
from src.evaluation import (
    evaluate_all,
    evaluate_extractive_distilbert,
    evaluate_model,
    load_and_preprocess_test_dataset,
)


# 1.  CLI argument parsing

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the pipeline entry point.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Hallucination-mitigation long-document summarization pipeline. "
            "Trains Longformer + DistilBERT extractive models and a T5 "
            "abstractive model, then rewrites extractive summaries using a "
            "GPT-powered self-critique loop."
        )
    )

    #  Paths 
    parser.add_argument(
        "--base_dir",
        type=str,
        default="./project_data",
        help="Root directory for all data, models, and outputs.",
    )

    #  Stage selection 
    parser.add_argument(
        "--stages",
        type=str,
        default="all",
        help=(
            "Comma-separated list of stages to execute, or 'all' for the "
            "full pipeline.  Available stages: download, build_datasets, "
            "bertscore_eval, longformer, baseline_distilbert, baseline_t5, "
            "rewrite, evaluate, inspect."
        ),
    )

    #  Dataset options 
    parser.add_argument(
        "--datasets",
        type=str,
        default="arxiv_enriched",
        help=(
            "Comma-separated list of enriched dataset names to process "
            "(default: 'arxiv_enriched').  Use 'arxiv_enriched,pubmed_enriched' "
            "for both."
        ),
    )

    #  Longformer hyper-parameters 
    parser.add_argument(
        "--longformer_model",
        type=str,
        default="allenai/longformer-base-4096",
        help="Longformer model identifier.",
    )

    #  T5 hyper-parameters 
    parser.add_argument(
        "--t5_model",
        type=str,
        default="t5-base",
        help="T5 model identifier or local path.",
    )
    parser.add_argument(
        "--t5_max_input_length",
        type=int,
        default=32,
        help="Maximum encoder input length for T5 (default: 32; increase for production).",
    )
    parser.add_argument(
        "--t5_max_target_length",
        type=int,
        default=32,
        help="Maximum decoder target length for T5 (default: 32).",
    )
    parser.add_argument(
        "--t5_epochs",
        type=int,
        default=1,
        help="Number of T5 fine-tuning epochs (default: 1).",
    )
    parser.add_argument(
        "--t5_train_batch_size",
        type=int,
        default=1,
        help="Per-device training batch size for T5.",
    )
    parser.add_argument(
        "--t5_eval_batch_size",
        type=int,
        default=1,
        help="Per-device evaluation batch size for T5.",
    )

    #  General 
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed.",
    )
    parser.add_argument(
        "--test_size",
        type=int,
        default=2500,
        help="Number of test examples to evaluate (default: 2500).",
    )

    return parser.parse_args()


# 2.  Individual stage runners

def stage_download(paths: dict[str, str], logger: logging.Logger) -> None:
    """
    Download the raw ArXiv and PubMed JSONL datasets from Hugging Face.

    Parameters
    ----------
    paths : dict[str, str]
        Project path map produced by :func:`src.utils.build_paths`.
    logger : logging.Logger
    """
    logger.info("Stage: download")
    download_datasets(paths["dataset"])
    logger.info("Download complete.")


def stage_build_datasets(paths: dict[str, str], logger: logging.Logger) -> None:
    """
    Generate TF-IDF oracle annotations and write enriched JSONL splits to disk.

    Parameters
    ----------
    paths : dict[str, str]
    logger : logging.Logger
    """
    logger.info("Stage: build_datasets")
    build_enriched_datasets(paths["dataset"], top_k=5)
    logger.info("Enriched datasets built.")


def stage_bertscore_eval(paths: dict[str, str], logger: logging.Logger) -> None:
    """
    Evaluate semantic similarity between TF-IDF oracle and gold summaries
    using BERTScore.

    Parameters
    ----------
    paths : dict[str, str]
    logger : logging.Logger
    """
    logger.info("Stage: bertscore_eval")
    run_bertscore_evaluation(paths["dataset"])
    logger.info("BERTScore evaluation complete.")


def stage_longformer(
    paths: dict[str, str],
    datasets: list[str],
    test_size: int,
    logger: logging.Logger,
) -> None:
    """
    Train the Longformer + cross-attention extractor and generate extractive
    summaries on the test split.

    Per-dataset ROUGE results are collected and written to
    ``<results>/<dataset>_results.csv``.

    Parameters
    ----------
    paths : dict[str, str]
    datasets : list[str]
    test_size : int
    logger : logging.Logger
    """
    logger.info("Stage: longformer")
    all_results: dict[str, Any] = {}

    for dataset_name in datasets:
        logger.info("Processing dataset: %s", dataset_name)
        print(f"\n{'='*55}\nProcessing {dataset_name.upper()}\n{'='*55}")

        extractor = LongformerExtractor(
            dataset_name=dataset_name,
            drive_path=paths["base"],
            model_dir=paths["models"],
            checkpoints_path=paths["checkpoints"],
        )
        extractor.train()

        summary_fn = SummaryGenerator(
            model=extractor.model,
            tokenizer=extractor.tokenizer,
            save_path=extractor.summaries_path,
        )

        df = evaluate_model(
            dataset_name=dataset_name,
            model_fn=summary_fn,
            drive_path=paths["base"],
            save_summaries=True,
            test_size=test_size,
        )
        all_results[dataset_name] = df.mean()

        result_df = pd.DataFrame(all_results).T
        csv_path = os.path.join(paths["results"], f"{dataset_name}_results.csv")
        result_df.to_csv(csv_path)
        logger.info("%s results saved → %s", dataset_name, csv_path)
        print(result_df.round(3))

    logger.info("Longformer stage complete.")


def stage_baseline_distilbert(
    paths: dict[str, str],
    logger: logging.Logger,
) -> None:
    """
    Train the DistilBERT sentence-level baseline and evaluate it on both
    ArXiv and PubMed test splits.

    Parameters
    ----------
    paths : dict[str, str]
    logger : logging.Logger
    """
    from src.data_processing import load_custom_dataset
    from src.models import DISTILBERT_MODEL_NAME
    from transformers import DistilBertTokenizerFast
    from src.utils import TRAIN_SIZE, VAL_SIZE, TEST_SIZE, DROP_KEYS

    import json
    from datasets import DatasetDict, Dataset, Features, Sequence, Value

    logger.info("Stage: baseline_distilbert")

    #  Load and preprocess document-level data 
    def preprocess_example(ex: dict[str, Any]) -> dict[str, Any]:
        if isinstance(ex.get("text"), list):
            ex["text"] = " ".join(ex["text"])
        if isinstance(ex.get("summary"), list):
            ex["summary"] = " ".join(ex["summary"])
        raw = ex.get("oracle_summary")
        if raw is None:
            ex["oracle_summary"] = ""
        elif isinstance(raw, list):
            ex["oracle_summary"] = " ".join(raw)
        else:
            ex["oracle_summary"] = str(raw).strip()
        for k in DROP_KEYS:
            ex.pop(k, None)
        return ex

    def load_split(base_dir: str, split_fname: str) -> Dataset:
        path = os.path.join(base_dir, split_fname)
        data: list[dict[str, Any]] = []
        all_keys: set[str] = set()
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                ex = preprocess_example(json.loads(line))
                data.append(ex)
                all_keys |= ex.keys()
        if not data:
            return Dataset.from_dict({})

        def default_for(key: str) -> Any:
            return "" if key in ("text", "summary", "oracle_summary") else []

        columnar = {
            key: [ex.get(key, default_for(key)) for ex in data]
            for key in sorted(all_keys)
        }
        features: dict[str, Any] = {}
        for key, col in columnar.items():
            sample = next((v for v in col if v not in (None, "", [])), col[0])
            if isinstance(sample, list):
                elem_type = "int64" if sample and isinstance(sample[0], int) else "string"
                features[key] = Sequence(Value(elem_type))
            elif isinstance(sample, int):
                features[key] = Value("int64")
            else:
                features[key] = Value("string")
        return Dataset.from_dict(columnar, features=Features(features))

    def load_and_subsample(base_dir: str) -> DatasetDict:
        splits = {
            "train":      ("train.jsonl", TRAIN_SIZE),
            "validation": ("val.jsonl",   VAL_SIZE),
            "test":       ("test.jsonl",  TEST_SIZE),
        }
        ds_dict: dict[str, Dataset] = {}
        for split, (fname, size) in splits.items():
            ds = load_split(base_dir, fname)
            ds = ds.shuffle(seed=42).select(range(min(size, len(ds))))
            ds_dict[split] = ds
        dd = DatasetDict(ds_dict)
        return dd.rename_column("oracle_summary", "labels").remove_columns("summary")

    arxiv_dir  = os.path.join(paths["dataset"], "arxiv_enriched")
    pubmed_dir = os.path.join(paths["dataset"], "pubmed_enriched")

    arxiv_ds  = load_and_subsample(arxiv_dir)
    pubmed_ds = load_and_subsample(pubmed_dir)

    output_dir = os.path.join(paths["baselines"], "distilbert")
    arxiv_trainer, pubmed_trainer = train_distilbert_baselines(
        arxiv_ds, pubmed_ds, output_dir
    )

    tokenizer = DistilBertTokenizerFast.from_pretrained(DISTILBERT_MODEL_NAME)

    arxiv_scores = evaluate_extractive_distilbert(
        arxiv_trainer, tokenizer, arxiv_ds,
        output_dir=os.path.join(output_dir, "arxiv"),
    )
    pubmed_scores = evaluate_extractive_distilbert(
        pubmed_trainer, tokenizer, pubmed_ds,
        output_dir=os.path.join(output_dir, "pubmed"),
    )

    logger.info("ArXiv DistilBERT ROUGE: %s", arxiv_scores)
    logger.info("PubMed DistilBERT ROUGE: %s", pubmed_scores)
    logger.info("Stage: baseline_distilbert complete.")


def stage_baseline_t5(
    paths: dict[str, str],
    datasets: list[str],
    args: argparse.Namespace,
    logger: logging.Logger,
) -> None:
    """
    Fine-tune T5 on extractive predictions and evaluate on the test split.

    Parameters
    ----------
    paths : dict[str, str]
    datasets : list[str]
    args : argparse.Namespace
    logger : logging.Logger
    """
    logger.info("Stage: baseline_t5")

    # Map enriched dataset names back to raw names for the extractive dir.
    raw_name_map = {
        "arxiv_enriched":  "arxiv",
        "pubmed_enriched": "pubmed",
        "arxiv":           "arxiv",
        "pubmed":          "pubmed",
    }

    for dataset_name in datasets:
        raw_name = raw_name_map.get(dataset_name, dataset_name)
        run_t5_abstractive_pipeline(
            drive_path=paths["base"],
            dataset_name=raw_name,
            model_name_or_path=args.t5_model,
            output_subdir="abstractive_t5",
            max_input_length=args.t5_max_input_length,
            max_target_length=args.t5_max_target_length,
            per_device_train_batch_size=args.t5_train_batch_size,
            per_device_eval_batch_size=args.t5_eval_batch_size,
            num_train_epochs=args.t5_epochs,
            seed=args.seed,
        )

    logger.info("Stage: baseline_t5 complete.")


def stage_rewrite(
    paths: dict[str, str],
    logger: logging.Logger,
) -> None:
    """
    Run the GPT-powered self-critique abstractive rewrite on all extractive
    summaries produced by the Longformer pipeline.

    Parameters
    ----------
    paths : dict[str, str]
    logger : logging.Logger
    """
    logger.info("Stage: rewrite")
    run_rewrite_pipeline(paths["base"])
    logger.info("Stage: rewrite complete.")


def stage_evaluate(
    paths: dict[str, str],
    datasets: list[str],
    logger: logging.Logger,
) -> None:
    """
    Evaluate the abstractive rewrite summaries using BERTScore and GPT FactCC,
    saving results to ``abs_summaries/<dataset>/evaluation_results.json``.

    Parameters
    ----------
    paths : dict[str, str]
    datasets : list[str]
    logger : logging.Logger
    """
    logger.info("Stage: evaluate")
    evaluate_all(base_dir=paths["base"], datasets=datasets)
    logger.info("Stage: evaluate complete.")


def stage_inspect(
    paths: dict[str, str],
    datasets: list[str],
    logger: logging.Logger,
) -> None:
    """
    Print brief structural summaries of the generated prediction JSONL files.

    Parameters
    ----------
    paths : dict[str, str]
    datasets : list[str]
    logger : logging.Logger
    """
    logger.info("Stage: inspect")
    for dataset_name in datasets:
        raw_name = dataset_name.replace("_enriched", "")
        for subset in ["arxiv", "pubmed"]:
            pred_path = os.path.join(
                paths["baselines"], "distilbert", subset, "test_predictions.jsonl"
            )
            if os.path.exists(pred_path):
                inspect_jsonl(pred_path)
            else:
                logger.warning("Prediction file not found: %s", pred_path)
    logger.info("Stage: inspect complete.")


# 3.  Main orchestrator

def main() -> None:
    """
    Parse CLI arguments, build the project directory structure, configure
    logging, and execute the requested pipeline stages in order.

    Stage execution order when ``--stages all`` is specified:

    1. ``download``           — fetch raw datasets from Hugging Face Hub
    2. ``build_datasets``     — generate TF-IDF oracle annotations
    3. ``bertscore_eval``     — oracle-vs-gold BERTScore evaluation
    4. ``longformer``         — Longformer training + extractive inference
    5. ``baseline_distilbert``— DistilBERT sentence classifier baseline
    6. ``baseline_t5``        — T5 abstractive refinement baseline
    7. ``rewrite``            — GPT self-critique abstractive rewrite
    8. ``evaluate``           — BERTScore + FactCC on rewritten summaries
    9. ``inspect``            — JSONL structural sanity checks
    """
    args = parse_args()

    #  Resolve stages 
    all_stages = [
        "download",
        "build_datasets",
        "bertscore_eval",
        "longformer",
        "baseline_distilbert",
        "baseline_t5",
        "rewrite",
        "evaluate",
        "inspect",
    ]
    if args.stages.strip().lower() == "all":
        stages = all_stages
    else:
        stages = [s.strip() for s in args.stages.split(",")]

    #  Resolve datasets 
    active_datasets: list[str] = [
        d.strip() for d in args.datasets.split(",")
    ]

    #  Build project directory tree 
    paths = build_paths(args.base_dir)

    #  Configure logging 
    logger = setup_logging(paths["logs"], filename="main_pipeline.log")
    logger.info("Pipeline started.  Stages: %s", stages)
    logger.info("Base directory:    %s", args.base_dir)
    logger.info("Active datasets:   %s", active_datasets)

    #  Execute stages 
    if "download" in stages:
        stage_download(paths, logger)

    if "build_datasets" in stages:
        stage_build_datasets(paths, logger)

    if "bertscore_eval" in stages:
        stage_bertscore_eval(paths, logger)

    if "longformer" in stages:
        stage_longformer(paths, active_datasets, args.test_size, logger)

    if "baseline_distilbert" in stages:
        stage_baseline_distilbert(paths, logger)

    if "baseline_t5" in stages:
        stage_baseline_t5(paths, active_datasets, args, logger)

    if "rewrite" in stages:
        stage_rewrite(paths, logger)

    if "evaluate" in stages:
        stage_evaluate(paths, active_datasets, logger)

    if "inspect" in stages:
        stage_inspect(paths, active_datasets, logger)

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
