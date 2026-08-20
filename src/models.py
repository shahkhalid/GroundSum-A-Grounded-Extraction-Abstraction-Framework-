"""
src/models.py
=============
All model definitions, training pipelines, and prompt-engineering structures
used in the project:

* :class:`LongformerWithCrossAttention` — Longformer encoder augmented with a
  learnable global cross-attention head for token-level sentence selection.
* :class:`SavePretrainedCallback` — HuggingFace Trainer callback that
  persists the custom model's state dictionary at every checkpoint.
* :class:`SummaryGenerator` — Inference wrapper for the Longformer extractor.
* :class:`LongformerExtractor` — End-to-end training and loading manager for
  the Longformer model.
* :func:`rewrite_with_self_critique` — Four-stage GPT-powered fact-anchored
  abstractive rewrite (Pipeline 2).
* DistilBERT sentence-level binary classifier training utilities (baseline).
* T5 abstractive fine-tuning pipeline (baseline).
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

import evaluate
import nltk
import numpy as np
import openai
import torch
import torch.nn as nn
from datasets import DatasetDict, Dataset, Features, Sequence, Value
from nltk.tokenize import sent_tokenize
from openai import APIError, OpenAIError, RateLimitError
from openai import APITimeoutError as OpenAITimeout
from rouge_score import rouge_scorer
from torch.optim import AdamW
from transformers import (
    DataCollatorForTokenClassification,
    DataCollatorWithPadding,
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    EarlyStoppingCallback,
    EvalPrediction,
    LongformerModel,
    LongformerTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5ForConditionalGeneration,
    T5TokenizerFast,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

import backoff

from src.utils import (
    BATCH_SIZE,
    EPOCHS,
    GPT_MODEL,
    GPT_TEMPERATURE,
    GPT_TEMPERATURE_BASELINE,
    MAX_LEN,
    SUMMARY_LENGTH,
    TRAIN_SIZE,
    VAL_SIZE,
    safe_batch_decode,
    save_json,
    save_jsonl,
    load_dataset
)

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


# 1.  OpenAI call wrapper (shared by Pipeline 2 and Baseline abstractive)

@backoff.on_exception(
    backoff.expo,
    (RateLimitError, APIError, OpenAITimeout, OpenAIError),
    max_tries=3,
    jitter=None,
)
def call_openai_chat(
    messages: list[dict[str, str]],
    max_tokens: int = 512,
    temperature: float = GPT_TEMPERATURE,
    stop: list[str] | None = None,
    model: str = GPT_MODEL,
) -> str:
    """
    Thin wrapper around the OpenAI Chat Completions endpoint with exponential
    back-off retries on transient errors.

    Parameters
    ----------
    messages : list[dict[str, str]]
        The conversation history in OpenAI message format
        (``[{"role": …, "content": …}, …]``).
    max_tokens : int
        Maximum tokens to generate in the response (default: 512).
    temperature : float
        Sampling temperature; 0.0 is fully deterministic (default: 0.0).
    stop : list[str] | None
        Optional list of stop sequences.
    model : str
        OpenAI model identifier (default: ``"gpt-4o-mini"``).

    Returns
    -------
    str
        Stripped text content of the first response choice.
    """
    start = time.time()
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=stop,
    )
    latency = time.time() - start
    text: str = response.choices[0].message.content.strip()
    print(
        f"[OpenAI] model={model} | messages={len(messages)} | "
        f"resp_tokens≈{len(text.split())} | latency={latency:.2f}s"
    )
    return text


# 2.  Pipeline-2: Four-stage fact-anchored abstractive rewrite

def rewrite_with_self_critique(chunk: str) -> str:
    """
    Apply a four-stage GPT pipeline to produce a factually grounded
    abstractive rewrite of an extractive *chunk*.

    The stages are:

    1. **Fact extraction** — mine atomic (subject, predicate, object) triples
       from the source chunk.
    2. **Guided rewrite** — reconstruct the chunk using only the mined facts,
       in fluent academic prose.
    3. **Self-critique** — flag unsupported phrases in the rewritten sentence
       using a ``[?…?]`` bracketing convention.
    4. **Automated revision** — if any flags were introduced, resolve them
       using the extracted facts or replace with ``"unspecified"``.

    Parameters
    ----------
    chunk : str
        A token-bounded text segment produced by
        :func:`src.utils.chunk_by_tokens`.

    Returns
    -------
    str
        Factually revised abstractive rewrite of *chunk*.
    """
    # Stage 1: Fact extraction
    facts_prompt: list[dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You are a fact miner. List atomic facts from this text "
                "as triples (subject, predicate, object)."
            ),
        },
        {"role": "user", "content": chunk},
    ]
    facts = call_openai_chat(facts_prompt)

    # Stage 2: Guided rewrite
    rewrite_prompt: list[dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "Rewrite using only these facts in fluent academic style. "
                "Reuse key domain terms from the source. Keep sentences concise. "
                "Do not add or omit information."
            ),
        },
        {"role": "user", "content": facts},
    ]
    sentence = call_openai_chat(rewrite_prompt)

    # Stage 3: Self-critique
    critique_prompt: list[dict[str, str]] = [
        {"role": "system", "content": "You are a critical reviewer."},
        {
            "role": "user",
            "content": (
                f"Review this sentence. Flag any unsupported phrase in [?…?]:\n\n"
                f"{sentence}"
            ),
        },
    ]
    critiqued = call_openai_chat(critique_prompt)

    # Stage 4: Automated revision (only if flags were introduced)
    if "[?" in critiqued:
        revise_prompt: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "Resolve flagged phrases with source facts only; "
                    "if uncertain, replace with 'unspecified.'"
                ),
            },
            {
                "role": "user",
                "content": f"Facts:\n{facts}\n\nSentence with flags:\n{critiqued}",
            },
        ]
        sentence = call_openai_chat(revise_prompt)

    return sentence


def run_rewrite_pipeline(base_dir: str) -> None:
    """
    Iterate over all extractive summary ``.txt`` files in the ``summaries``
    sub-directory, apply :func:`rewrite_with_self_critique` chunk by chunk,
    and write the rewritten outputs to the ``abs_summaries`` sub-directory.

    Parameters
    ----------
    base_dir : str
        Project root directory.  Expected structure::

            <base_dir>/
              summaries/<dataset>/          ← input extractive summaries
              abs_summaries/<dataset>/
                test_summaries/             ← output abstractive summaries
    """
    from src.utils import chunk_by_tokens, DATASETS

    for dataset in DATASETS:
        src_dir = Path(base_dir) / "summaries" / dataset
        out_dir = Path(base_dir) / "abs_summaries" / dataset / "test_summaries"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Remove any stale outputs from a previous run.
        for stale in out_dir.glob("*.txt"):
            stale.unlink()

        for infile in sorted(src_dir.glob("*.txt"), key=lambda p: int(p.stem)):
            idx = infile.stem
            outfile = out_dir / f"{idx}.txt"
            text = infile.read_text(encoding="utf-8").strip()
            print(f"[{dataset}] Rewriting file {idx}.txt …")

            chunks = chunk_by_tokens(text)
            rewritten_chunks: list[str] = []
            for chunk in chunks:
                try:
                    rewritten_chunks.append(rewrite_with_self_critique(chunk))
                except Exception as exc:
                    print(f"Chunk rewrite failed: {exc}")
                    rewritten_chunks.append(chunk)

            output = " ".join(rewritten_chunks).replace("\n", " ").strip()
            outfile.write_text(output, encoding="utf-8")

        print(f"Abstractive rewriting complete for {dataset}.")


# 3.  Longformer: custom model and training infrastructure

class LongformerWithCrossAttention(nn.Module):
    """
    Longformer encoder augmented with a learnable global cross-attention head.

    Architecture:
    * A pre-trained ``allenai/longformer-base-4096`` backbone encodes the
      full input sequence (up to 4096 tokens) via sliding-window attention.
    * A single learnable ``global_query`` parameter acts as a compressed
      global context vector.
    * Multi-head cross-attention is applied between the per-token hidden
      states (queries) and the global context (keys / values), allowing
      every token to be re-scored in light of the document-level summary
      signal.
    * A linear classifier maps each attended token representation to a
      binary label (0 = non-summary, 1 = summary-relevant).

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier for the Longformer backbone.
    num_labels : int
        Number of classification labels (2 for binary sentence selection).
    num_heads : int
        Number of attention heads in the cross-attention layer (default: 8).
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        self.longformer = LongformerModel.from_pretrained(model_name)
        self.config = self.longformer.config
        hidden_size: int = self.config.hidden_size

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
        )
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.global_query = nn.Parameter(torch.randn(1, hidden_size))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass through the Longformer backbone, cross-attention layer,
        and token-level classifier.

        Parameters
        ----------
        input_ids : torch.Tensor
            Token ID tensor of shape ``(batch, seq_len)``.
        attention_mask : torch.Tensor
            Attention mask of shape ``(batch, seq_len)``.
        labels : torch.Tensor | None
            Optional integer label tensor of shape ``(batch, seq_len)``
            for supervised training.

        Returns
        -------
        dict[str, torch.Tensor]
            Always contains ``"logits"``.  Also contains ``"loss"`` when
            *labels* are provided.
        """
        outputs = self.longformer(input_ids, attention_mask=attention_mask)
        hidden_states: torch.Tensor = outputs.last_hidden_state  # (B, L, H)

        batch_size = hidden_states.size(0)
        # Expand the global query to the current batch size: (B, 1, H)
        global_context = self.global_query.unsqueeze(0).expand(batch_size, -1, -1)

        # MultiheadAttention expects (L, B, H) inputs.
        query = hidden_states.transpose(0, 1)       # (L, B, H)
        key   = global_context.transpose(0, 1)      # (1, B, H)
        value = global_context.transpose(0, 1)      # (1, B, H)

        cross_attn_output, _ = self.cross_attention(
            query=query, key=key, value=value
        )
        cross_attended_states = cross_attn_output.transpose(0, 1)  # (B, L, H)

        logits: torch.Tensor = self.classifier(cross_attended_states)  # (B, L, num_labels)

        loss: torch.Tensor | None = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )

        if loss is not None:
            return {"loss": loss, "logits": logits}
        return {"logits": logits}

    def save_pretrained(self, save_directory: str) -> None:
        """
        Persist the model state dictionary and Longformer config to disk.

        Parameters
        ----------
        save_directory : str
            Directory into which ``pytorch_model.bin`` and ``config.json``
            are written (created automatically if absent).
        """
        os.makedirs(save_directory, exist_ok=True)
        torch.save(
            self.state_dict(),
            os.path.join(save_directory, "pytorch_model.bin"),
        )
        with open(os.path.join(save_directory, "config.json"), "w") as fh:
            json.dump(self.config.to_dict(), fh)


class SavePretrainedCallback(TrainerCallback):
    """
    HuggingFace Trainer callback that invokes :meth:`model.save_pretrained`
    at every checkpoint save event.

    This is necessary because :class:`LongformerWithCrossAttention` is a
    plain ``nn.Module`` rather than a ``PreTrainedModel``, so the default
    Trainer save logic does not apply.
    """

    def on_save(
        self,
        args: TrainingArguments,
        state: Any,
        control: Any,
        model: LongformerWithCrossAttention | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Called by the Trainer when a checkpoint is saved.

        Parameters
        ----------
        args : TrainingArguments
        state : TrainerState
        control : TrainerControl
        model : LongformerWithCrossAttention | None
            The model being trained.
        """
        save_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        print(f"Custom callback: saving model checkpoint → {save_dir}")
        if model is not None:
            model.save_pretrained(save_dir)
        return control


class SummaryGenerator:
    """
    Inference wrapper for :class:`LongformerWithCrossAttention` that produces
    a multi-sentence extractive summary from a raw text string.

    Token-level softmax scores produced by the classifier are aggregated per
    sentence via mean pooling.  The top-*k* scoring sentences are selected and
    returned in their original document order.

    Optionally saves each generated summary as a numbered ``.txt`` file to
    support the abstractive rewrite pipeline.

    Parameters
    ----------
    model : LongformerWithCrossAttention
        Trained Longformer model.
    tokenizer : LongformerTokenizer
        Corresponding tokenizer.
    save_path : str | None
        If provided, generated summaries are written to
        ``<save_path>/<summary_id>.txt``.
    device : str
        PyTorch device string (default: ``"cuda"``).
    """

    def __init__(
        self,
        model: LongformerWithCrossAttention,
        tokenizer: LongformerTokenizer,
        save_path: str | None = None,
        device: str = "cuda",
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.save_path = save_path

    def __call__(self, text: str, summary_id: str | None = None) -> str:
        """
        Generate an extractive summary for *text*.

        Parameters
        ----------
        text : str
            Full document text (sentences will be detected automatically).
        summary_id : str | None
            If provided and :attr:`save_path` is set, the summary is saved as
            ``<save_path>/<summary_id>.txt``.

        Returns
        -------
        str
            Space-joined extractive summary of the top-:data:`SUMMARY_LENGTH`
            sentences.
        """
        inputs = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=4096,
            return_tensors="pt",
        ).to(self.device)

        self.model.eval()
        with torch.no_grad():
            raw_outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )

        logits = raw_outputs["logits"] if isinstance(raw_outputs, dict) else raw_outputs
        # Token-level probability of being a summary token (class 1).
        token_scores: np.ndarray = (
            torch.softmax(logits, dim=-1)[0, :, 1].cpu().numpy()
        )

        sents = sent_tokenize(text)
        sent_scores: list[float] = []
        pos = 0
        for sent in sents:
            token_ids = self.tokenizer(sent, add_special_tokens=False)["input_ids"]
            span_len = len(token_ids)
            sent_scores.append(float(np.mean(token_scores[pos : pos + span_len])))
            pos += span_len

        top_indices = np.argsort(sent_scores)[-SUMMARY_LENGTH:][::-1]
        summary = " ".join([sents[i] for i in sorted(top_indices)])

        if self.save_path and summary_id:
            os.makedirs(self.save_path, exist_ok=True)
            file_path = os.path.join(self.save_path, f"{summary_id}.txt")
            with open(file_path, "w", encoding="utf-8") as fh:
                fh.write(summary)
            print(f"Saved summary: {file_path}")

        return summary


def get_latest_checkpoint(checkpoint_path: str) -> str | None:
    """
    Scan *checkpoint_path* for directories named ``checkpoint-<step>`` and
    return the path of the latest one by global step number.

    Parameters
    ----------
    checkpoint_path : str
        Directory to search.

    Returns
    -------
    str | None
        Path to the latest checkpoint directory, or ``None`` if none exist.
    """
    checkpoint_dirs = [
        os.path.join(checkpoint_path, name)
        for name in os.listdir(checkpoint_path)
        if name.startswith("checkpoint-")
    ]
    if not checkpoint_dirs:
        return None

    def _extract_step(d: str) -> int:
        match = re.search(r"checkpoint-(\d+)", d)
        return int(match.group(1)) if match else -1

    return sorted(checkpoint_dirs, key=_extract_step)[-1]


class LongformerExtractor:
    """
    End-to-end manager for loading, training, and running inference with the
    :class:`LongformerWithCrossAttention` model on a single dataset.

    On initialisation the manager attempts to restore model weights in the
    following priority order:

    1. ``<model_dir>/<dataset_name>/final_model/pytorch_model.bin``
    2. Latest checkpoint under ``<checkpoints_dir>/<dataset_name>/``
    3. Fresh ``allenai/longformer-base-4096`` weights (no fine-tuning).

    Parameters
    ----------
    dataset_name : str
        One of ``"arxiv_enriched"`` or ``"pubmed_enriched"``.
    drive_path : str
        Project root directory.
    model_dir : str
        Directory for final saved models.
    checkpoints_path : str
        Directory for Trainer checkpoints.
    """

    LONGFORMER_BASE: str = "allenai/longformer-base-4096"

    def __init__(
        self,
        dataset_name: str,
        drive_path: str,
        model_dir: str,
        checkpoints_path: str,
    ) -> None:
        self.dataset_name = dataset_name
        self.drive_path = drive_path
        self.tokenizer: LongformerTokenizer = LongformerTokenizer.from_pretrained(
            self.LONGFORMER_BASE
        )
        self.model_path = os.path.join(model_dir, dataset_name)
        self.checkpoint_path = os.path.join(checkpoints_path, dataset_name)
        self.summaries_path = os.path.join(drive_path, "summaries", dataset_name)
        os.makedirs(self.summaries_path, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()

        # Populated during training for use in compute_metrics.
        self._val_dataset: Dataset | None = None
        self.summary_fn: SummaryGenerator | None = None

    def _load_model(self) -> LongformerWithCrossAttention:
        """
        Attempt to restore model weights from disk in priority order.

        Returns
        -------
        LongformerWithCrossAttention
            Model on the appropriate device.
        """
        model = LongformerWithCrossAttention(self.LONGFORMER_BASE, num_labels=2)
        final_bin = os.path.join(self.model_path, "final_model", "pytorch_model.bin")

        if os.path.exists(final_bin):
            print(f"Loading final model from {final_bin} …")
            state_dict = torch.load(final_bin, map_location=self.device)
            model.load_state_dict(state_dict, strict=False)
        elif os.path.exists(self.checkpoint_path):
            ckpt = get_latest_checkpoint(self.checkpoint_path)
            if ckpt:
                ckpt_bin = os.path.join(ckpt, "pytorch_model.bin")
                print(f"Loading checkpoint from {ckpt_bin} …")
                state_dict = torch.load(ckpt_bin, map_location=self.device)
                model.load_state_dict(state_dict, strict=False)
            else:
                print("No checkpoints found. Initialising fresh model.")
        else:
            print("No saved model found. Initialising fresh model.")

        return model.to(self.device)

    #  data helpers 

    def _load_and_preprocess_split(self, split_name: str) -> Dataset:
        """
        Load a JSONL split from the enriched dataset directory and return
        a HuggingFace Dataset with columnar layout.

        Parameters
        ----------
        split_name : str
            One of ``"train"``, ``"val"``, ``"test"``.

        Returns
        -------
        Dataset
        """
        from src.utils import load_jsonl

        file_path = os.path.join(
            self.drive_path, "dataset", self.dataset_name, f"{split_name}.jsonl"
        )
        records = load_jsonl(file_path)
        if not records:
            return Dataset.from_dict({})
        columnar = {key: [ex.get(key, "") for ex in records] for key in records[0]}
        return Dataset.from_dict(columnar)

    def _create_labels_validation(
        self, example: dict[str, Any], max_length: int = 4096
    ) -> dict[str, Any]:
        """
        Tokenise a single validation example and attach an all-zero label
        sequence together with the raw reference summary string.

        Parameters
        ----------
        example : dict
        max_length : int
            Maximum token sequence length (default: 4096).

        Returns
        -------
        dict
            Tokenised example with ``"labels"`` and ``"ref_summary"`` keys.
        """
        input_text = (
            " ".join(example["text"])
            if isinstance(example["text"], list)
            else example.get("text") or ""
        )
        tokenized = self.tokenizer(
            input_text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        tokenized["labels"] = [0] * len(tokenized["input_ids"])

        raw_oracle = example.get("oracle_summary", "")
        ref_summary = (
            " ".join(raw_oracle) if isinstance(raw_oracle, list) else raw_oracle or ""
        )
        tokenized["ref_summary"] = ref_summary
        tokenized["labels"]         = np.asarray(tokenized["labels"],         dtype=np.int64)
        tokenized["input_ids"]      = np.asarray(tokenized["input_ids"],      dtype=np.int64)
        tokenized["attention_mask"] = np.asarray(tokenized["attention_mask"], dtype=np.int64)
        return tokenized

    #  training 

    def train(self) -> None:
        """
        Execute the complete Longformer training loop.

        Steps:
        1. Load and filter train / validation splits from the enriched JSONL.
        2. Sub-sample to :data:`TRAIN_SIZE` / :data:`VAL_SIZE` examples.
        3. Tokenise and create token-level binary labels aligned to oracle
           sentence boundaries.
        4. Configure a HuggingFace Trainer with ROUGE-based evaluation and
           checkpoint-save callbacks.
        5. Resume from the latest checkpoint when available; otherwise train
           from scratch.
        6. Save the final model to ``<model_path>/final_model/``.
        """
        raw_dataset = {
            "train": self._load_and_preprocess_split("train"),
            "val":   self._load_and_preprocess_split("val"),
        }

        # Filter out examples with no oracle summary.
        raw_dataset["train"] = raw_dataset["train"].filter(
            lambda ex: bool(ex.get("oracle_summary"))
        )
        raw_dataset["train"] = raw_dataset["train"].filter(
            lambda ex: ex.get("indices") is not None
        )
        raw_dataset["train"] = (
            raw_dataset["train"]
            .shuffle(seed=42)
            .select(range(min(TRAIN_SIZE, len(raw_dataset["train"]))))
        )
        raw_dataset["val"] = (
            raw_dataset["val"]
            .shuffle(seed=42)
            .select(range(min(VAL_SIZE, len(raw_dataset["val"]))))
        )

        print(f"Train size (after filter): {len(raw_dataset['train'])}")
        print(f"Val size (after filter):   {len(raw_dataset['val'])}")

        #  tokenise training split 
        def create_labels(examples: dict[str, list]) -> dict[str, Any]:
            """Batched tokenisation with token-level oracle label alignment."""
            texts_joined = [
                " ".join(t) if isinstance(t, list) else t or ""
                for t in examples["text"]
            ]
            summaries_raw = examples.get("oracle_summary", [])
            summaries_joined: list[str] = []
            valid_indices: list[int] = []

            for i, summ in enumerate(summaries_raw):
                if summ:
                    summaries_joined.append(
                        " ".join(summ) if isinstance(summ, list) else summ
                    )
                    valid_indices.append(i)

            if not valid_indices:
                print("Skipping batch — all oracle summaries are missing.")
                return {}

            texts_joined = [texts_joined[i] for i in valid_indices]

            try:
                tokenized = self.tokenizer(
                    texts_joined,
                    truncation=True,
                    max_length=4096,
                    padding="max_length",
                )
            except Exception as exc:
                print(f"Tokeniser error: {exc}")
                batch_size = len(texts_joined)
                return {
                    "input_ids":      [[] for _ in range(batch_size)],
                    "attention_mask": [[] for _ in range(batch_size)],
                    "labels":         [[] for _ in range(batch_size)],
                }

            labels: list[list[int]] = []
            for doc_text, oracle_summary, token_ids in zip(
                texts_joined, summaries_joined, tokenized["input_ids"]
            ):
                sentences = sent_tokenize(doc_text)
                oracle_sents = set(sent_tokenize(oracle_summary))
                sentence_length = len(token_ids) // max(len(sentences), 1)
                label = [0] * len(token_ids)

                for sent_idx, sent in enumerate(sentences):
                    start_idx = sent_idx * sentence_length
                    if sent.strip() in oracle_sents and start_idx < len(label):
                        label[start_idx] = 1

                labels.append(label)

            tokenized["labels"]         = np.asarray(labels,                               dtype=np.int64)
            tokenized["input_ids"]      = np.asarray(tokenized["input_ids"],               dtype=np.int64)
            tokenized["attention_mask"] = np.asarray(tokenized["attention_mask"],           dtype=np.int64)
            return tokenized

        remove_cols_train = [
            col for col in ["text", "indices", "summary", "score", "sorted_indices"]
            if col in raw_dataset["train"].column_names
        ]
        remove_cols_val = [
            col for col in ["text", "indices", "summary", "score", "sorted_indices"]
            if col in raw_dataset["val"].column_names
        ]

        processed_train = raw_dataset["train"].map(
            create_labels,
            batched=True,
            remove_columns=remove_cols_train,
        )
        processed_train = processed_train.remove_columns(
            [c for c in ["oracle_summary"] if c in processed_train.column_names]
        )
        processed_train.set_format(
            type="torch", columns=["input_ids", "attention_mask", "labels"]
        )
        print(f"Processed train size: {len(processed_train)}")

        processed_val = raw_dataset["val"].map(
            lambda ex: self._create_labels_validation(ex),
            batched=False,
            remove_columns=remove_cols_val,
        )
        processed_val = processed_val.remove_columns(
            [c for c in ["oracle_summary"] if c in processed_val.column_names]
        )
        processed_val.set_format(
            type="torch", columns=["input_ids", "attention_mask", "labels"]
        )
        print(f"Processed val size:   {len(processed_val)}")

        # Store for use in compute_metrics closure.
        self._val_dataset = raw_dataset["val"]
        self.summary_fn = SummaryGenerator(
            model=self.model, tokenizer=self.tokenizer, save_path=None
        )

        #  ROUGE compute_metrics 
        def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
            scorer = rouge_scorer.RougeScorer(
                ["rouge1", "rouge2", "rougeL"], use_stemmer=True
            )
            total_r1 = total_r2 = total_rL = 0.0
            count = 0

            assert self._val_dataset is not None
            assert self.summary_fn is not None

            for ex in self._val_dataset:
                text = (
                    " ".join(ex["text"])
                    if isinstance(ex["text"], list)
                    else ex.get("text") or ""
                )
                oracle = ex.get("oracle_summary", "")
                if isinstance(oracle, list):
                    oracle = " ".join(oracle)

                pred = self.summary_fn(text).strip()
                if pred and oracle.strip():
                    sc = scorer.score(oracle.strip(), pred)
                    total_r1 += sc["rouge1"].fmeasure
                    total_r2 += sc["rouge2"].fmeasure
                    total_rL += sc["rougeL"].fmeasure
                    count += 1

            if count == 0:
                return {"rouge1_f": 0.0, "rouge2_f": 0.0, "rougeL_f": 0.0}
            return {
                "rouge1_f": total_r1 / count,
                "rouge2_f": total_r2 / count,
                "rougeL_f": total_rL / count,
            }

        #  Trainer setup 
        checkpoint_interval = 500
        training_args = TrainingArguments(
            output_dir=self.checkpoint_path,
            eval_strategy="steps",
            save_strategy="steps",
            save_steps=checkpoint_interval,
            logging_strategy="steps",
            logging_steps=checkpoint_interval,
            logging_dir="./logs",
            learning_rate=3e-5,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            num_train_epochs=3,
            load_best_model_at_end=True,
            metric_for_best_model="rouge1_f",
            greater_is_better=True,
            fp16=True,
            save_total_limit=2,
            report_to="none",
            remove_unused_columns=False,
            resume_from_checkpoint=True,
        )

        data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=processed_train,
            eval_dataset=processed_val,
            compute_metrics=compute_metrics,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            callbacks=[SavePretrainedCallback()],
        )
        trainer.label_names = ["labels"]

        #  Resume or start 
        latest_ckpt: str | None = None
        if os.path.exists(self.checkpoint_path):
            latest_ckpt = get_latest_checkpoint(self.checkpoint_path)
            if latest_ckpt:
                print(f"Resuming from checkpoint: {latest_ckpt}")
            else:
                print("No valid checkpoints found. Starting from scratch.")
        else:
            print("Checkpoint path does not exist. Starting from scratch.")

        try:
            trainer.train(resume_from_checkpoint=latest_ckpt)
        except KeyboardInterrupt:
            print("Training interrupted. Saving interrupted model …")
            trainer.save_model(os.path.join(self.model_path, "interrupted"))
            raise
        except Exception as exc:
            print(f"Training error: {exc}")
            raise

        print("Training complete. Saving final model …")
        final_path = os.path.join(self.model_path, "final_model")
        self.model.save_pretrained(final_path)
        print(f"Final model saved → {final_path}")


# 4.  Baseline: DistilBERT sentence-level binary classifier

DISTILBERT_MODEL_NAME: str = "distilbert-base-uncased"


def build_sentence_dataset(
    ds_dict: DatasetDict,
) -> DatasetDict:
    """
    Flatten a document-level dataset into a sentence-level classification
    dataset.

    Each document is tokenised into sentences via NLTK.  A sentence receives
    label ``1`` if it appears verbatim in the oracle summary, else ``0``.

    Parameters
    ----------
    ds_dict : DatasetDict
        Document-level dataset with ``"text"`` (str) and ``"labels"`` (str,
        oracle summary) columns.

    Returns
    -------
    DatasetDict
        Sentence-level dataset with ``"sentence"`` and ``"label"`` columns.
    """
    def _flatten(split_ds: Dataset) -> Dataset:
        examples: list[dict[str, Any]] = []
        for doc in split_ds:
            sents = nltk.sent_tokenize(doc["text"])
            gold = set(nltk.sent_tokenize(doc["labels"]))
            for sent in sents:
                examples.append({"sentence": sent, "label": int(sent in gold)})
        return Dataset.from_list(examples)

    return DatasetDict(
        {split: _flatten(ds_dict[split]) for split in ds_dict}
    )


def get_distilbert_trainer(
    tokenized_ds: DatasetDict,
    output_dir: str,
    tokenizer: DistilBertTokenizerFast,
) -> Trainer:
    """
    Construct a HuggingFace Trainer for DistilBERT binary sentence
    classification.

    Evaluation metric is ROC-AUC, computed from softmax probabilities of the
    positive class.

    Parameters
    ----------
    tokenized_ds : DatasetDict
        Tokenised sentence dataset with ``"input_ids"``, ``"attention_mask"``,
        and ``"label"`` columns.
    output_dir : str
        Directory for checkpoints and logs.
    tokenizer : DistilBertTokenizerFast
        Tokenizer (also used by the data collator).

    Returns
    -------
    Trainer
        Configured HuggingFace Trainer ready for ``.train()``.
    """

    def compute_metrics(pred: EvalPrediction) -> dict[str, float]:
        logits, labels = pred
        probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
        roc_auc_metric = evaluate.load("roc_auc")
        return {
            "roc_auc": roc_auc_metric.compute(
                prediction_scores=probs, references=labels
            )["roc_auc"]
        }

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        save_total_limit=2,
        metric_for_best_model="roc_auc",
        load_best_model_at_end=True,
        report_to="none",
    )

    model = DistilBertForSequenceClassification.from_pretrained(
        DISTILBERT_MODEL_NAME, num_labels=2
    )
    data_collator = DataCollatorWithPadding(tokenizer)

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )


def train_distilbert_baselines(
    arxiv_ds: DatasetDict,
    pubmed_ds: DatasetDict,
    output_dir: str,
) -> tuple[Trainer, Trainer]:
    """
    Train DistilBERT sentence-level classifiers on both ArXiv and PubMed
    datasets.

    Parameters
    ----------
    arxiv_ds : DatasetDict
        Pre-processed ArXiv dataset (document level; string ``"text"`` and
        ``"labels"`` columns).
    pubmed_ds : DatasetDict
        Pre-processed PubMed dataset.
    output_dir : str
        Root output directory; sub-directories ``"arxiv"`` and ``"pubmed"``
        are created automatically.

    Returns
    -------
    tuple[Trainer, Trainer]
        Trained ArXiv and PubMed Trainer instances.
    """
    tokenizer = DistilBertTokenizerFast.from_pretrained(DISTILBERT_MODEL_NAME)

    def tokenize_fn(batch: dict[str, list]) -> dict[str, Any]:
        return tokenizer(
            batch["sentence"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
        )

    # Build sentence-level datasets.
    arxiv_sent  = build_sentence_dataset(arxiv_ds)
    pubmed_sent = build_sentence_dataset(pubmed_ds)

    # Tokenise.
    arxiv_tok  = arxiv_sent.map(tokenize_fn, batched=True, remove_columns=["sentence"])
    pubmed_tok = pubmed_sent.map(tokenize_fn, batched=True, remove_columns=["sentence"])
    arxiv_tok.set_format("torch")
    pubmed_tok.set_format("torch")

    # Train ArXiv.
    arxiv_trainer = get_distilbert_trainer(
        arxiv_tok, os.path.join(output_dir, "arxiv"), tokenizer
    )
    last_ckpt = get_last_checkpoint(arxiv_trainer.args.output_dir)
    if last_ckpt:
        print(f"⏩  Resuming ArXiv DistilBERT training from {last_ckpt}")
    arxiv_trainer.train(resume_from_checkpoint=last_ckpt)

    # Train PubMed.
    pubmed_trainer = get_distilbert_trainer(
        pubmed_tok, os.path.join(output_dir, "pubmed"), tokenizer
    )
    last_ckpt = get_last_checkpoint(pubmed_trainer.args.output_dir)
    if last_ckpt:
        print(f"⏩  Resuming PubMed DistilBERT training from {last_ckpt}")
    pubmed_trainer.train(resume_from_checkpoint=last_ckpt)

    return arxiv_trainer, pubmed_trainer


# 5.  Baseline: T5 abstractive fine-tuning

def load_and_split_extractive_preds(
    input_file: str,
    seed: int = 42,
    n_total: int = 2500,
) -> DatasetDict:
    """
    Load a JSONL file of extractive predictions (output of the Longformer or
    DistilBERT pipeline) and split it into fixed train / val / test folds.

    Splits:
    * Train:      indices 0 – 1499 (1 500 examples)
    * Validation: indices 1500 – 1999 (500 examples)
    * Test:       indices 2000 – 2499 (500 examples)

    Parameters
    ----------
    input_file : str
        Path to ``test_predictions.jsonl`` produced by the extractive pipeline.
    seed : int
        Random seed for shuffling (default: 42).
    n_total : int
        Total number of examples to load (default: 2500).

    Returns
    -------
    DatasetDict
        ``"train"``, ``"validation"``, ``"test"`` splits.
    """
    ds = load_dataset("json", data_files=input_file, split="train")
    ds = ds.select(range(min(n_total, len(ds)))).shuffle(seed=seed)

    return DatasetDict({
        "train":      ds.select(range(1500)),
        "validation": ds.select(range(1500, 2000)),
        "test":       ds.select(range(2000, min(2500, len(ds)))),
    })


def preprocess_for_t5(
    datasets: DatasetDict,
    tokenizer: T5TokenizerFast,
    max_input_length: int = 32,
    max_target_length: int = 32,
) -> DatasetDict:
    """
    Tokenise extractive prediction / reference pairs for T5 seq-to-seq training.

    The input sequence is the extractive ``"prediction"`` and the target is the
    ``"reference"`` (gold summary).  Both are truncated and padded to fixed
    lengths for efficient batching.

    Parameters
    ----------
    datasets : DatasetDict
        Dataset with ``"prediction"`` and ``"reference"`` string columns.
    tokenizer : T5TokenizerFast
        T5 tokenizer.
    max_input_length : int
        Maximum input token length (default: 32; increase for real runs).
    max_target_length : int
        Maximum target token length (default: 32; increase for real runs).

    Returns
    -------
    DatasetDict
        Tokenised dataset with ``"input_ids"``, ``"attention_mask"``, and
        ``"labels"`` columns.
    """

    def _prep(examples: dict[str, list]) -> dict[str, Any]:
        model_inputs = tokenizer(
            examples["prediction"],
            max_length=max_input_length,
            truncation=True,
            padding="max_length",
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["reference"],
                max_length=max_target_length,
                truncation=True,
                padding="max_length",
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return datasets.map(
        _prep, batched=True, remove_columns=["prediction", "reference"]
    )


def run_t5_abstractive_pipeline(
    drive_path: str,
    dataset_name: str = "arxiv",
    model_name_or_path: str = "t5-base",
    output_subdir: str = "abstractive_t5",
    max_input_length: int = 32,
    max_target_length: int = 32,
    per_device_train_batch_size: int = 1,
    per_device_eval_batch_size: int = 1,
    num_train_epochs: int = 1,
    save_total_limit: int = 3,
    seed: int = 42,
) -> None:
    """
    Fine-tune T5 on extractive prediction → gold summary pairs and evaluate
    using BERTScore and GPT-based FactCC.

    Outputs are saved to ``<drive_path>/<output_subdir>/<dataset_name>/``:
    * ``metrics/test_metrics.json``  — BERTScore + FactCC results.
    * ``summaries/test_abstractive.jsonl`` — per-example predictions.

    Parameters
    ----------
    drive_path : str
        Project root directory.
    dataset_name : str
        Dataset to process (``"arxiv"`` or ``"pubmed"``).
    model_name_or_path : str
        HuggingFace model ID or local path for T5 (default: ``"t5-base"``).
    output_subdir : str
        Sub-directory under *drive_path* for all T5 outputs.
    max_input_length : int
        Maximum encoder input tokens.
    max_target_length : int
        Maximum decoder target tokens.
    per_device_train_batch_size : int
    per_device_eval_batch_size : int
    num_train_epochs : int
    save_total_limit : int
    seed : int
    """
    from src.evaluation import compute_factcc_via_gpt

    set_seed(seed)

    extractive_dir = os.path.join(drive_path, "extractive")
    input_file = os.path.join(extractive_dir, dataset_name, "test_predictions.jsonl")
    output_dir = os.path.join(drive_path, output_subdir, dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"Running T5 abstractive pipeline on {dataset_name.upper()}")
    print(f"{'='*55}")

    tokenizer = T5TokenizerFast.from_pretrained(model_name_or_path)
    raw_datasets = load_and_split_extractive_preds(input_file, seed=seed)
    tokenized = preprocess_for_t5(
        raw_datasets, tokenizer, max_input_length, max_target_length
    )

    model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)
    bertscore_metric = evaluate.load("bertscore")

    def compute_bertscore_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
        preds, labels = eval_pred
        decoded_preds   = safe_batch_decode(preds,   tokenizer)
        labels_clean    = np.where(labels == -100, tokenizer.pad_token_id, labels)
        decoded_labels  = safe_batch_decode(labels_clean, tokenizer)
        bs = bertscore_metric.compute(
            predictions=decoded_preds, references=decoded_labels, lang="en"
        )
        return {
            "bertscore_precision": float(np.mean(bs["precision"])),
            "bertscore_recall":    float(np.mean(bs["recall"])),
            "bertscore_f1":        float(np.mean(bs["f1"])),
        }

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=save_total_limit,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        predict_with_generate=True,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=50,
        seed=seed,
        num_train_epochs=num_train_epochs,
        metric_for_best_model="bertscore_f1",
        load_best_model_at_end=False,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_bertscore_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )

    trainer.train()

    #  Test-set evaluation 
    test_out = trainer.predict(tokenized["test"])
    test_preds = safe_batch_decode(test_out.predictions, tokenizer)
    test_refs  = raw_datasets["test"]["reference"]

    bs_test = bertscore_metric.compute(
        predictions=test_preds, references=test_refs, lang="en"
    )
    fc_test = compute_factcc_via_gpt(test_refs, test_preds)

    test_metrics: dict[str, float] = {
        "bertscore_precision": float(np.mean(bs_test["precision"])),
        "bertscore_recall":    float(np.mean(bs_test["recall"])),
        "bertscore_f1":        float(np.mean(bs_test["f1"])),
        **fc_test,
    }

    metrics_dir   = os.path.join(output_dir, "metrics")
    summaries_dir = os.path.join(output_dir, "summaries")
    os.makedirs(metrics_dir,   exist_ok=True)
    os.makedirs(summaries_dir, exist_ok=True)

    save_json(test_metrics, os.path.join(metrics_dir, "test_metrics.json"))
    save_jsonl(
        [
            {"input": inp, "reference": ref, "prediction": pred}
            for inp, ref, pred in zip(
                raw_datasets["test"]["prediction"], test_refs, test_preds
            )
        ],
        os.path.join(summaries_dir, "test_abstractive.jsonl"),
    )

    print(f"\nFinished {dataset_name.upper()} — outputs in {output_dir}")
    print(json.dumps(test_metrics, indent=2))
