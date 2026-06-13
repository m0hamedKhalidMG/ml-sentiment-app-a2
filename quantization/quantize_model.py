"""Quantize a causal LM using GPTQ for efficient serving.

This script handles quantization. Your task is to add post-quantization
evaluation and MLflow logging.
"""

from __future__ import annotations

import argparse
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def quantize_gptq(
    model_name: str,
    output_dir: str,
    bits: int = 4,
    group_size: int = 128,
    dataset_name: str = "wikitext",
    num_samples: int = 48,
) -> None:
    """Quantize a model using GPTQ.

    Args:
        model_name: HuggingFace model name or local path.
        output_dir: Directory to save the quantized model.
        bits: Quantization bit width (4 or 8).
        group_size: Group size for quantization.
        dataset_name: Calibration dataset name.
        num_samples: Number of calibration samples.
    """
    from gptqmodel import GPTQModel, QuantizeConfig

    print(f"Quantizing {model_name} with GPTQ ({bits}-bit)...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    quantize_config = QuantizeConfig(
        bits=bits,
        group_size=group_size,
        desc_act=False,
        sym=True,
    )

    model = GPTQModel.load(
        model_name,
        quantize_config=quantize_config,
    )

    # Prepare calibration data
    from datasets import load_dataset

    config = "wikitext-2-raw-v1" if dataset_name == "wikitext" or dataset_name == "Salesforce/wikitext" else None
    dataset = load_dataset("Salesforce/wikitext", config, split=f"train[:{num_samples * 4}]")
    # gptqmodel expects raw strings; filter out empty lines
    calibration_data = [text for text in dataset["text"] if text.strip()][:num_samples]

    start_time = time.time()
    model.quantize(calibration_data)
    quantize_time = time.time() - start_time

    model.save_quantized(output_dir)
    tokenizer.save_pretrained(output_dir)

    # GPTQ requires float16; patch config.json if the source model used bfloat16
    import json
    config_path = os.path.join(output_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        if config.get("dtype") == "bfloat16" or config.get("torch_dtype") == "bfloat16":
            config["dtype"] = "float16"
            config["torch_dtype"] = "float16"
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
            print("  Patched config.json: dtype bfloat16 → float16 (required by GPTQ)")

    # Newer transformers expects extra_special_tokens as a dict, not a list
    tok_config_path = os.path.join(output_dir, "tokenizer_config.json")
    if os.path.exists(tok_config_path):
        with open(tok_config_path) as f:
            tok_config = json.load(f)
        tokens = tok_config.get("extra_special_tokens", [])
        if isinstance(tokens, list):
            tok_config["extra_special_tokens"] = {t: t for t in tokens}
            with open(tok_config_path, "w") as f:
                json.dump(tok_config, f, indent=2)
            print("  Patched tokenizer_config.json: extra_special_tokens list → dict")

    print(f"Quantization complete in {quantize_time:.1f}s")
    print(f"Quantized model saved to {output_dir}")


def compare_model_sizes(original_path: str, quantized_path: str) -> dict:
    """Compare model sizes before and after quantization."""
    def get_dir_size(path: str) -> int:
        total = 0
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total += os.path.getsize(fp)
        return total

    original_size = get_dir_size(original_path)
    quantized_size = get_dir_size(quantized_path)

    return {
        "original_size_gb": round(original_size / 1024**3, 2),
        "quantized_size_gb": round(quantized_size / 1024**3, 2),
        "compression_ratio": round(original_size / quantized_size, 2) if quantized_size > 0 else 0,
        "size_reduction_pct": round((1 - quantized_size / original_size) * 100, 1) if original_size > 0 else 0,
    }


def evaluate_quantized_model(
    model_path: str,
    eval_dataset_path: str,
    mlflow_uri: str | None = None,
    experiment_name: str | None = None,
) -> dict:
    """Evaluate a quantized model on the eval dataset and log metrics to MLflow.

    Args:
        model_path: Path to the quantized model directory.
        eval_dataset_path: Path to the evaluation CSV dataset.
        mlflow_uri: MLflow tracking URI. If provided, logs metrics/model to MLflow.
        experiment_name: Name of the experiment.

    Returns:
        Dict with evaluation metrics.
    """
    import csv

    from gptqmodel import GPTQModel

    print(f"Evaluating quantized model at {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = GPTQModel.load(model_path)

    # Load eval dataset
    samples = []
    with open(eval_dataset_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append({"text": row["text"], "label": row["label"]})

    y_true = [s["label"] for s in samples]
    y_pred = []

    prompt_template = (
        "Classify the sentiment of the following text as positive, negative, or neutral.\n"
        "Text: {text}\nSentiment:"
    )

    start_time = time.time()
    for sample in samples:
        prompt = prompt_template.format(text=sample["text"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        label = generated.strip().lower()
        if "positive" in label:
            y_pred.append("positive")
        elif "negative" in label:
            y_pred.append("negative")
        else:
            y_pred.append("neutral")

    elapsed = time.time() - start_time

    # Compute metrics
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "total_inference_time_s": elapsed,
        "avg_latency_s": elapsed / len(samples) if samples else 0,
    }

    # Model size
    model_size = compare_model_sizes(
        os.path.expanduser("~/.cache/huggingface/hub"),
        model_path,
    )
    metrics["quantized_size_gb"] = model_size["quantized_size_gb"]

    # Derived convenience metrics matching the few-shot naming conventions
    metrics["f1_score"] = metrics["f1_macro"]
    metrics["latency_ms"] = metrics["avg_latency_s"] * 1000
    metrics["model_size_mb"] = metrics["quantized_size_gb"] * 1024

    print(f"  Quantized model eval: accuracy={metrics['accuracy']:.3f}, f1={metrics['f1_macro']:.3f}")
    print(f"  Avg latency: {metrics['avg_latency_s']:.3f}s, model size: {metrics['quantized_size_gb']:.2f} GB")

    # Log all metrics and parameters to MLflow
    if mlflow_uri:
        import mlflow

        mlflow.set_tracking_uri(mlflow_uri)
        if experiment_name:
            mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name="quantization_eval"):
            # --- Parameters ---
            mlflow.log_param("experiment_type", "quantization_eval")
            mlflow.log_param("vllm_model_path", os.path.abspath(model_path))
            mlflow.log_param("bits", 4)
            mlflow.log_param("method", "gptq")
            mlflow.log_param("model_name", "Qwen/Qwen2.5-1.5B")
            mlflow.log_param("num_samples", 48)

            # --- Metrics ---
            mlflow.log_metric("accuracy", metrics["accuracy"])
            mlflow.log_metric("f1_score", metrics["f1_score"])          # used by registration.py
            mlflow.log_metric("f1_macro", metrics["f1_macro"])
            mlflow.log_metric("precision_macro", metrics["precision_macro"])
            mlflow.log_metric("recall_macro", metrics["recall_macro"])
            mlflow.log_metric("latency_ms", metrics["latency_ms"])
            mlflow.log_metric("model_size_mb", metrics["model_size_mb"])
            mlflow.log_metric("avg_latency_s", metrics["avg_latency_s"])
            mlflow.log_metric("total_inference_time_s", metrics["total_inference_time_s"])
            mlflow.log_metric("quantized_size_gb", metrics["quantized_size_gb"])

            # --- Artifacts: predictions CSV ---
            import csv as csv_module
            predictions_path = "quantization_predictions.csv"
            with open(predictions_path, "w", newline="") as csvfile:
                writer = csv_module.DictWriter(csvfile, fieldnames=["text", "true_label", "predicted_label"])
                writer.writeheader()
                for sample, pred in zip(samples, y_pred):
                    writer.writerow({
                        "text": sample["text"],
                        "true_label": sample["label"],
                        "predicted_label": pred,
                    })
            mlflow.log_artifact(predictions_path)

            # --- Artifacts: confusion matrix plot ---
            try:
                import matplotlib.pyplot as plt
                import numpy as np
                from sklearn.metrics import confusion_matrix

                labels = ["positive", "negative", "neutral"]
                cm = confusion_matrix(y_true, y_pred, labels=labels)
                fig, ax = plt.subplots(figsize=(6, 5))
                im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
                plt.colorbar(im, ax=ax)
                ax.set(
                    xticks=np.arange(len(labels)),
                    yticks=np.arange(len(labels)),
                    xticklabels=labels,
                    yticklabels=labels,
                    title="Confusion Matrix — Quantized Model",
                    ylabel="True label",
                    xlabel="Predicted label",
                )
                for i in range(len(labels)):
                    for j in range(len(labels)):
                        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                                color="white" if cm[i, j] > cm.max() / 2 else "black")
                plt.tight_layout()
                cm_path = "quantization_confusion_matrix.png"
                plt.savefig(cm_path)
                plt.close()
                mlflow.log_artifact(cm_path)
            except Exception as e:
                print(f"  Warning: could not save confusion matrix plot: {e}")

            print(f"  MLflow run logged to experiment '{experiment_name}' at {mlflow_uri}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize a causal LM")
    parser.add_argument(
        "--model", default="Qwen/Qwen2.5-1.5B", help="Model to quantize"
    )
    parser.add_argument("--output", default="./qwen2.5-1.5b-gptq-4bit", help="Output directory")
    parser.add_argument("--method", choices=["gptq"], default="gptq", help="Quantization method")
    parser.add_argument("--bits", type=int, default=4, help="Quantization bits")
    parser.add_argument("--group-size", type=int, default=128, help="Group size")
    parser.add_argument("--num-samples", type=int, default=48, help="Number of calibration samples")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate quantized model after quantization")
    parser.add_argument("--eval-dataset", default=None, help="Path to evaluation dataset CSV")
    parser.add_argument("--mlflow-uri", default=None, help="MLflow tracking URI for logging eval metrics")
    parser.add_argument("--experiment-name", default=None, help="MLflow experiment name to log runs into")
    args = parser.parse_args()

    if args.method == "gptq":
        quantize_gptq(
            args.model, args.output,
            bits=args.bits, group_size=args.group_size,
            num_samples=args.num_samples,
        )
    else:
        raise ValueError(f"Unexpected quantization method: {args.method}")

    if args.evaluate:
        if args.eval_dataset is None:
            print("Error: --eval-dataset is required when using --evaluate")
        else:
            evaluate_quantized_model(
                args.output,
                args.eval_dataset,
                mlflow_uri=args.mlflow_uri,
                experiment_name=args.experiment_name,
            )
