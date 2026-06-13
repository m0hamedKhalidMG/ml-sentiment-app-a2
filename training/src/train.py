"""Training script with MLflow integration.

This script:
1. Sets up MLflow experiment tracking
2. Runs few-shot prompting experiments (with MLflow logging)
3. Runs quantization evaluation (with MLflow logging)
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime

import mlflow
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.experiments.few_shot import run_few_shot


def load_config(config_path: str = "configs/experiment_config.yaml") -> dict:
    """Load experiment configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def setup_mlflow(experiment_name: str, tracking_uri: str) -> None:
    """Set up MLflow tracking.

    Creates or gets an experiment with artifact proxying enabled.
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()
    
    # Try to create experiment with artifact proxying
    try:
        experiment_id = client.create_experiment(
            experiment_name,
            artifact_location="mlflow-artifacts:/"
        )
        print(f"Created new experiment: {experiment_name} (ID: {experiment_id})")
    except mlflow.exceptions.MlflowException:
        # Experiment already exists
        experiment = client.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
        print(f"Using existing experiment: {experiment_name} (ID: {experiment_id})")
    
    # Set as active experiment
    mlflow.set_experiment(experiment_name)


def load_model_and_tokenizer(
    model_name: str, device: str = "auto"
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load the model and tokenizer."""
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=device,
    )
    return model, tokenizer


def run_quantization_eval(
    tracking_uri: str,
    experiment_name: str,
    model_name: str,
    eval_path: str,
    quantization_cfg: dict,
) -> None:
    """Run quantization with evaluation and MLflow logging.

    Calls quantize_model.py with --evaluate and --mlflow-uri flags so that
    post-quantization metrics (accuracy, F1, latency, model size) are logged
    to the same MLflow tracking server.
    """
    output_dir = quantization_cfg.get("output_dir", "./qwen2.5-1.5b-gptq-4bit")
    num_samples = quantization_cfg.get("num_samples", 48)
    bits = quantization_cfg.get("bits", 4)

    quantize_script = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "..", "quantization", "quantize_model.py",
    )

    cmd = [
        sys.executable, quantize_script,
        "--model", model_name,
        "--output", output_dir,
        "--method", "gptq",
        "--bits", str(bits),
        "--num-samples", str(num_samples),
        "--evaluate",
        "--eval-dataset", eval_path,
        "--mlflow-uri", tracking_uri,
        "--experiment-name", experiment_name
    ]
    print(f"  Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    

if __name__ == "__main__":
    config = load_config()

    tracking_uri = config["mlflow"]["tracking_uri"]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_name = config["mlflow"]["experiment_name"]

    print("Setting up MLflow...")
    setup_mlflow(experiment_name, tracking_uri)

    model_name = config["model"]["name"]
    eval_path = config["data"]["eval_dataset"]

    model, tokenizer = load_model_and_tokenizer(
        model_name, device=config["model"].get("device", "auto")
    )

    quant_cfg = config.get("quantization", {})
    
    # Run few-shot experiments with MLflow logging
    with mlflow.start_run(run_name="few_shot_experiments") as parent_run:
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("experiment_type", "few_shot")
        run_few_shot(model, tokenizer, eval_path, model_name)
    
    # Run quantization evaluation
    run_quantization_eval(
        tracking_uri,
        experiment_name,
        model_name,
        eval_path,
        quant_cfg,
    )

    print("Done!")
