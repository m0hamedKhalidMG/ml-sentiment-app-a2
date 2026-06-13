"""Register the best model of a given experiment type in the MLflow Model Registry."""

from __future__ import annotations

import argparse
import os

import mlflow
from mlflow.tracking import MlflowClient
import yaml


def load_config(config_path: str = "configs/experiment_config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def register_model(
    tracking_uri: str,
    experiment_name: str,
    experiment_type: str,
    alias: str,
) -> None:
    """Register the best run of a given experiment type under the given alias."""
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    # Get experiment by name
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"Experiment '{experiment_name}' not found.")

    # Search runs for the given experiment type, ordered by F1 score (descending)
    # Try both f1_score and f1_macro metric names
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"params.experiment_type = '{experiment_type}'",
        order_by=["metrics.f1_score DESC", "metrics.f1_macro DESC"],
    )

    if not runs:
        raise ValueError(f"No runs found for experiment_type='{experiment_type}' in experiment '{experiment_name}'")

    best_run = runs[0]
    run_id = best_run.info.run_id
    f1_val = best_run.data.metrics.get("f1_score") or best_run.data.metrics.get("f1_macro") or "N/A"
    print(f"Best run found for {experiment_type}: {run_id} (F1: {f1_val})")

    # Create a dummy artifact to enable model registration (since we don't store actual model weights)
    dummy_path = "dummy_registry_marker.txt"
    with open(dummy_path, "w") as f:
        f.write("Registry reference metadata placeholder.")

    # Log the dummy artifact to the run
    with mlflow.start_run(run_id=run_id):
        mlflow.log_artifact(dummy_path, artifact_path="model")

    # Clean up dummy file
    if os.path.exists(dummy_path):
        os.remove(dummy_path)

    # Register the model (create if it doesn't exist)
    model_name = experiment_name
    try:
        client.create_registered_model(model_name)
        print(f"Created new registered model: {model_name}")
    except mlflow.exceptions.MlflowException:
        # Model already exists
        pass

    # Create a new model version pointing to the run's artifact
    model_version = client.create_model_version(
        name=model_name,
        source=f"runs:/{run_id}/model",
        run_id=run_id,
    )

    # Set the alias on the model version
    client.set_registered_model_alias(
        name=model_name,
        alias=alias,
        version=str(model_version.version),
    )

    # Set the vllm_model_path tag on the model version
    vllm_path = best_run.data.params.get("vllm_model_path", "unknown")
    client.set_model_version_tag(
        name=model_name,
        version=str(model_version.version),
        key="vllm_model_path",
        value=vllm_path,
    )

    print(f"Successfully registered '{model_name}' version {model_version.version} with alias '{alias}'!")
    print(f"Set version tag 'vllm_model_path' to: {vllm_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register a model in MLflow Model Registry")
    parser.add_argument("--tracking-uri", required=True, help="MLflow tracking URI")
    parser.add_argument("--experiment-name", required=True, help="MLflow experiment name")
    parser.add_argument("--experiment-type", required=True, help="Value of params.experiment_type to filter on")
    parser.add_argument("--alias", required=True, help="Model alias to assign (e.g. champion, challenger)")
    args = parser.parse_args()

    register_model(args.tracking_uri, args.experiment_name, args.experiment_type, args.alias)
