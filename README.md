# ML Sentiment Analysis — End-to-End MLOps Pipeline

A production-ready sentiment analysis system built around **Qwen2.5-1.5B**, featuring few-shot prompting, 4-bit GPTQ quantization, MLflow experiment tracking, A/B testing with traffic routing, automated drift detection, and Kubernetes canary deployments.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Training & Experimentation](#training--experimentation)
- [Quantization](#quantization)
- [Serving & A/B Testing](#serving--ab-testing)
- [Monitoring & Drift Detection](#monitoring--drift-detection)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Results](#results)
- [License](#license)

---

## Overview

This project demonstrates a full MLOps lifecycle for a Large Language Model (LLM) application:

1. **Experimentation** — Few-shot prompt engineering with MLflow tracking
2. **Optimization** — GPTQ 4-bit quantization to reduce model size and inference cost
3. **Serving** — FastAPI traffic router for A/B testing between model versions
4. **Monitoring** — Automated data drift detection with Evidently AI and Prometheus metrics
5. **Deployment** — Canary rollouts on Kubernetes with Argo Rollouts and automated analysis gates

---

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Training &    │────▶│   MLflow Registry  │────▶│   Quantization  │
│   Few-Shot Exp  │     │   & Tracking       │     │   (GPTQ 4-bit)  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                          │
                          ┌───────────────────────────────┘
                          ▼
               ┌────────────────────┐
               │   FastAPI Router   │────▶ Logs predictions to JSONL
               │  (A/B: 80/20 split)│
               └────────────────────┘
                          │
           ┌──────────────┴──────────────┐
           ▼                             ▼
    ┌─────────────┐               ┌─────────────┐
    │   V1 (FP16) │               │   V2 (GPTQ) │
    │  vLLM Server│               │  vLLM Server│
    └─────────────┘               └─────────────┘
           │                             │
           └──────────────┬──────────────┘
                          ▼
               ┌────────────────────┐
               │  Evidently Drift   │
               │   Detection +      │
               │  Prometheus Metrics │
               └────────────────────┘
                          │
                          ▼
               ┌────────────────────┐
               │  Argo Rollouts     │
               │  (Canary K8s)      │
               └────────────────────┘
```

---

## Features

- **Few-Shot Prompt Engineering** — Systematically evaluated 1-shot, 3-shot, and 5-shot prompting strategies with MLflow tracking
- **Model Quantization** — Compressed Qwen2.5-1.5B from ~3GB to ~1GB using GPTQ 4-bit quantization with minimal accuracy loss
- **A/B Traffic Routing** — FastAPI router splits traffic (80/20 by default) between original (FP16) and quantized model versions
- **MLflow Integration** — Full experiment tracking, metric logging, artifact storage, and model registry
- **Drift Detection** — Automated data and target drift monitoring using Evidently AI with HTML reports
- **Prometheus Metrics** — Pushgateway integration for drift scores and latency metrics
- **Canary Deployments** — Kubernetes-native progressive rollouts with Argo Rollouts and automated drift analysis gates
- **Comprehensive Logging** — All predictions logged to JSONL for auditability and monitoring

---

## Tech Stack

| Category | Technologies |
|----------|-------------|
| **Model** | Qwen2.5-1.5B, Transformers, PyTorch |
| **Quantization** | GPTQ, AutoGPTQ, gptqmodel |
| **Experiment Tracking** | MLflow |
| **Serving** | FastAPI, vLLM, HTTPX |
| **Monitoring** | Evidently AI, Prometheus, Grafana |
| **Orchestration** | Kubernetes, Argo Rollouts |
| **Infrastructure** | Docker, ConfigMaps, ServiceMonitors |

---

## Project Structure

```
ml-sentiment-app/
├── data/                       # Sample datasets
│   ├── train_sentiment.csv
│   └── eval_sentiment.csv
├── training/
│   ├── configs/
│   │   └── experiment_config.yaml    # MLflow & model config
│   ├── src/
│   │   ├── train.py                  # Main training & experiment runner
│   │   ├── data_loader.py            # Dataset generation & loading
│   │   ├── evaluate.py               # Evaluation utilities
│   │   ├── registration.py           # MLflow model registry
│   │   └── experiments/
│   │       └── few_shot.py           # Few-shot prompting experiments
│   └── requirements.txt
├── quantization/
│   ├── quantize_model.py             # GPTQ quantization + eval
│   └── __pycache__/
├── serving/
│   ├── traffic_router.py             # FastAPI A/B router
│   ├── Dockerfile
│   └── requirements.txt
├── monitoring/
│   ├── drift_detector.py             # Evidently drift detection
│   ├── send_predictions.py           # Push metrics to Prometheus
│   ├── Dockerfile.analysis
│   ├── analysis_entrypoint.sh
│   └── evidently_config.yaml
├── k8s/                              # Kubernetes manifests
│   ├── rollout.yaml                  # Argo Rollouts canary config
│   ├── analysis-template.yaml        # Automated drift analysis
│   ├── service.yaml
│   ├── configmap.yaml
│   ├── grafana-dashboard.yaml
│   └── pushgateway-servicemonitor.yaml
├── report/                           # Assignment report materials
└── README.md
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended for inference)
- Docker (for serving)
- Kubernetes cluster (for deployment)

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r training/requirements.txt
pip install -r serving/requirements.txt
```

### 2. Start MLflow

```bash
mlflow server --host 0.0.0.0 --port 5050
```

### 3. Generate Sample Data

```bash
python -c "from training.src.data_loader import create_sample_dataset; create_sample_dataset('data/eval_sentiment.csv')"
```

---

## Training & Experimentation

### Run Few-Shot Experiments

```bash
python -m training.src.train
```

This will:
- Set up MLflow experiment tracking
- Run 1-shot, 3-shot, and 5-shot prompting experiments
- Log metrics, parameters, and artifacts (confusion matrices, predictions) to MLflow

### View Results

Navigate to `http://localhost:5050` to see tracked experiments in the MLflow UI.

---

## Quantization

### Quantize the Model

```bash
python quantization/quantize_model.py \
  --model Qwen/Qwen2.5-1.5B \
  --output ./qwen2.5-1.5b-gptq-4bit \
  --method gptq \
  --bits 4 \
  --num-samples 48 \
  --evaluate \
  --eval-dataset data/eval_sentiment.csv \
  --mlflow-uri http://localhost:5050 \
  --experiment-name sentiment-qwen2.5-experiments
```

### Quantization Results

| Metric | Value |
|--------|-------|
| Original Size | ~3.0 GB |
| Quantized Size | ~1.0 GB |
| Compression Ratio | ~3.0x |
| Size Reduction | ~67% |

---

## Serving & A/B Testing

### Start the Traffic Router

```bash
# Option 1: Direct Python
python -m serving.traffic_router

# Option 2: Docker
docker build -t sentiment-router:latest serving/
docker run -p 8000:8000 sentiment-router:latest
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/predict` | POST | Sentiment prediction (JSON: `{"text": "..."}`) |
| `/logs` | GET | Retrieve prediction logs (JSONL) |
| `/stats` | GET | A/B testing statistics |

### Example Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is amazing!"}'
```

### Response

```json
{
  "label": "positive",
  "model_version": "v1",
  "latency_ms": 142.5
}
```

---

## Monitoring & Drift Detection

### Generate Drift Report

```bash
python monitoring/drift_detector.py \
  --reference v1_predictions.jsonl \
  --current v2_predictions.jsonl \
  --output drift_report.html \
  --push \
  --evidently-url http://localhost:8085
```

### Push Metrics to Prometheus

```bash
python monitoring/drift_detector.py \
  --reference v1_predictions.jsonl \
  --current v2_predictions.jsonl \
  --pushgateway-url http://localhost:9091
```

### View Reports

- **Evidently UI**: `http://localhost:8085`
- **Grafana**: `http://localhost:3000`
- **Local HTML**: Open `drift_report.html` in your browser

---

## Kubernetes Deployment

### Deploy to Kubernetes

```bash
# Apply all manifests
kubectl apply -f k8s/

# Check rollout status
kubectl argo rollouts get rollout sentiment-model

# Watch the canary progression
kubectl argo rollouts watch rollout sentiment-model
```

### Canary Rollout Steps

1. **20% Traffic** → Drift analysis gate
2. **50% Traffic** → 30s pause
3. **80% Traffic** → 30s pause
4. **100% Traffic** → Rollout complete

The rollout automatically aborts if the drift score exceeds the threshold (0.5).

---

## Results

### Few-Shot Performance

| Experiment | Accuracy | F1 Score | Latency (ms) |
|------------|----------|----------|--------------|
| 1-shot | ~0.65 | ~0.62 | ~120 |
| 3-shot | ~0.72 | ~0.70 | ~145 |
| 5-shot | ~0.75 | ~0.73 | ~160 |

### Quantization vs. Baseline

| Model | Accuracy | F1 Score | Size | Latency |
|-------|----------|----------|------|---------|
| V1 (FP16) | ~0.78 | ~0.76 | ~3.0 GB | ~140 ms |
| V2 (GPTQ 4-bit) | ~0.76 | ~0.74 | ~1.0 GB | ~90 ms |

---

## License

This project is for educational purposes (CISC-814 Assignment 2).

---

## Author

Built as part of a graduate-level MLOps course, demonstrating end-to-end LLM operationalization from experimentation to production deployment.

---

## Acknowledgments

- [Qwen](https://github.com/QwenLM/Qwen) for the base model
- [MLflow](https://mlflow.org/) for experiment tracking
- [Evidently AI](https://www.evidentlyai.com/) for drift detection
- [Argo Rollouts](https://argoproj.github.io/argo-rollouts/) for progressive delivery
- [vLLM](https://github.com/vllm-project/vllm) for high-throughput serving
