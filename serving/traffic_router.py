"""FastAPI traffic router for A/B testing between model versions.

Routes 80% of traffic to V1 (original) and 20% to V2 (quantized).
Logs all predictions to a JSON lines file.
"""

from __future__ import annotations

import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

app = FastAPI(title="Sentiment A/B Traffic Router", version="2.0.0")

# Model server endpoints (K8s manifests set V1_URL / V2_URL env vars)
V1_URL = os.environ.get("V1_URL", "http://localhost:8001/v1/completions")
V2_URL = os.environ.get("V2_URL", "http://localhost:8002/v1/completions")

# Traffic split: 80% V1, 20% V2
V1_WEIGHT = float(os.environ.get("V1_WEIGHT", "0.80"))

# Prediction log file
LOG_FILE = Path(os.environ.get("LOG_FILE", "predictions.jsonl"))

PROMPT_TEMPLATE = (
    "Classify the sentiment of the following text as positive, negative, or neutral.\n"
    "Text: {text}\n"
    "Sentiment:"
)


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    label: str
    model_version: str
    latency_ms: float


def select_model_version() -> tuple[str, str]:
    """Select model version and completions URL based on traffic weights."""
    if random.random() < V1_WEIGHT:
        return "v1", V1_URL
    return "v2", V2_URL


def parse_sentiment(text: str) -> str:
    """Extract sentiment label from completion."""
    text_lower = text.strip().lower()
    for label in ("positive", "negative", "neutral"):
        if label in text_lower:
            return label
    return "neutral"


def log_prediction(
    version: str,
    input_text: str,
    output_label: str,
    latency_ms: float,
) -> None:
    """Append prediction to JSON lines log file."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_version": version,
        "input_text": input_text,
        "predicted_label": output_label,
        "latency_ms": latency_ms,
    }
    with LOG_FILE.open("a") as f:
        f.write(json.dumps(entry) + "\n")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Empty input text")

    # 1. Format the prompt using PROMPT_TEMPLATE
    prompt = PROMPT_TEMPLATE.format(text=request.text)
    
    # 2. Select the right model version
    version, url = select_model_version()
    
    # 3. Send the request to the selected model version, making sure to time the call's latency
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            start_time = time.time()
            response = await client.post(
                url,
                json={
                    "prompt": prompt,
                    "max_tokens": 10,
                    "temperature": 0.0,
                },
            )
            response.raise_for_status()
            latency_ms = (time.time() - start_time) * 1000
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"Model server error: {e}")
    
    # 4. Parse the output to extract sentiment label
    result = response.json()
    generated_text = result["choices"][0]["text"]
    label = parse_sentiment(generated_text)
    
    # 5. Log the request, the obtained label and latency
    log_prediction(version, request.text, label, latency_ms)
    
    # 6. Return a PredictResponse object as result
    return PredictResponse(label=label, model_version=version, latency_ms=latency_ms)


@app.get("/logs", response_class=PlainTextResponse)
def get_logs(n: int | None = None) -> str:
    """Return prediction logs as raw JSONL for drift monitoring.

    Args:
        n: If provided, return only the last n lines.
    """
    if not LOG_FILE.exists():
        return ""
    lines = LOG_FILE.read_text().splitlines()
    if n is not None:
        lines = lines[-n:]
    return "\n".join(lines) + "\n" if lines else ""


@app.get("/stats")
def stats() -> dict:
    """Return basic A/B testing statistics."""
    if not LOG_FILE.exists():
        return {"total_predictions": 0}

    total_predictions = 0
    v1_count = 0
    v2_count = 0
    v1_latencies = []
    v2_latencies = []

    with LOG_FILE.open() as f:
        for line in f:
            entry = json.loads(line.strip())
            total_predictions += 1
            if entry["model_version"] == "v1":
                v1_count += 1
                v1_latencies.append(entry["latency_ms"])
            else:
                v2_count += 1
                v2_latencies.append(entry["latency_ms"])

    v1_pct = (v1_count / total_predictions * 100) if total_predictions > 0 else 0
    v2_pct = (v2_count / total_predictions * 100) if total_predictions > 0 else 0
    v1_avg_latency_ms = sum(v1_latencies) / len(v1_latencies) if v1_latencies else 0
    v2_avg_latency_ms = sum(v2_latencies) / len(v2_latencies) if v2_latencies else 0

    return {
        "total_predictions": total_predictions,
        "v1_count": v1_count,
        "v2_count": v2_count,
        "v1_pct": round(v1_pct, 2),
        "v2_pct": round(v2_pct, 2),
        "v1_avg_latency_ms": round(v1_avg_latency_ms, 2),
        "v2_avg_latency_ms": round(v2_avg_latency_ms, 2),
    }
