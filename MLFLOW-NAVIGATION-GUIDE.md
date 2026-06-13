# MLflow UI Navigation Guide - Finding V1 and V2 Models

## Important Note

In MLflow, the models are **NOT** labeled as "V1" and "V2". They are registered as:
- **V1** = Model with alias **"champion"**
- **V2** = Model with alias **"challenger"**

Both versions are under the registered model name: **team-queens-a2-sentiment**

---

## Step-by-Step Navigation

### Step 1: Open MLflow UI
Go to: `http://localhost:5050`

### Step 2: Click "Models" in Left Sidebar

```
┌─────────────────────────────────────┐
│  MLflow                             │
│                                     │
│  📊 Experiments                     │
│  🧩 Models        ← CLICK HERE      │
│  ☐ Experiments (beta)               │
│  ⚙️ Admin                           │
└─────────────────────────────────────┘
```

### Step 3: Find the Registered Model

You will see a table like this:

```
┌─────────────────────────────────────────────────────┐
│ Registered Models                                   │
├─────────────────────────────────┬─────────┬─────────┤
│ Name                            │ Version │ Created │
├─────────────────────────────────┼─────────┼─────────┤
│ team-queens-a2-sentiment        │ 2       │ Jun 3   │
└─────────────────────────────────┴─────────┴─────────┘
```

Click on: **team-queens-a2-sentiment**

### Step 4: View Model Versions

You will see two versions:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ team-queens-a2-sentiment                                                    │
├──────────┬──────────────┬───────────────────────────────────────────────────┤
│ Version  │ Alias        │ Tags                                              │
├──────────┼──────────────┼───────────────────────────────────────────────────┤
│ Version 1│ 🏆 champion  │ vllm_model_path: Qwen/Qwen2.5-1.5B              │
│          │              │                                                   │
│ Version 2│ ⚔️ challenger│ vllm_model_path:                                 │
│          │              │ /home/localuser/ml-sentiment-app-a2/            │
│          │              │ qwen2.5-1.5b-gptq-4bit                          │
└──────────┴──────────────┴───────────────────────────────────────────────────┘
```

### Step 5: Map to V1/V2

| Version | Alias | Maps To | Model Path |
|---------|-------|---------|------------|
| Version 1 | champion | **V1** | `Qwen/Qwen2.5-1.5B` |
| Version 2 | challenger | **V2** | `/home/localuser/ml-sentiment-app-a2/qwen2.5-1.5b-gptq-4bit` |

### Step 6: Click on a Version for Details

Click on **Version 1** or **Version 2** to see:
- Run ID
- Source experiment
- All tags
- Creation time
- Model signature (if any)

---

## Quick Reference

### Champion (V1)
- **Alias:** champion
- **Version:** 1
- **Run ID:** 0ca90753e34b4853aa1660dff60772fd
- **vllm_model_path:** `Qwen/Qwen2.5-1.5B`
- **Type:** Original fp16 model
- **Server:** http://localhost:8001

### Challenger (V2)
- **Alias:** challenger
- **Version:** 2
- **Run ID:** 187b71310ead4e1f8de65658d3383ff4
- **vllm_model_path:** `/home/localuser/ml-sentiment-app-a2/qwen2.5-1.5b-gptq-4bit`
- **Type:** GPTQ 4-bit quantized model
- **Server:** http://localhost:8002

---

## Common Issue

**"I only see Experiments, not Models"**

Make sure you clicked **Models** in the left sidebar, not **Experiments**. The models are registered separately from experiment runs.

---

## Alternative: Query via Python

If you still can't find them in the UI, run this Python script:

```python
import mlflow
mlflow.set_tracking_uri('http://localhost:5050')
client = mlflow.tracking.MlflowClient()

for rm in client.search_registered_models():
    print(f"\nModel: {rm.name}")
    for alias, version in rm.aliases.items():
        mv = client.get_model_version(rm.name, version)
        print(f"  [{alias}] Version {version}")
        print(f"    Path: {mv.tags.get('vllm_model_path')}")
```
