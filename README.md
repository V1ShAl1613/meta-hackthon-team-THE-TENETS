---
title: OpenEnv Email Triage
emoji: "📧"
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
tags:
  - openenv
---

# OpenEnv: Enterprise Email Triage & Workflow Simulation

A deterministic environment for evaluating agents on enterprise email triage:
classification, routing, escalation, and reply drafting.

## Overview

This repo exposes a FastAPI environment with three tasks:

1. `task_1`: classify a password reset request as `support`
2. `task_2`: classify a sales inquiry, route it to `sales_department`, and draft a relevant reply
3. `task_3`: classify an urgent VIP outage, route it to `engineering_escalation`, explicitly escalate it, and draft an urgent reply

The environment is deterministic, uses dense rewards, and applies loop detection to stop repeated action patterns.

## Score Rules

- Final reward scores are always clamped into the safe band `[0.01, 0.99]`
- Reward scores therefore remain strictly inside `(0, 1)`
- Reward breakdown values are also clamped into `[0.01, 0.99]`
- Invalid inputs fall back to `0.5` before clamping logic is applied downstream

## API

Available endpoints:

- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /`

Example `POST /step` body:

```json
{
  "action_type": "classify_email",
  "arguments": {
    "category": "support"
  }
}
```

Example response shape:

```json
{
  "observation": {
    "email_id": "email_001",
    "subject": "Cannot reset my password",
    "sender_type": "customer",
    "email_body": "Hi, I've been trying to reset my password...",
    "urgency_score": 0.4,
    "sentiment": "frustrated",
    "thread_history": [],
    "current_status": "classified: support",
    "previous_actions": ["classify_email"]
  },
  "reward": {
    "score": 0.92,
    "breakdown": {
      "classification": 0.7,
      "routing": 0.1,
      "reply_quality": 0.1,
      "efficiency": 0.17,
      "bonus": 0.2,
      "penalties": 0.1
    }
  },
  "done": true,
  "info": {
    "step": 1,
    "task_id": "task_1",
    "reason": "Task complete"
  }
}
```

## Local Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the server:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860 --workers 1
```

Run the inference agent in another terminal:

```bash
export HF_TOKEN="your-api-key"
export MODEL_NAME="gpt-4o-mini"
python inference.py
```

PowerShell:

```powershell
$env:HF_TOKEN="your-api-key"
$env:MODEL_NAME="gpt-4o-mini"
python inference.py
```

## Docker

Build:

```bash
docker build -t openenv-email .
```

Run:

```bash
docker run -p 7860:7860 openenv-email
```

## Validation

Useful checks:

- `python -m pytest tests/test_score_bounds.py -q`
- `openenv validate`
- `bash validate-submission.sh https://<your-space>.hf.space .`

## Project Layout

```text
env/
  __init__.py
  environment.py
  graders.py
  models.py
  tasks.py
server/
  app.py
tests/
  test_score_bounds.py
Dockerfile
inference.py
openenv.yaml
requirements.txt
```
