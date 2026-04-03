---
title: OpenEnv Email Triage
emoji: 📧
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
tags: [openenv]
---
# OpenEnv: Enterprise Email Triage & Workflow Simulation (EETWS)

A deterministic reinforcement learning benchmark for evaluating AI agents on enterprise email triage workflows — classification, routing, escalation, and safe reply drafting.

---

## Problem Statement

Enterprise support teams process thousands of daily emails spanning password resets, sales inquiries, and critical VIP escalations. Automating initial triage (classify → route → respond) reduces resolution time dramatically, but AI agents must handle this **safely, accurately, and without hallucination**.

EETWS provides a rigorous, deterministic evaluation environment with dense reward shaping, anti-gaming protections, and realistic multi-step workflows across three difficulty levels.

---

## Architecture

```
┌─────────────────────────────────────────────┐
│  FastAPI Wrapper (env/environment.py)       │
│  POST /reset  · POST /step  · GET /state    │
├─────────────────────────────────────────────┤
│  EmailEnv Core (Pure Python)                │
│  Deterministic state · Action validation    │
│  Loop detection · Step limits               │
├──────────────┬──────────────────────────────┤
│  Tasks       │  Graders                     │
│  (tasks.py)  │  (graders.py)                │
│  3 scenarios │  Dense reward [0.0 – 1.0]    │
└──────────────┴──────────────────────────────┘
```

**Key design decisions:**
- Environment state is fully deterministic — no randomness after initialization
- Single-worker deployment prevents state corruption
- All rewards are clamped to `[0.0, 1.0]` after every computation
- Loop detection terminates episodes that exhibit repeated or alternating action patterns

---

## Schema Definitions

### Observation Space

Returned by `/reset` and within every `/step` response:

```json
{
  "email_id": "email_001",
  "subject": "Cannot reset my password",
  "sender_type": "customer",
  "email_body": "Hi, I've been trying to reset my password...",
  "urgency_score": 0.4,
  "sentiment": "frustrated",
  "thread_history": [],
  "current_status": "unread",
  "previous_actions": []
}
```

### Action Space

Sent to `/step`:

```json
{
  "action_type": "classify_email",
  "arguments": {
    "category": "support"
  }
}
```

**Valid `action_type` values:** `classify_email`, `route_to`, `draft_reply`, `escalate`, `mark_spam`, `request_more_info`, `noop`

### Step Response

Returned by `/step`:

```json
{
  "observation": { "..." : "..." },
  "reward": {
    "score": 0.85,
    "breakdown": {
      "classification": 0.8,
      "routing": 0.0,
      "reply_quality": 0.0,
      "efficiency": 0.15,
      "bonus": 0.1,
      "penalties": -0.03
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

---

## Tasks

### Task 1 — Basic Classification (Easy)

**Scenario:** A customer emails about a password reset issue.  
**Goal:** Classify the email as `support` in a single action.  
**Grading:** Exact match on classification. Penalties for unnecessary extra actions.

### Task 2 — Classification + Routing + Reply (Medium)

**Scenario:** A customer inquires about upgrading to the Enterprise plan.  
**Goal:** Classify as `sales` → Route to `sales_department` → Draft a reply referencing the upgrade/demo.  
**Grading:** Checks each step independently. Reply evaluated for keyword relevance and politeness.

### Task 3 — Full Workflow with VIP Escalation (Hard)

**Scenario:** A VIP client reports a production server outage requiring immediate attention.  
**Goal:** Classify as `support` → Escalate → Route to `engineering_escalation` → Draft an urgent acknowledgment reply.  
**Grading:** Mandatory escalation check. Heavy penalty for failing to escalate VIP issues. Reply must reference urgency.

---

## Reward Design

All rewards are bounded to `[0.0, 1.0]`. The breakdown tracks six components:

| Component | Description |
|-----------|-------------|
| `classification` | Correct category match |
| `routing` | Correct department routing |
| `reply_quality` | Keyword relevance in drafted replies |
| `efficiency` | Bonus for completing tasks in fewer steps |
| `bonus` | Politeness bonus, perfect-sequence bonus, escalation credit |
| `penalties` | Step cost (0.03/step), duplicate actions, incorrect values |

**Anti-gaming protections:**
- Duplicate classifications or routings incur `-0.2` penalty
- Loop detection (repeated or alternating actions) terminates the episode with score penalty
- Verbose, low-relevance replies are penalized
- Missing mandatory escalation (task 3) costs `-0.2`

---

## Local Setup

### Prerequisites

- Python 3.10+
- An OpenAI-compatible API key (set as `HF_TOKEN` or `API_KEY`)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Start the Environment Server

```bash
uvicorn env.environment:app --host 0.0.0.0 --port 7860 --workers 1
```

The server will be available at `http://localhost:7860`.

### Run the Inference Agent

In a **separate terminal**, with the server running:

```bash
export HF_TOKEN="your-api-key"
export MODEL_NAME="gpt-4o-mini"
python inference.py
```

On Windows (PowerShell):

```powershell
$env:HF_TOKEN="your-api-key"
$env:MODEL_NAME="gpt-4o-mini"
python inference.py
```

---

## Docker Usage

### Build

```bash
docker build -t openenv-email .
```

### Run

```bash
docker run -p 7860:7860 openenv-email
```

The environment API will be accessible at `http://localhost:7860`.

### Run Inference Against Docker Container

With the container running:

```bash
export ENV_BASE_URL="http://localhost:7860"
export HF_TOKEN="your-api-key"
python inference.py
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/reset` | Reset environment. Body: `{"task_id": "task_1"}` |
| `POST` | `/step` | Take an action. Body: `{"action_type": "...", "arguments": {...}}` |
| `GET` | `/state` | Get current environment state (debug) |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `Connection refused` on inference | Ensure the environment server is running on port 7860 before running `inference.py` |
| `Invalid API key` error | Set `HF_TOKEN` or `API_KEY` environment variable with a valid key |
| Score is 0.0 on all tasks | Check that `MODEL_NAME` points to a model that supports JSON mode |
| Docker port conflict | Ensure nothing else is using port 7860, or map to a different port: `docker run -p 8080:7860 openenv-email` |
| `ModuleNotFoundError: env` | Run uvicorn from the project root directory, not from inside `env/` |

---

## Project Structure

```
├── env/
│   ├── __init__.py          # Package init
│   ├── models.py            # Pydantic models (Observation, Action, Reward, etc.)
│   ├── tasks.py             # Task definitions (3 scenarios)
│   ├── graders.py           # Deterministic grading logic
│   └── environment.py       # EmailEnv class + FastAPI endpoints
├── inference.py             # OpenAI-compatible inference agent
├── openenv.yaml             # OpenEnv schema specification
├── requirements.txt         # Python dependencies
├── Dockerfile               # Container build configuration
└── README.md
```

---

## Pre-Validation Checklist

Before submitting to the hackathon, ensure the following are validated:
- [x] HF Space returns `200` on the root (`/`) path.
- [x] `reset()` endpoint works remotely via `POST /reset`.
- [x] `step()`, `reset()`, and `state()` are callable.
- [x] `openenv.yaml` is valid and present in the ROOT directory.
- [x] Typed Pydantic models are implemented for Observations, Actions, and Rewards.
- [x] `Dockerfile` builds successfully and exposes port 7860.
- [x] `inference.py` is at the root and runs automatically without manual input.
- [x] `inference.py` uses `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN` from the environment.
- [x] At least 3 tasks are defined (Easy, Medium, Hard).
- [x] Graders return a bounded float value in `[0.0, 1.0]`.
- [x] Reward function provides partial progress signals (dense reward).
