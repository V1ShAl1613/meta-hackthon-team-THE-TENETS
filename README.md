# OpenEnv: Enterprise Email Triage & Workflow Simulation (EETWS)

## 📌 Problem Motivation
Customer support teams at large enterprises are overwhelmed by thousands of daily emails ranging from simple password resets to critical VIP escalations. Automating the initial triage—classification, routing, and drafting safe preliminary responses—can drastically reduce resolution time and operational costs. However, training AI agents to accurately and safely perform these actions without hallucinations or improper escalations is a significant challenge.

The **EETWS OpenEnv** provides a deterministic, rigorous RL benchmark for evaluating how well an agent can handle these triage and workflow tasks under constrained steps and safety rules.

---

## 🏗️ Architecture Explanation
The environment strictly separates the **Core Environment logic** from its **API Wrapper**, ensuring it is both highly performant and easily deployable.

1. **`EmailEnv` (Core Class)**: A pure Python implementation that maintains a deterministic internal state (action history, current active email observation, step count). It executes state transitions without stochasticity once a task is initialized.
2. **FastAPI Wrapper**: Provides standard OpenEnv-compliant HTTP endpoints (`/reset`, `/step`, `/state`) specifically to wrap the pure Python logic, exposing port 7860 to be native to Hugging Face Spaces.

## 🧩 Schema Definitions

### Observation Space
Provided at every step to the agent:
```json
{
  "email_id": "string",
  "subject": "string",
  "sender_type": "string (customer | spam | internal | VIP)",
  "email_body": "string",
  "urgency_score": 0.0,
  "sentiment": "string",
  "thread_history": [],
  "current_status": "string",
  "previous_actions": []
}
```

### Action Space
Strict format using a unified model.
```json
{
  "action_type": "<classify_email|route_to|draft_reply|escalate|mark_spam|request_more_info|noop>",
  "arguments": {
     // Key-value pairs matching the specific action_type
  }
}
```

---

## 🎯 Task Descriptions

### Task 1 — Basic Classification (Easy)
- **Goal**: Classify an incoming support complaint correctly without taking unnecessary intermediary steps.
- **Grader Focus**: Exact match on classification output.

### Task 2 — Routing + Reply (Medium)
- **Goal**: Correctly identify the intent, route to the correct department, and draft an appropriate reply containing required information.
- **Grader Focus**: Checks for deterministic sequence matching on classification and routing, and evaluates drafted text utilizing a relevance ratio scoring method.

### Task 3 — Full Workflow Automation (Hard)
- **Goal**: Detect a VIP's urgent situation (e.g. server outage), classify it, route it, trigger an internal escalation step, and notify the user safely.
- **Grader Focus**: Heavily penalizes unsafe replies or failure to escalate VIP contexts, tracking exact required workflows.

---

## 🧮 Reward Design
The environment implements a **balanced, dense reward function** returning values exactly within the bounds `[0.0, 1.0]`. Average baseline agents score around `0.5 - 0.75`.

### Structured Reward Breakdown
At each step, the reward exposes a distinct breakdown:
```json
{
  "classification": 0.0,
  "routing": 0.0,
  "reply_quality": 0.0,  
  "efficiency": 0.0,
  "bonus": 0.0,
  "penalties": 0.0
}
```

### Key Grading Logic
1. **Reply Evaluation (Relevance Ratio)**: Replies are not bounded by an arbitrary hard limit. Instead, the relevance ratio is calculated: `relevant_keywords / total_words`. If the text contains extreme fluff (e.g., `< 0.05` relevance and `>30` words), keyword stuffing penalties apply.
2. **Politeness Bonus**: Replies containing polite semantics ("please", "thank you", "apologies") natively accrue +0.1 to the bonus track.
3. **Perfect Sequence Bonus**: Agents that successfully complete a full valid workflow directly implicitly unlock an absolute +0.1/+0.2 efficiency bonus on top of their core task score.
4. **Adversarial Spam Constraints**: Multi-predicting classifications or routing to guess the correct tag is strictly monitored and penalizes via the `penalties` tracking system.

> **Note on Telemetry**: All intermediate step errors are appended iteratively into `info["mistake"] / info["suggestion"]` to help agent curriculum learning strategies.

---

## 🚀 Setup Steps

### Local Python Setup
Ensure Python 3.10+ is installed.
```bash
pip install -r requirements.txt
uvicorn env.environment:app --host 0.0.0.0 --port 7860
```

### 🐳 Docker Usage
```bash
docker build -t openenv-email .
docker run -p 7860:7860 openenv-email
```

### 🌐 Hugging Face Deployment
1. Create a anew Space (Docker SDK).
2. Push this repository's contents.
3. Hugging Face Space automatically runs the `Dockerfile`, binding the app to the default `7860` port. The system will be fully responsive and show an "openenv" tag.

---

## 🧪 OpenEnv Validation & Testing
We provide an `inference.py` script locally out of the box designed to use the OpenAI API interface (adaptable to HF routers, vLLM, etc.) to traverse all tasks.

```bash
export HF_TOKEN="your-key"
export MODEL_NAME="gpt-4o"
python inference.py
```
