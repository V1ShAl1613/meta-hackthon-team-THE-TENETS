import os
import sys
import json
import requests
from typing import Dict, Any
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL")
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")

client_kwargs = {"api_key": HF_TOKEN}
if API_BASE_URL:
    client_kwargs["base_url"] = API_BASE_URL

client = OpenAI(**client_kwargs)

TASKS_TO_RUN = ["task_1", "task_2", "task_3"]
MAX_STEPS = 8
REQUEST_TIMEOUT = 30

SYSTEM_PROMPT = """You are an Enterprise Email Agent.
Your goal is to handle incoming emails appropriately.
Allowed Actions: classify_email, route_to, draft_reply, escalate, mark_spam, request_more_info, noop.

You MUST respond strictly in valid JSON format containing two keys: "action_type" and "arguments".

Example 1:
{"action_type": "classify_email", "arguments": {"category": "support"}}

Example 2:
{"action_type": "route_to", "arguments": {"department": "sales_department"}}

Example 3:
{"action_type": "draft_reply", "arguments": {"text": "We are looking into this immediate issue."}}

Always choose the most relevant action. Avoid unnecessary actions.
"""

NOOP_ACTION = {"action_type": "noop", "arguments": {}}


def reset_env(task_id: str) -> dict:
    url = f"{ENV_BASE_URL}/reset"
    try:
        response = requests.post(url, json={"task_id": task_id}, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"[ERROR] Reset failed: {e}", file=sys.stderr)
        return {}


def step_env(action: dict) -> dict:
    url = f"{ENV_BASE_URL}/step"
    try:
        response = requests.post(url, json=action, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"[ERROR] Step failed: {e}", file=sys.stderr)
        return {
            "observation": {},
            "reward": {"score": 0.01, "breakdown": {}},
            "done": True,
            "info": {"error": str(e)}
        }


def get_action_from_llm(obs: dict) -> dict:
    prompt = f"Current Observation: {json.dumps(obs, indent=2)}\n\nWhat action will you take next? Reply ONLY with JSON."

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=150,
            response_format={"type": "json_object"}
        )

        reply = completion.choices[0].message.content
        if not reply:
            return dict(NOOP_ACTION)

        try:
            action_obj = json.loads(reply)
        except (json.JSONDecodeError, TypeError):
            return dict(NOOP_ACTION)

        if not isinstance(action_obj, dict):
            return dict(NOOP_ACTION)

        if "action_type" not in action_obj or not isinstance(action_obj["action_type"], str):
            return dict(NOOP_ACTION)

        if "arguments" not in action_obj or not isinstance(action_obj["arguments"], dict):
            action_obj["arguments"] = {}

        return action_obj

    except Exception as e:
        print(f"[ERROR] LLM call failed: {e}", file=sys.stderr)
        return dict(NOOP_ACTION)


def run_task(task_id: str) -> float:
    print("[START]")

    obs = reset_env(task_id)
    if not obs:
        print("[STEP] reward=0.01 done=true success=false")
        print("[END]")
        return 0.01

    done = False
    steps = 0
    reward_dict: dict = {"score": 0.01}

    while not done and steps < MAX_STEPS:
        steps += 1
        action = get_action_from_llm(obs)

        response = step_env(action)
        obs = response.get("observation", {})
        reward_dict = response.get("reward", {"score": 0.01})
        done = response.get("done", True)

        score_now = reward_dict.get("score", 0.01) if isinstance(reward_dict, dict) else 0.01
        score_now = max(0.01, min(0.99, float(score_now)))
        success = done and score_now > 0.5
        print(f"[STEP] reward={score_now:.4f} done={str(done).lower()} success={str(success).lower()}")

    print("[END]")
    final_score = reward_dict.get("score", 0.01) if isinstance(reward_dict, dict) else 0.01
    final_score = max(0.01, min(0.99, final_score))
    print(f"Task {task_id} complete. Final score: {final_score}")
    return final_score


if __name__ == "__main__":
    total_score = 0.0
    scores: Dict[str, float] = {}

    for task in TASKS_TO_RUN:
        score = run_task(task)
        scores[task] = score
        total_score += score

    avg_score = total_score / len(TASKS_TO_RUN)
    print(f"\nAverage Score: {avg_score:.4f}")
    if avg_score > 0.8:
        print("Result: EXCELLENT - Agent performs exceptionally well.")
    elif avg_score > 0.5:
        print("Result: ADEQUATE - Agent performs well but has room for improvement.")
    else:
        print("Result: NEEDS IMPROVEMENT - Agent performance is below threshold.")
