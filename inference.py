import os
import json
import requests
from typing import Dict, Any
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")

# URL for testing the env against the FastAPI wrapper
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

TASKS_TO_RUN = ["task_1", "task_2", "task_3"]

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
"""

def reset_env(task_id: str):
    url = f"{ENV_BASE_URL}/reset"
    req_data = {"task_id": task_id}
    response = requests.post(url, json=req_data)
    response.raise_for_status()
    return response.json()

def step_env(action: dict):
    url = f"{ENV_BASE_URL}/step"
    response = requests.post(url, json=action)
    response.raise_for_status()
    return response.json()

def get_action_from_llm(obs: dict) -> dict:
    prompt = f"Current Observation: {json.dumps(obs, indent=2)}\n\nWhat action will you take next? Reply ONLY with JSON."
    
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0, # Deterministic
            response_format={"type": "json_object"}
        )
        
        reply = completion.choices[0].message.content
        action_obj = json.loads(reply)
        
        # Ensure it has the correct structure loosely
        if "action_type" not in action_obj:
            return {"action_type": "noop", "arguments": {}}
        
        if "arguments" not in action_obj:
            action_obj["arguments"] = {}
            
        return action_obj
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return {"action_type": "noop", "arguments": {"error": str(e)}}

def run_task(task_id: str):
    print(f"\n{'='*40}")
    print(f"Starting {task_id}")
    print(f"{'='*40}")
    
    obs = reset_env(task_id)
    done = False
    
    while not done:
        action = get_action_from_llm(obs)
        print(f"Action Taken: {action}")
        
        response = step_env(action)
        obs = response["observation"]
        reward = response["reward"]
        done = response["done"]
        info = response["info"]
        
        print(f"Reward: {reward}")
        print(f"Step Info: {info}")
        print("-" * 20)
        
    print(f"Task {task_id} complete. Final score: {reward.get('score', 0)}")
    return reward.get("score", 0)

if __name__ == "__main__":
    total_score = 0.0
    scores = {}
    
    for task in TASKS_TO_RUN:
        score = run_task(task)
        scores[task] = score
        total_score += score
        
    print("\n\n" + "="*40)
    print("FINAL RESULTS")
    print("="*40)
    for t, s in scores.items():
        print(f"{t}: {s}")
        
    avg_score = total_score / len(TASKS_TO_RUN)
    print(f"Average Score: {avg_score:.2f}")
    if avg_score >= 0.8:
        print("Success! The agent performed very well.")
    else:
        print("Agent performance could be improved.")
