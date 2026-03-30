import copy
import json
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

from .models import Observation, Action, Reward, StepResponse, ResetRequest
from .tasks import TASKS
from .graders import calculate_reward, MAX_STEPS

class EmailEnv:
    def __init__(self):
        self.task_id = "task_1"
        self.current_obs = copy.deepcopy(TASKS[self.task_id]["initial_state"])
        self.action_history = []
        self.step_count = 0
        self.is_done = False
        
    def reset(self, task_id: str = "task_1") -> Observation:
        if not task_id or task_id not in TASKS:
            task_id = "task_1"
            
        self.task_id = task_id
        self.current_obs = copy.deepcopy(TASKS[self.task_id]["initial_state"])
        self.action_history = []
        self.step_count = 0
        self.is_done = False
        
        return self.current_obs

    def step(self, action: Action) -> StepResponse:
        if self.is_done:
            return StepResponse(
                observation=self.current_obs,
                reward=Reward(score=0.0, breakdown={"penalties": -1.0}),
                done=True,
                info={"error": "Episode already done"}
            )
            
        allowed_actions = ["classify_email", "route_to", "draft_reply", "escalate", "mark_spam", "request_more_info", "noop"]
        if action.action_type not in allowed_actions:
            self.is_done = True
            return StepResponse(
                observation=self.current_obs,
                reward=Reward(score=0.0, breakdown={"penalties": -1.0}),
                done=True,
                info={"error": "Invalid action_type", "reason": f"{action.action_type} is not allowed"}
            )
            
        self.step_count += 1
        self.action_history.append(action)
        self.current_obs.previous_actions = self.current_obs.previous_actions + [action.action_type]
        
        # update state based on action
        if action.action_type == "classify_email":
            val = action.arguments.get('category', 'unknown')
            self.current_obs.current_status = f"classified: {val}"
        elif action.action_type == "route_to":
            val = action.arguments.get('department', 'unknown')
            self.current_obs.current_status = f"routed: {val}"
        elif action.action_type == "draft_reply":
            self.current_obs.current_status = "replied"
        elif action.action_type == "escalate":
            self.current_obs.current_status = "escalated"
        elif action.action_type == "mark_spam":
            self.current_obs.current_status = "spam"
        elif action.action_type == "request_more_info":
            self.current_obs.current_status = "waiting_for_info"
            
        task_data = TASKS[self.task_id]
        
        # Termination conditions
        loop_detected = False
        loop_pattern = None
        if len(self.action_history) >= 4:
            last_four = [json.dumps(a.dict(), sort_keys=True) for a in self.action_history[-4:]]
            if last_four[0] == last_four[2] and last_four[1] == last_four[3]:
                loop_detected = True
                loop_pattern = "alternating_pattern"
        if not loop_detected and len(self.action_history) >= 2:
            last_two = [json.dumps(a.dict(), sort_keys=True) for a in self.action_history[-2:]]
            if len(set(last_two)) == 1:
                loop_detected = True
                loop_pattern = "repeated_action"

        reward, info_additions = calculate_reward(self.task_id, self.action_history, task_data)
        
        history_types = [a.action_type for a in self.action_history]
        task_complete = False
        if self.task_id == "task_1" and action.action_type == "classify_email":
            task_complete = True
        elif self.task_id == "task_2" and action.action_type == "draft_reply":
            if "classify_email" in history_types and "route_to" in history_types:
                task_complete = True
        elif self.task_id == "task_3" and action.action_type == "draft_reply":
            if (
                "classify_email" in history_types and
                "escalate" in history_types and
                "route_to" in history_types
            ):
                task_complete = True
            
        info = {
            "step": self.step_count, 
            "task_id": self.task_id
        }
        
        info.update(info_additions)
            
        if self.step_count >= MAX_STEPS or task_complete or loop_detected:
            self.is_done = True
            if loop_detected:
                reward.score -= 1.0
                penalties = reward.breakdown.get("penalties", 0.0)
                reward.breakdown["penalties"] = max(-1.0, penalties - 1.0)
                info["reason"] = "Loop detected"
                info["loop_pattern"] = loop_pattern
            elif self.step_count >= MAX_STEPS:
                info["reason"] = "Max steps reached"
            elif task_complete:
                info["reason"] = "Task complete"
                
        # Clamp reward bounding after all modifications
        reward.score = max(0.0, min(1.0, float(reward.score)))
                
        return StepResponse(
            observation=self.current_obs,
            reward=reward,
            done=self.is_done,
            info=info
        )
        
    def state(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "step": self.step_count,
            "done": self.is_done,
            "action_history": [a.dict() for a in self.action_history],
            "current_observation": self.current_obs.dict() if self.current_obs else None
        }

app = FastAPI(title="OpenEnv: Enterprise Email Triage & Workflow Simulation")
env = EmailEnv()

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    bad_req_response = StepResponse(
        observation=env.current_obs,
        reward=Reward(score=0.0, breakdown={"penalties": -1.0}),
        done=True,
        info={"error": "Invalid action format received", "details": str(exc)}
    )
    env.is_done = True
    return JSONResponse(status_code=200, content=bad_req_response.dict())

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    bad_req_response = StepResponse(
        observation=env.current_obs, # type: ignore
        reward=Reward(score=0.0, breakdown={"penalties": -1.0}),
        done=True,
        info={"error": "System crash averted", "details": str(exc)}
    )
    env.is_done = True
    return JSONResponse(status_code=200, content=bad_req_response.dict())


@app.post("/reset", response_model=Observation)
def reset_endpoint(req: ResetRequest = None):
    task_id = req.task_id if req and req.task_id else "task_1"
    return env.reset(task_id)

@app.post("/step", response_model=StepResponse)
def step_endpoint(action: Action):
    return env.step(action)

@app.get("/state")
def state_endpoint():
    return env.state()
