import copy
import json
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from .models import Observation, Action, Reward, StepResponse, ResetRequest, clamp_score
from .tasks import TASKS
from .graders import calculate_reward, MAX_STEPS

ALLOWED_ACTIONS = [
    "classify_email", "route_to", "draft_reply",
    "escalate", "mark_spam", "request_more_info", "noop"
]


class EmailEnv:
    def __init__(self):
        self.task_id: str = "task_1"
        self.current_obs: Observation = copy.deepcopy(TASKS[self.task_id]["initial_state"])
        self.action_history: List[Action] = []
        self.step_count: int = 0
        self.is_done: bool = False

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
                reward=Reward(score=0.01, breakdown={"penalties": -0.99}),
                done=True,
                info={"error": "Episode already done"}
            )

        if action.action_type not in ALLOWED_ACTIONS:
            self.is_done = True
            return StepResponse(
                observation=self.current_obs,
                reward=Reward(score=0.01, breakdown={"penalties": -0.99}),
                done=True,
                info={"error": "Invalid action_type", "reason": f"{action.action_type} is not allowed"}
            )

        self.step_count += 1
        self.action_history.append(action)
        self.current_obs.previous_actions = self.current_obs.previous_actions + [action.action_type]

        # Update state based on action
        if action.action_type == "classify_email":
            val = action.arguments.get("category", "unknown")
            self.current_obs.current_status = f"classified: {val}"
        elif action.action_type == "route_to":
            val = action.arguments.get("department", "unknown")
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

        # Loop detection
        loop_detected = False
        loop_pattern = None

        def _action_key(a: Action) -> str:
            return json.dumps({"action_type": a.action_type, "arguments": a.arguments}, sort_keys=True)

        if len(self.action_history) >= 4:
            last_four = [_action_key(a) for a in self.action_history[-4:]]
            if last_four[0] == last_four[2] and last_four[1] == last_four[3]:
                loop_detected = True
                loop_pattern = "alternating_pattern"
        if not loop_detected and len(self.action_history) >= 2:
            last_two = [_action_key(a) for a in self.action_history[-2:]]
            if last_two[0] == last_two[1]:
                loop_detected = True
                loop_pattern = "repeated_action"

        reward, info_additions = calculate_reward(self.task_id, self.action_history, task_data)

        # Task completion checks
        history_types = [a.action_type for a in self.action_history]
        task_complete = False
        if self.task_id == "task_1" and action.action_type == "classify_email":
            task_complete = True
        elif self.task_id == "task_2" and action.action_type == "draft_reply":
            if "classify_email" in history_types and "route_to" in history_types:
                task_complete = True
        elif self.task_id == "task_3" and action.action_type == "draft_reply":
            if (
                "classify_email" in history_types
                and "escalate" in history_types
                and "route_to" in history_types
            ):
                task_complete = True

        info: Dict[str, Any] = {
            "step": self.step_count,
            "task_id": self.task_id
        }
        info.update(info_additions)

        if self.step_count >= MAX_STEPS or task_complete or loop_detected:
            self.is_done = True
            if loop_detected:
                reward.score = clamp_score(reward.score - 0.5)
                penalties = reward.breakdown.get("penalties", 0.1)
                reward.breakdown["penalties"] = max(0.05, penalties + 0.5)
                info["reason"] = "Loop detected"
                info["loop_pattern"] = loop_pattern
            elif self.step_count >= MAX_STEPS:
                info["reason"] = "Max steps reached"
            elif task_complete:
                info["reason"] = "Task complete"

        # Final safety clamp: Ensure score is strictly between 0 and 1
        reward.score = clamp_score(reward.score)

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
            "action_history": [{"action_type": a.action_type, "arguments": a.arguments} for a in self.action_history],
            "current_observation": self.current_obs.model_dump() if hasattr(self.current_obs, "model_dump") else self.current_obs.dict()
        }


app = FastAPI(title="OpenEnv: Enterprise Email Triage & Workflow Simulation")
env = EmailEnv()


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    response = StepResponse(
        observation=env.current_obs,
        reward=Reward(score=0.1, breakdown={"penalties": 0.5}),
        done=True,
        info={"error": "Invalid action format received", "details": str(exc)}
    )
    env.is_done = True
    payload = response.model_dump() if hasattr(response, "model_dump") else response.dict()
    return JSONResponse(status_code=200, content=payload)


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    response = StepResponse(
        observation=env.current_obs,
        reward=Reward(score=0.1, breakdown={"penalties": 0.5}),
        done=True,
        info={"error": "System crash averted", "details": str(exc)}
    )
    env.is_done = True
    payload = response.model_dump() if hasattr(response, "model_dump") else response.dict()
    return JSONResponse(status_code=200, content=payload)


@app.post("/reset")
def reset_endpoint(req: Optional[ResetRequest] = None):
    task_id = req.task_id if req and req.task_id else "task_1"
    obs = env.reset(task_id)
    return obs.model_dump() if hasattr(obs, "model_dump") else obs.dict()


@app.post("/step")
def step_endpoint(action: Action):
    result = env.step(action)
    return result.model_dump() if hasattr(result, "model_dump") else result.dict()


@app.get("/state")
def state_endpoint():
    return env.state()

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Enterprise Email Triage Environment Operational"}
