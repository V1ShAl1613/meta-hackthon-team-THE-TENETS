from typing import Dict, Any, List, Optional, Literal
from pydantic import BaseModel, Field

ActionType = Literal[
    "classify_email", "route_to", "draft_reply",
    "escalate", "mark_spam", "request_more_info", "noop"
]


class Observation(BaseModel):
    email_id: str
    subject: str
    sender_type: str
    email_body: str
    urgency_score: float = Field(gt=0.0, lt=1.0)
    sentiment: str
    thread_history: List[str] = Field(default_factory=list)
    current_status: str
    previous_actions: List[str] = Field(default_factory=list)


class Action(BaseModel):
    action_type: ActionType
    arguments: Dict[str, Any] = Field(default_factory=dict)


class Reward(BaseModel):
    score: float = Field(gt=0.0, lt=1.0, default=0.01)
    breakdown: Dict[str, float] = Field(default_factory=dict)


class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class ResetRequest(BaseModel):
    task_id: Optional[str] = "task_1"
