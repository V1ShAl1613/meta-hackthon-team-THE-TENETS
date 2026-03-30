from typing import Dict, Any, List, Optional, Literal
from pydantic import BaseModel, Field

ActionType = Literal[
    "classify_email", "route_to", "draft_reply", 
    "escalate", "mark_spam", "request_more_info", "noop"
]

class Observation(BaseModel):
    email_id: str
    subject: str
    sender_type: str  # customer, spam, internal, VIP
    email_body: str
    urgency_score: float  # 0 to 1
    sentiment: str
    thread_history: List[str]
    current_status: str
    previous_actions: List[str]

class Action(BaseModel):
    action_type: ActionType
    arguments: Dict[str, Any] = Field(default_factory=dict)

class Reward(BaseModel):
    score: float
    breakdown: Dict[str, float]

class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any]

class ResetRequest(BaseModel):
    task_id: Optional[str] = "task_1"
