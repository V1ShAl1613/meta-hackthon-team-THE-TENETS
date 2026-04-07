from typing import Dict, Any, List, Optional, Literal
from pydantic import BaseModel, Field, field_validator

ActionType = Literal[
    "classify_email", "route_to", "draft_reply",
    "escalate", "mark_spam", "request_more_info", "noop"
]


def clamp_score(v: float) -> float:
    """Clamp score to strictly between 0 and 1 (exclusive)."""
    return float(max(0.01, min(0.99, round(float(v), 4))))


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
    score: float = Field(default=0.5)
    breakdown: Dict[str, float] = Field(default_factory=dict)

    @field_validator("score", mode="before")
    @classmethod
    def clamp_score_range(cls, v: Any) -> float:
        """Ensure score is always strictly between 0 and 1."""
        try:
            v = float(v)
        except (TypeError, ValueError):
            return 0.5
        return clamp_score(v)


class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class ResetRequest(BaseModel):
    task_id: Optional[str] = "task_1"
