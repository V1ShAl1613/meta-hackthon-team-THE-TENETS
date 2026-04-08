from typing import Dict, Any, List, Optional, Literal
from pydantic import BaseModel, Field, field_validator

ActionType = Literal[
    "classify_email", "route_to", "draft_reply",
    "escalate", "mark_spam", "request_more_info", "noop"
]

SCORE_MIN = 0.0001
SCORE_MAX = 0.9999
SCORE_DEFAULT = 0.5

def enforce_valid_score(score: Any) -> float:
    """Clamp arbitrary input into the repo's safe score band [0.01, 0.99]."""
    try:
        score = float(score)
        if score != score:  # NaN check
            return SCORE_DEFAULT
    except (TypeError, ValueError):
        return SCORE_DEFAULT

    if score < SCORE_MIN:
        return SCORE_MIN
    if score > SCORE_MAX:
        return SCORE_MAX
    return score

def validate_scores(scores: List[float]):
    for s in scores:
        if not (SCORE_MIN <= s <= SCORE_MAX):
            raise ValueError(f"Invalid score detected: {s}")

def clamp_score(v: float) -> float:
    return enforce_valid_score(v)

def clamp_breakdown(breakdown: Dict[str, float]) -> Dict[str, float]:
    return {k: enforce_valid_score(v) for k, v in breakdown.items()}

class Observation(BaseModel):
    email_id: str
    subject: str
    sender_type: str
    email_body: str
    urgency_score: float = Field(default=0.5)

    @field_validator("urgency_score", mode="before")
    @classmethod
    def clamp_urgency(cls, v: Any) -> float:
        return enforce_valid_score(v)
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
        return enforce_valid_score(v)

    @field_validator("breakdown", mode="before")
    @classmethod
    def clamp_breakdown_range(cls, v: Any) -> Dict[str, float]:
        if isinstance(v, dict):
            return clamp_breakdown(v)
        return v

class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)

class ResetRequest(BaseModel):
    task_id: Optional[str] = "task_1"
