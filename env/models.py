from typing import Dict, Any, List, Optional, Literal
from pydantic import BaseModel, Field, field_validator

ActionType = Literal[
    "classify_email", "route_to", "draft_reply",
    "escalate", "mark_spam", "request_more_info", "noop"
]

# ── Strict bounds: scores never touch 0.0 or 1.0 ──────────────────
SCORE_MIN = 0.1
SCORE_MAX = 0.9


def clamp_score(v: float) -> float:
    """Clamp score to strictly between 0 and 1 (exclusive).

    Guaranteed range: [SCORE_MIN, SCORE_MAX] == [0.1, 0.9].
    """
    try:
        val = float(v)
    except (TypeError, ValueError):
        return 0.5
    return float(max(SCORE_MIN, min(SCORE_MAX, round(val, 4))))


def clamp_breakdown(breakdown: Dict[str, float]) -> Dict[str, float]:
    """Clamp every value in a breakdown dict to [SCORE_MIN, SCORE_MAX].

    This prevents any metadata/breakdown field from leaking 0.0 or 1.0.
    """
    return {k: float(max(SCORE_MIN, min(SCORE_MAX, round(v, 4)))) for k, v in breakdown.items()}


class Observation(BaseModel):
    email_id: str
    subject: str
    sender_type: str
    email_body: str
    urgency_score: float = Field(default=0.5)

    @field_validator("urgency_score", mode="before")
    @classmethod
    def clamp_urgency(cls, v: Any) -> float:
        return clamp_score(v)
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

    @field_validator("breakdown", mode="before")
    @classmethod
    def clamp_breakdown_range(cls, v: Any) -> Dict[str, float]:
        """Ensure every breakdown value is strictly between 0 and 1."""
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
