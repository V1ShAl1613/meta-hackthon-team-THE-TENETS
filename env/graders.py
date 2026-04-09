import re
from typing import Dict, Any, List, Optional, Tuple
from .models import (
    Reward,
    Action,
    clamp_score,
    clamp_breakdown,
    SCORE_MIN,
    SCORE_MAX,
    enforce_valid_score,
    is_strict_score,
)


MAX_STEPS = 8
STEP_PENALTY = 0.03
POLITE_WORDS = ["please", "thank", "appreciate", "sorry", "apologies"]


def _strict_score(value: float) -> float:
    """Normalize a score and guarantee it stays in the strict open interval."""
    score = enforce_valid_score(value)
    if not is_strict_score(score):
        raise ValueError(f"Score escaped strict bounds: {score}")
    return score


def _strip_punctuation(text: str) -> str:
    """Remove punctuation so keyword matching isn't broken by trailing periods, commas, etc."""
    return re.sub(r"[^\w\s]", "", text)


def _calculate_efficiency_bonus(steps: int) -> float:
    return _strict_score(0.9 - (steps / MAX_STEPS))


def _check_politeness(text: str) -> float:
    cleaned = _strip_punctuation(text.lower())
    for word in POLITE_WORDS:
        if word in cleaned:
            return 0.1
    return 0.1


def _evaluate_reply(text: str, keywords: List[str]) -> Tuple[Optional[float], str]:
    """Evaluate reply quality. Returns (score_delta, mistake_message).

    Invalid replies return ``None`` so downstream code can avoid treating
    them as a real score while still preserving the mistake message.
    """
    cleaned = _strip_punctuation(text.lower())
    words = cleaned.split()
    if not words:
        return None, "Reply is empty."

    words_set = set(words)
    matches = sum(1 for kw in keywords if kw.lower() in words_set)
    if matches == 0:
        return None, "Reply missing required keywords."

    relevance = matches / len(words)
    if relevance < 0.05 and len(words) > 30:
        return None, "Reply is too verbose or lacks density."

    base = 0.3
    bonus = min(0.2, matches * 0.05)
    return _strict_score(base + bonus), ""


def _safe_reward(score: float, breakdown: Dict[str, float]) -> Reward:
    """Build a Reward with all values clamped. Single exit point for safety."""
    reward = Reward(score=_strict_score(score), breakdown=clamp_breakdown(breakdown))
    if not is_strict_score(reward.score):
        raise ValueError(f"Reward score escaped strict bounds: {reward.score}")
    for key, value in reward.breakdown.items():
        if not (SCORE_MIN <= value <= SCORE_MAX):
            raise ValueError(f"Reward breakdown escaped safe bounds: {key}={value}")
    return reward


def grade_task_1(action_history: List[Action], task_data: Dict[str, Any]) -> Tuple[Reward, Dict[str, str]]:
    target = task_data["target_classification"]

    score = 0.1
    breakdown = {
        "classification": 0.1,
        "routing": 0.1,
        "reply_quality": 0.1,
        "efficiency": 0.1,
        "bonus": 0.1,
        "penalties": 0.1
    }
    info_dict = {}
    classifications = 0

    for act in action_history:
        if act.action_type == "classify_email":
            classifications += 1
            cat = act.arguments.get("category", "")
            if cat and cat.strip().lower() == target.lower():
                if classifications == 1:
                    score = enforce_valid_score(score + 0.6)
                    breakdown["classification"] = 0.7
                    breakdown["bonus"] = 0.2
                    score = enforce_valid_score(score + 0.1)
            else:
                score = enforce_valid_score(max(score - 0.15, SCORE_MIN))
                breakdown["penalties"] = 0.3
                info_dict["mistake"] = f"Incorrect classification: {cat}"
                info_dict["expected"] = target
                info_dict["suggestion"] = "Ensure you classify the email correctly based on intent."

    if classifications > 1:
        score = enforce_valid_score(max(score - 0.15, SCORE_MIN))
        breakdown["penalties"] = 0.35

    steps = len(action_history)
    step_penalty = steps * STEP_PENALTY
    score = enforce_valid_score(max(score - step_penalty, SCORE_MIN))

    if breakdown["classification"] > 0.1:
        eff = _calculate_efficiency_bonus(steps) * 0.2
        score = enforce_valid_score(score + eff)
        breakdown["efficiency"] = clamp_score(eff)

    if not info_dict and breakdown["classification"] > 0.1:
        info_dict["suggestion"] = "Great job!"

    score = enforce_valid_score(score)
    return _safe_reward(score, breakdown), info_dict


def grade_task_2(action_history: List[Action], task_data: Dict[str, Any]) -> Tuple[Reward, Dict[str, str]]:
    target_class = task_data["target_classification"]
    target_route = task_data["target_routing"]
    keywords = task_data["reply_keywords"]

    score = 0.1
    breakdown = {
        "classification": 0.1,
        "routing": 0.1,
        "reply_quality": 0.1,
        "efficiency": 0.1,
        "bonus": 0.1,
        "penalties": 0.1
    }
    info_dict = {}

    classification_correct = False
    routing_correct = False
    reply_correct = False

    classifications = 0
    routings = 0

    for act in action_history:
        if act.action_type == "classify_email":
            classifications += 1
            val = act.arguments.get("category", "")
            if val and val.strip().lower() == target_class.lower():
                if classifications == 1:
                    score = enforce_valid_score(score + 0.2)
                    breakdown["classification"] = 0.3
                    classification_correct = True
            else:
                score = enforce_valid_score(max(score - 0.1, SCORE_MIN))
                breakdown["penalties"] = 0.2
                info_dict["mistake"] = f"Incorrect classification: {val}"
                info_dict["expected"] = target_class

        elif act.action_type == "route_to":
            routings += 1
            val = act.arguments.get("department", "")
            if val and val.strip().lower() == target_route.lower():
                if routings == 1:
                    score = enforce_valid_score(score + 0.2)
                    breakdown["routing"] = 0.3
                    routing_correct = True
            else:
                score = enforce_valid_score(max(score - 0.1, SCORE_MIN))
                breakdown["penalties"] = 0.2
                if "mistake" not in info_dict:
                    info_dict["mistake"] = f"Incorrect routing: {val}"
                    info_dict["expected"] = target_route

        elif act.action_type == "draft_reply":
            text = act.arguments.get("text", "").lower()
            reply_score, mistake = _evaluate_reply(text, keywords)

            if reply_score is not None and reply_score > 0:
                score = enforce_valid_score(score + reply_score)
                breakdown["reply_quality"] = clamp_score(0.1 + reply_score)
                reply_correct = True
            else:
                if "mistake" not in info_dict:
                    info_dict["mistake"] = mistake

            polite = _check_politeness(text)
            score = enforce_valid_score(score + polite)
            breakdown["bonus"] = clamp_score(breakdown.get("bonus", 0.1) + polite)

    if classifications > 1 or routings > 1:
        score = enforce_valid_score(max(score - 0.15, SCORE_MIN))
        breakdown["penalties"] = 0.3

    steps = len(action_history)
    step_penalty = steps * STEP_PENALTY
    score = enforce_valid_score(max(score - step_penalty, SCORE_MIN))

    if classification_correct and routing_correct and reply_correct:
        eff = _calculate_efficiency_bonus(steps) * 0.1
        score = enforce_valid_score(score + eff)
        breakdown["efficiency"] = clamp_score(eff)

        bonus = 0.1
        breakdown["bonus"] = clamp_score(breakdown.get("bonus", 0.1) + bonus)
        score = enforce_valid_score(score + bonus)

        info_dict["suggestion"] = "Perfect workflow complete."
    elif not info_dict:
        info_dict["suggestion"] = "Ensure all required steps are completed appropriately."

    score = enforce_valid_score(score)
    return _safe_reward(score, breakdown), info_dict


def grade_task_3(action_history: List[Action], task_data: Dict[str, Any]) -> Tuple[Reward, Dict[str, str]]:
    target_class = task_data["target_classification"]
    target_route = task_data["target_routing"]
    keywords = task_data["reply_keywords"]

    score = 0.1
    breakdown = {
        "classification": 0.1,
        "routing": 0.1,
        "reply_quality": 0.1,
        "efficiency": 0.1,
        "bonus": 0.1,
        "penalties": 0.1
    }
    info_dict = {}

    classification_correct = False
    routing_correct = False
    escalation_correct = False
    reply_correct = False

    classifications = 0
    routings = 0
    escalations = 0

    for act in action_history:
        if act.action_type == "classify_email":
            classifications += 1
            val = act.arguments.get("category", "")
            if val and val.strip().lower() == target_class.lower():
                if classifications == 1:
                    score = enforce_valid_score(score + 0.15)
                    breakdown["classification"] = 0.25
                    classification_correct = True
            else:
                score = enforce_valid_score(max(score - 0.1, SCORE_MIN))
                breakdown["penalties"] = 0.2
                info_dict["mistake"] = f"Incorrect classification: {val}"
                info_dict["expected"] = target_class

        elif act.action_type == "route_to":
            routings += 1
            val = act.arguments.get("department", "")
            if val and val.strip().lower() == target_route.lower():
                if routings == 1:
                    score = enforce_valid_score(score + 0.15)
                    breakdown["routing"] = 0.25
                    routing_correct = True
            else:
                score = enforce_valid_score(max(score - 0.1, SCORE_MIN))
                breakdown["penalties"] = 0.2
                if "mistake" not in info_dict:
                    info_dict["mistake"] = f"Incorrect routing: {val}"

        elif act.action_type == "escalate":
            escalations += 1
            if task_data.get("must_escalate"):
                if escalations == 1:
                    score = enforce_valid_score(score + 0.15)
                    breakdown["bonus"] = clamp_score(breakdown.get("bonus", 0.1) + 0.15)
                    escalation_correct = True

        elif act.action_type == "draft_reply":
            text = act.arguments.get("text", "").lower()
            reply_score, mistake = _evaluate_reply(text, keywords)

            if reply_score is not None and reply_score > 0:
                score = enforce_valid_score(score + reply_score)
                breakdown["reply_quality"] = clamp_score(0.1 + reply_score)
                reply_correct = True
            else:
                if "mistake" not in info_dict:
                    info_dict["mistake"] = mistake

            polite = _check_politeness(text)
            score = enforce_valid_score(score + polite)
            breakdown["bonus"] = clamp_score(breakdown.get("bonus", 0.1) + polite)

    if task_data.get("must_escalate") and escalations == 0:
        score = enforce_valid_score(max(score - 0.15, SCORE_MIN))
        breakdown["penalties"] = 0.3

    if classifications > 1 or routings > 1 or escalations > 1:
        score = enforce_valid_score(max(score - 0.15, SCORE_MIN))
        breakdown["penalties"] = 0.35

    steps = len(action_history)
    step_penalty = steps * STEP_PENALTY
    score = enforce_valid_score(max(score - step_penalty, SCORE_MIN))

    if classification_correct and routing_correct and escalation_correct and reply_correct:
        eff = _calculate_efficiency_bonus(steps) * 0.1
        score = enforce_valid_score(score + eff)
        breakdown["efficiency"] = clamp_score(eff)

        breakdown["bonus"] = clamp_score(breakdown.get("bonus", 0.1) + 0.1)
        score = enforce_valid_score(score + 0.1)
        info_dict["suggestion"] = "Perfect escalation workflow complete."
    elif not info_dict:
        info_dict["suggestion"] = "Ensure VIP emails are escalated explicitly."

    score = enforce_valid_score(score)
    return _safe_reward(score, breakdown), info_dict


def calculate_reward(task_id: str, action_history: List[Action], task_data: Dict[str, Any]) -> Tuple[Reward, Dict[str, str]]:
    if task_id == "task_1":
        return grade_task_1(action_history, task_data)
    elif task_id == "task_2":
        return grade_task_2(action_history, task_data)
    elif task_id == "task_3":
        return grade_task_3(action_history, task_data)
    else:
        return _safe_reward(0.1, {"penalties": 0.1}), {"error": "Invalid task ID"}
