import re
from typing import Dict, Any, List, Tuple
from .models import Reward, Action, clamp_score

MAX_STEPS = 8
STEP_PENALTY = 0.03
POLITE_WORDS = ["please", "thank", "appreciate", "sorry", "apologies"]


def _strip_punctuation(text: str) -> str:
    """Remove punctuation so keyword matching isn't broken by trailing periods, commas, etc."""
    return re.sub(r"[^\w\s]", "", text)


def _calculate_efficiency_bonus(steps: int) -> float:
    return clamp_score(0.9 - (steps / MAX_STEPS))


def _check_politeness(text: str) -> float:
    cleaned = _strip_punctuation(text.lower())
    for word in POLITE_WORDS:
        if word in cleaned:
            return 0.1
    return 0.1


def _evaluate_reply(text: str, keywords: List[str]) -> Tuple[float, str]:
    cleaned = _strip_punctuation(text.lower())
    words = cleaned.split()
    if not words:
        return -0.2, "Reply is empty."

    words_set = set(words)
    matches = sum(1 for kw in keywords if kw.lower() in words_set)
    if matches == 0:
        return -0.3, "Reply missing required keywords."

    relevance = matches / len(words)
    if relevance < 0.05 and len(words) > 30:
        return -0.1, "Reply is too verbose or lacks density."

    base = 0.3
    bonus = min(0.2, matches * 0.05)
    return base + bonus, ""


def grade_task_1(action_history: List[Action], task_data: Dict[str, Any]) -> Tuple[Reward, Dict[str, str]]:
    target = task_data["target_classification"]

    score = 0.1
    breakdown = {
        "classification": 0.1,
        "routing": 0.1,
        "reply_quality": 0.1,
        "efficiency": 0.1,
        "bonus": 0.1,
        "penalties": 0.0
    }
    info_dict = {}
    classifications = 0

    for act in action_history:
        if act.action_type == "classify_email":
            classifications += 1
            cat = act.arguments.get("category", "")
            if cat and cat.strip().lower() == target.lower():
                if classifications == 1:
                    score += 0.8
                    breakdown["classification"] = 0.8
                    breakdown["bonus"] += 0.1
                    score += 0.1
            else:
                score -= 0.2
                breakdown["penalties"] -= 0.2
                info_dict["mistake"] = f"Incorrect classification: {cat}"
                info_dict["expected"] = target
                info_dict["suggestion"] = "Ensure you classify the email correctly based on intent."

    if classifications > 1:
        score -= 0.2
        breakdown["penalties"] -= 0.2

    steps = len(action_history)
    step_penalty = steps * STEP_PENALTY
    score -= step_penalty
    breakdown["penalties"] -= step_penalty

    if breakdown["classification"] > 0:
        eff = _calculate_efficiency_bonus(steps) * 0.2
        score += eff
        breakdown["efficiency"] = eff

    if not info_dict and breakdown["classification"] > 0:
        info_dict["suggestion"] = "Great job!"

    score = clamp_score(score)
    return Reward(score=score, breakdown=breakdown), info_dict


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
        "penalties": 0.0
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
                    score += 0.2
                    breakdown["classification"] = 0.2
                    classification_correct = True
            else:
                score -= 0.1
                breakdown["penalties"] -= 0.1
                info_dict["mistake"] = f"Incorrect classification: {val}"
                info_dict["expected"] = target_class
        elif act.action_type == "route_to":
            routings += 1
            val = act.arguments.get("department", "")
            if val and val.strip().lower() == target_route.lower():
                if routings == 1:
                    score += 0.2
                    breakdown["routing"] = 0.2
                    routing_correct = True
            else:
                score -= 0.1
                breakdown["penalties"] -= 0.1
                if "mistake" not in info_dict:
                    info_dict["mistake"] = f"Incorrect routing: {val}"
                    info_dict["expected"] = target_route
        elif act.action_type == "draft_reply":
            text = act.arguments.get("text", "").lower()
            reply_score, mistake = _evaluate_reply(text, keywords)

            if reply_score > 0:
                score += reply_score
                breakdown["reply_quality"] += reply_score
                reply_correct = True
            else:
                score += reply_score
                breakdown["penalties"] += reply_score
                if "mistake" not in info_dict:
                    info_dict["mistake"] = mistake

            polite = _check_politeness(text)
            score += polite
            breakdown["bonus"] += polite

    if classifications > 1 or routings > 1:
        score -= 0.2
        breakdown["penalties"] -= 0.2

    steps = len(action_history)
    step_penalty = steps * STEP_PENALTY
    score -= step_penalty
    breakdown["penalties"] -= step_penalty

    if classification_correct and routing_correct and reply_correct:
        eff = _calculate_efficiency_bonus(steps) * 0.1
        score += eff
        breakdown["efficiency"] = eff

        bonus = 0.1
        breakdown["bonus"] += bonus
        score += bonus

        info_dict["suggestion"] = "Perfect workflow complete."
    elif not info_dict:
        info_dict["suggestion"] = "Ensure all required steps are completed appropriately."

    score = clamp_score(score)
    return Reward(score=score, breakdown=breakdown), info_dict


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
        "penalties": 0.0
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
                    score += 0.15
                    breakdown["classification"] = 0.15
                    classification_correct = True
            else:
                score -= 0.1
                breakdown["penalties"] -= 0.1
                info_dict["mistake"] = f"Incorrect classification: {val}"
                info_dict["expected"] = target_class
        elif act.action_type == "route_to":
            routings += 1
            val = act.arguments.get("department", "")
            if val and val.strip().lower() == target_route.lower():
                if routings == 1:
                    score += 0.15
                    breakdown["routing"] = 0.15
                    routing_correct = True
            else:
                score -= 0.1
                breakdown["penalties"] -= 0.1
                if "mistake" not in info_dict:
                    info_dict["mistake"] = f"Incorrect routing: {val}"
        elif act.action_type == "escalate":
            escalations += 1
            if task_data.get("must_escalate"):
                if escalations == 1:
                    score += 0.2
                    breakdown["bonus"] += 0.2
                    escalation_correct = True
        elif act.action_type == "draft_reply":
            text = act.arguments.get("text", "").lower()
            reply_score, mistake = _evaluate_reply(text, keywords)

            if reply_score > 0:
                score += reply_score
                breakdown["reply_quality"] += reply_score
                reply_correct = True
            else:
                score += reply_score
                breakdown["penalties"] += reply_score
                if "mistake" not in info_dict:
                    info_dict["mistake"] = mistake

            polite = _check_politeness(text)
            score += polite
            breakdown["bonus"] += polite

    if task_data.get("must_escalate") and escalations == 0:
        score -= 0.2
        breakdown["penalties"] -= 0.2

    if classifications > 1 or routings > 1 or escalations > 1:
        score -= 0.2
        breakdown["penalties"] -= 0.2

    steps = len(action_history)
    step_penalty = steps * STEP_PENALTY
    score -= step_penalty
    breakdown["penalties"] -= step_penalty

    if classification_correct and routing_correct and escalation_correct and reply_correct:
        eff = _calculate_efficiency_bonus(steps) * 0.1
        score += eff
        breakdown["efficiency"] = eff
        breakdown["bonus"] += 0.1
        score += 0.1
        info_dict["suggestion"] = "Perfect escalation workflow complete."
    elif not info_dict:
        info_dict["suggestion"] = "Ensure VIP emails are escalated explicitly."

    score = clamp_score(score)
    return Reward(score=score, breakdown=breakdown), info_dict


def calculate_reward(task_id: str, action_history: List[Action], task_data: Dict[str, Any]) -> Tuple[Reward, Dict[str, str]]:
    if task_id == "task_1":
        return grade_task_1(action_history, task_data)
    elif task_id == "task_2":
        return grade_task_2(action_history, task_data)
    elif task_id == "task_3":
        return grade_task_3(action_history, task_data)
    else:
        return Reward(score=0.1, breakdown={"penalties": 0.05}), {"error": "Invalid task ID"}
