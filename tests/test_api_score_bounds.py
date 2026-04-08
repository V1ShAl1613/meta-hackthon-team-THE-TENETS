"""End-to-end API score safety tests.

These tests validate the real FastAPI runtime path used by deployment:
`/reset` + `/step` responses must always produce reward scores strictly in (0, 1).
"""

import random
from typing import Any, Dict, List

from fastapi.testclient import TestClient

from env.models import SCORE_MAX, SCORE_MIN
from env.tasks import TASKS
from env.environment import app


client = TestClient(app)


def _assert_strict_score(value: float, label: str) -> None:
    assert 0.0 < value < 1.0, f"{label}: {value} must be strictly in (0, 1)"
    assert SCORE_MIN <= value <= SCORE_MAX, (
        f"{label}: {value} must be within safe band [{SCORE_MIN}, {SCORE_MAX}]"
    )


def _assert_breakdown(breakdown: Dict[str, Any], label: str) -> None:
    assert isinstance(breakdown, dict), f"{label}: breakdown must be a dict"
    for key, raw_value in breakdown.items():
        _assert_strict_score(float(raw_value), f"{label}.breakdown[{key}]")


def _reset_task(task_id: str) -> Dict[str, Any]:
    response = client.post("/reset", json={"task_id": task_id})
    assert response.status_code == 200
    payload = response.json()
    _assert_strict_score(float(payload["urgency_score"]), f"/reset({task_id}).urgency_score")
    return payload


def _step(action_type: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    response = client.post("/step", json={"action_type": action_type, "arguments": arguments})
    assert response.status_code == 200
    payload = response.json()
    reward = payload["reward"]
    _assert_strict_score(float(reward["score"]), f"/step({action_type}).score")
    _assert_breakdown(reward.get("breakdown", {}), f"/step({action_type})")
    return payload


def test_reset_urgency_scores_are_strictly_between_zero_and_one() -> None:
    for task_id in TASKS:
        _reset_task(task_id)


def test_step_score_bounds_for_representative_task_flows() -> None:
    flows: Dict[str, List[Dict[str, Any]]] = {
        "task_1": [
            {"action_type": "classify_email", "arguments": {"category": "support"}},
            {"action_type": "classify_email", "arguments": {"category": "billing"}},
            {"action_type": "noop", "arguments": {}},
        ],
        "task_2": [
            {"action_type": "classify_email", "arguments": {"category": "sales"}},
            {"action_type": "route_to", "arguments": {"department": "sales_department"}},
            {
                "action_type": "draft_reply",
                "arguments": {"text": "We can provide a demo for your enterprise team upgrade. Thank you."},
            },
        ],
        "task_3": [
            {"action_type": "classify_email", "arguments": {"category": "support"}},
            {"action_type": "route_to", "arguments": {"department": "engineering_escalation"}},
            {"action_type": "escalate", "arguments": {}},
            {
                "action_type": "draft_reply",
                "arguments": {"text": "We are investigating this urgent issue and escalating immediately."},
            },
        ],
    }

    for task_id, steps in flows.items():
        _reset_task(task_id)
        for action in steps:
            payload = _step(action["action_type"], action["arguments"])
            if payload.get("done"):
                break


def test_validation_error_path_still_returns_safe_score() -> None:
    _reset_task("task_1")
    response = client.post("/step", json={"action_type": "not_allowed", "arguments": {}})
    assert response.status_code == 200
    payload = response.json()
    _assert_strict_score(float(payload["reward"]["score"]), "validation_error.score")
    _assert_breakdown(payload["reward"].get("breakdown", {}), "validation_error")


def test_randomized_api_fuzz_preserves_score_bounds() -> None:
    rng = random.Random(42)
    action_templates = [
        ("classify_email", lambda: {"category": rng.choice(["support", "sales", "billing", "other"])}),
        (
            "route_to",
            lambda: {
                "department": rng.choice(
                    [
                        "sales_department",
                        "engineering_escalation",
                        "finance_department",
                        "wrong_dept",
                    ]
                )
            },
        ),
        (
            "draft_reply",
            lambda: {
                "text": rng.choice(
                    [
                        "",
                        "idk",
                        "Please investigate and escalate this urgent issue.",
                        "We can provide a demo for your enterprise team upgrade. Thank you.",
                    ]
                )
            },
        ),
        ("escalate", lambda: {}),
        ("mark_spam", lambda: {}),
        ("request_more_info", lambda: {}),
        ("noop", lambda: {}),
    ]

    episodes_per_task = 250
    max_steps = 8

    for task_id in TASKS:
        for _ in range(episodes_per_task):
            _reset_task(task_id)
            for _ in range(max_steps):
                action_type, arg_factory = rng.choice(action_templates)
                payload = _step(action_type, arg_factory())
                if payload.get("done"):
                    break
