"""Exhaustive test: every possible grading path produces scores strictly in (0, 1).

Run with:  python -m pytest tests/test_score_bounds.py -v
"""
import itertools
import pytest
from env.models import Action, Reward, clamp_score, clamp_breakdown, SCORE_MIN, SCORE_MAX
from env.graders import (
    grade_task_1, grade_task_2, grade_task_3, calculate_reward,
    _evaluate_reply, _check_politeness, _calculate_efficiency_bonus,
)
from env.tasks import TASKS


# ─── helpers ────────────────────────────────────────────────────────
def _assert_strict_bounds(score: float, label: str = ""):
    """Score must be strictly between 0 and 1 (exclusive)."""
    assert score > 0.0, f"{label}: score {score} <= 0.0"
    assert score < 1.0, f"{label}: score {score} >= 1.0"
    assert score >= SCORE_MIN, f"{label}: score {score} < SCORE_MIN ({SCORE_MIN})"
    assert score <= SCORE_MAX, f"{label}: score {score} > SCORE_MAX ({SCORE_MAX})"


def _assert_breakdown_bounds(breakdown: dict, label: str = ""):
    """Every breakdown value must be strictly between 0 and 1."""
    for k, v in breakdown.items():
        assert v > 0.0, f"{label}: breakdown[{k}] = {v} <= 0.0"
        assert v < 1.0, f"{label}: breakdown[{k}] = {v} >= 1.0"


# ─── clamp_score basic tests ───────────────────────────────────────
class TestClampScore:
    @pytest.mark.parametrize("inp,expected", [
        (-10.0, SCORE_MIN),
        (-0.5, SCORE_MIN),
        (0.0, SCORE_MIN),
        (0.05, SCORE_MIN),
        (0.1, SCORE_MIN),
        (0.5, 0.5),
        (0.9, SCORE_MAX),
        (0.95, SCORE_MAX),
        (1.0, SCORE_MAX),
        (1.5, SCORE_MAX),
        (100.0, SCORE_MAX),
    ])
    def test_clamp_values(self, inp, expected):
        result = clamp_score(inp)
        assert result == expected
        _assert_strict_bounds(result, f"clamp_score({inp})")

    def test_clamp_non_numeric(self):
        assert clamp_score("bad") == 0.5
        assert clamp_score(None) == 0.5


# ─── clamp_breakdown tests ─────────────────────────────────────────
class TestClampBreakdown:
    def test_clamps_negatives(self):
        bd = clamp_breakdown({"a": -5.0, "b": 0.0, "c": 1.0, "d": 2.0})
        for k, v in bd.items():
            _assert_strict_bounds(v, f"clamp_breakdown[{k}]")

    def test_preserves_safe_values(self):
        bd = clamp_breakdown({"x": 0.5})
        assert bd["x"] == 0.5


# ─── helper function tests ─────────────────────────────────────────
class TestHelpers:
    def test_evaluate_reply_empty(self):
        score, msg = _evaluate_reply("", ["test"])
        assert score >= 0.0, "Reply score must never be negative"

    def test_evaluate_reply_no_keywords(self):
        score, msg = _evaluate_reply("hello world nothing here", ["missing"])
        assert score >= 0.0, "Reply score must never be negative"

    def test_evaluate_reply_verbose(self):
        text = " ".join(["word"] * 50) + " keyword"
        score, msg = _evaluate_reply(text, ["keyword"])
        assert score >= 0.0, "Reply score must never be negative"

    def test_evaluate_reply_good(self):
        score, msg = _evaluate_reply("demo enterprise team upgrade", ["demo", "enterprise", "team", "upgrade"])
        assert score > 0, "Good reply should score positively"

    def test_politeness_always_positive(self):
        assert _check_politeness("some text") >= 0.0
        assert _check_politeness("please help") >= 0.0

    @pytest.mark.parametrize("steps", range(0, 20))
    def test_efficiency_bonus_bounded(self, steps):
        val = _calculate_efficiency_bonus(steps)
        _assert_strict_bounds(val, f"efficiency(steps={steps})")


# ─── Reward model tests ────────────────────────────────────────────
class TestRewardModel:
    @pytest.mark.parametrize("raw_score", [-1.0, 0.0, 0.001, 0.5, 0.999, 1.0, 2.0])
    def test_reward_score_always_clamped(self, raw_score):
        r = Reward(score=raw_score)
        _assert_strict_bounds(r.score, f"Reward(score={raw_score})")

    def test_reward_breakdown_clamped(self):
        r = Reward(score=0.5, breakdown={"a": -0.5, "b": 0.0, "c": 1.0, "d": 1.5})
        _assert_breakdown_bounds(r.breakdown, "Reward breakdown")


# ─── Task grading — exhaustive action combos ────────────────────────
# Build action sequences that cover every possible grading path

CLASSIFY_CORRECT_T1 = Action(action_type="classify_email", arguments={"category": "support"})
CLASSIFY_WRONG = Action(action_type="classify_email", arguments={"category": "billing"})
ROUTE_SALES = Action(action_type="route_to", arguments={"department": "sales_department"})
ROUTE_WRONG = Action(action_type="route_to", arguments={"department": "wrong_dept"})
ROUTE_ENG = Action(action_type="route_to", arguments={"department": "engineering_escalation"})
ESCALATE = Action(action_type="escalate", arguments={})
DRAFT_GOOD_T2 = Action(action_type="draft_reply", arguments={"text": "We'd be happy to provide a demo for your enterprise team upgrade."})
DRAFT_GOOD_T3 = Action(action_type="draft_reply", arguments={"text": "We are investigating this urgent issue and will escalate immediately."})
DRAFT_BAD = Action(action_type="draft_reply", arguments={"text": "idk"})
DRAFT_EMPTY = Action(action_type="draft_reply", arguments={"text": ""})
NOOP = Action(action_type="noop", arguments={})
SPAM = Action(action_type="mark_spam", arguments={})


class TestTask1Grading:
    task_data = TASKS["task_1"]

    @pytest.mark.parametrize("actions", [
        [],
        [CLASSIFY_CORRECT_T1],
        [CLASSIFY_WRONG],
        [CLASSIFY_CORRECT_T1, CLASSIFY_CORRECT_T1],  # duplicate
        [CLASSIFY_WRONG, CLASSIFY_CORRECT_T1],
        [NOOP] * 8,  # max steps, all noop
        [CLASSIFY_WRONG, CLASSIFY_WRONG, CLASSIFY_WRONG],  # triple wrong
    ])
    def test_score_bounds(self, actions):
        reward, info = grade_task_1(actions, self.task_data)
        _assert_strict_bounds(reward.score, f"task_1 actions={[a.action_type for a in actions]}")
        _assert_breakdown_bounds(reward.breakdown, f"task_1 breakdown")


class TestTask2Grading:
    task_data = TASKS["task_2"]

    @pytest.mark.parametrize("actions", [
        [],
        [Action(action_type="classify_email", arguments={"category": "sales"})],
        [Action(action_type="classify_email", arguments={"category": "sales"}), ROUTE_SALES],
        [Action(action_type="classify_email", arguments={"category": "sales"}), ROUTE_SALES, DRAFT_GOOD_T2],
        [CLASSIFY_WRONG, ROUTE_WRONG, DRAFT_BAD],
        [CLASSIFY_WRONG, ROUTE_WRONG, DRAFT_EMPTY],
        [Action(action_type="classify_email", arguments={"category": "sales"}), ROUTE_SALES, DRAFT_GOOD_T2, NOOP, NOOP],
        [NOOP] * 8,
        # Double classification + double routing
        [Action(action_type="classify_email", arguments={"category": "sales"}),
         Action(action_type="classify_email", arguments={"category": "sales"}),
         ROUTE_SALES, ROUTE_SALES, DRAFT_GOOD_T2],
    ])
    def test_score_bounds(self, actions):
        reward, info = grade_task_2(actions, self.task_data)
        _assert_strict_bounds(reward.score, f"task_2 actions={[a.action_type for a in actions]}")
        _assert_breakdown_bounds(reward.breakdown, f"task_2 breakdown")


class TestTask3Grading:
    task_data = TASKS["task_3"]

    @pytest.mark.parametrize("actions", [
        [],
        [CLASSIFY_CORRECT_T1],
        [CLASSIFY_CORRECT_T1, ROUTE_ENG],
        [CLASSIFY_CORRECT_T1, ROUTE_ENG, ESCALATE],
        [CLASSIFY_CORRECT_T1, ROUTE_ENG, ESCALATE, DRAFT_GOOD_T3],  # perfect run
        [CLASSIFY_WRONG, ROUTE_WRONG, DRAFT_BAD],
        [CLASSIFY_WRONG, ROUTE_WRONG, DRAFT_EMPTY],
        [NOOP] * 8,
        # no escalation when required
        [CLASSIFY_CORRECT_T1, ROUTE_ENG, DRAFT_GOOD_T3],
        # triple everything wrong
        [CLASSIFY_WRONG, CLASSIFY_WRONG, ROUTE_WRONG, ROUTE_WRONG, ESCALATE, ESCALATE, DRAFT_BAD, DRAFT_EMPTY],
    ])
    def test_score_bounds(self, actions):
        reward, info = grade_task_3(actions, self.task_data)
        _assert_strict_bounds(reward.score, f"task_3 actions={[a.action_type for a in actions]}")
        _assert_breakdown_bounds(reward.breakdown, f"task_3 breakdown")


class TestCalculateReward:
    def test_invalid_task_id(self):
        reward, info = calculate_reward("task_999", [], {})
        _assert_strict_bounds(reward.score, "invalid task_id")
        _assert_breakdown_bounds(reward.breakdown, "invalid task_id breakdown")


# ─── Stress test: random-ish long sequences ─────────────────────────
class TestStress:
    ALL_ACTIONS = [
        CLASSIFY_CORRECT_T1, CLASSIFY_WRONG, ROUTE_SALES, ROUTE_WRONG,
        ROUTE_ENG, ESCALATE, DRAFT_GOOD_T2, DRAFT_GOOD_T3, DRAFT_BAD,
        DRAFT_EMPTY, NOOP, SPAM,
    ]

    @pytest.mark.parametrize("task_id", ["task_1", "task_2", "task_3"])
    def test_all_single_actions(self, task_id):
        task_data = TASKS[task_id]
        for act in self.ALL_ACTIONS:
            reward, info = calculate_reward(task_id, [act], task_data)
            _assert_strict_bounds(reward.score, f"{task_id} single {act.action_type}")
            _assert_breakdown_bounds(reward.breakdown, f"{task_id} single {act.action_type} breakdown")

    @pytest.mark.parametrize("task_id", ["task_1", "task_2", "task_3"])
    def test_long_noop_sequence(self, task_id):
        task_data = TASKS[task_id]
        actions = [NOOP] * 20
        reward, info = calculate_reward(task_id, actions, task_data)
        _assert_strict_bounds(reward.score, f"{task_id} 20x noop")
        _assert_breakdown_bounds(reward.breakdown, f"{task_id} 20x noop breakdown")

    @pytest.mark.parametrize("task_id", ["task_1", "task_2", "task_3"])
    def test_all_wrong_sequence(self, task_id):
        task_data = TASKS[task_id]
        actions = [CLASSIFY_WRONG, ROUTE_WRONG, DRAFT_BAD, DRAFT_EMPTY] * 3
        reward, info = calculate_reward(task_id, actions, task_data)
        _assert_strict_bounds(reward.score, f"{task_id} all-wrong")
        _assert_breakdown_bounds(reward.breakdown, f"{task_id} all-wrong breakdown")
