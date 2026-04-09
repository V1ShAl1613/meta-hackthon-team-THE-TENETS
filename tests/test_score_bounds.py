"""Verify task scores are binary while breakdown values stay in the safe score band."""

import pytest

from env.graders import (
    _calculate_efficiency_bonus,
    _check_politeness,
    _evaluate_reply,
    calculate_reward,
    grade_task_1,
    grade_task_2,
    grade_task_3,
)
from env.models import (
    Action,
    Reward,
    SCORE_MAX,
    SCORE_MIN,
    clamp_breakdown,
    clamp_score,
    coerce_binary_score,
    is_binary_score,
)
from env.tasks import TASKS


def _assert_binary_score(score: int, label: str = "") -> None:
    assert score in (0, 1), f"{label}: {score} must be binary"
    assert is_binary_score(score), f"{label}: {score} must satisfy the binary score contract"


def _assert_fraction_band(score: float, label: str = "") -> None:
    assert SCORE_MIN <= score <= SCORE_MAX, (
        f"{label}: {score} must stay inside the repo score band [{SCORE_MIN}, {SCORE_MAX}]"
    )


def _assert_breakdown_band(breakdown: dict, label: str = "") -> None:
    for key, value in breakdown.items():
        _assert_fraction_band(value, f"{label} breakdown[{key}]")


CLASSIFY_SUPPORT = Action(action_type="classify_email", arguments={"category": "support"})
CLASSIFY_SALES = Action(action_type="classify_email", arguments={"category": "sales"})
CLASSIFY_WRONG = Action(action_type="classify_email", arguments={"category": "billing"})
ROUTE_SALES = Action(action_type="route_to", arguments={"department": "sales_department"})
ROUTE_ENGINEERING = Action(
    action_type="route_to", arguments={"department": "engineering_escalation"}
)
ROUTE_WRONG = Action(action_type="route_to", arguments={"department": "wrong_dept"})
ESCALATE = Action(action_type="escalate", arguments={})
DRAFT_GOOD_T2 = Action(
    action_type="draft_reply",
    arguments={"text": "We would be happy to provide a demo for your enterprise team upgrade."},
)
DRAFT_GOOD_T3 = Action(
    action_type="draft_reply",
    arguments={"text": "We are investigating this urgent issue and will escalate immediately."},
)
DRAFT_BAD = Action(action_type="draft_reply", arguments={"text": "idk"})
DRAFT_EMPTY = Action(action_type="draft_reply", arguments={"text": ""})
NOOP = Action(action_type="noop", arguments={})
SPAM = Action(action_type="mark_spam", arguments={})


class TestClampScore:
    @pytest.mark.parametrize(
        ("raw_value", "expected"),
        [
            (-10.0, SCORE_MIN),
            (-0.5, SCORE_MIN),
            (0, SCORE_MIN),
            (0.00001, SCORE_MIN),
            (0.001, SCORE_MIN),
            (0.01, 0.01),
            (0.05, 0.05),
            (0.5, 0.5),
            (0.95, 0.95),
            (0.99, 0.99),
            (0.999, SCORE_MAX),
            (0.99999, SCORE_MAX),
            (1, SCORE_MAX),
            (2.0, SCORE_MAX),
        ],
    )
    def test_clamp_values(self, raw_value: float, expected: float) -> None:
        result = clamp_score(raw_value)
        assert result == expected
        _assert_fraction_band(result, f"clamp_score({raw_value})")

    def test_clamp_non_numeric_values(self) -> None:
        assert clamp_score("bad") == 0.5
        assert clamp_score(None) == 0.5

    def test_binary_score_matches_contract(self) -> None:
        assert is_binary_score(0)
        assert is_binary_score(1)
        assert not is_binary_score(0.5)
        assert not is_binary_score(SCORE_MIN)
        assert not is_binary_score(SCORE_MAX)

    @pytest.mark.parametrize(
        ("raw_value", "expected"),
        [
            (-10.0, 0),
            (0, 0),
            (0.49, 0),
            (0.5, 1),
            (0.99, 1),
            (1, 1),
            (2.0, 1),
        ],
    )
    def test_coerce_binary_score(self, raw_value: float, expected: int) -> None:
        assert coerce_binary_score(raw_value) == expected


class TestClampBreakdown:
    def test_clamps_breakdown_entries(self) -> None:
        breakdown = clamp_breakdown({"a": -5.0, "b": 0, "c": 1, "d": 2.0, "e": 0.5})
        _assert_breakdown_band(breakdown, "clamp_breakdown")
        assert breakdown["e"] == 0.5


class TestHelpers:
    def test_evaluate_reply_invalid_replies_return_none(self) -> None:
        for text in ("", "hello world", " ".join(["word"] * 50) + " keyword"):
            score, _ = _evaluate_reply(text, ["keyword"])
            assert score is None

    def test_evaluate_reply_rewards_good_replies(self) -> None:
        score, _ = _evaluate_reply(
            "demo enterprise team upgrade", ["demo", "enterprise", "team", "upgrade"]
        )
        assert score is not None
        _assert_fraction_band(score, "reply_score")

    def test_politeness_bonus_stays_in_score_band(self) -> None:
        for text in ("some text", "please help"):
            _assert_fraction_band(_check_politeness(text), f"_check_politeness({text!r})")

    @pytest.mark.parametrize("steps", range(0, 20))
    def test_efficiency_bonus_stays_in_score_band(self, steps: int) -> None:
        _assert_fraction_band(_calculate_efficiency_bonus(steps), f"efficiency({steps})")


class TestRewardModel:
    @pytest.mark.parametrize("raw_score", [-1.0, 0, 0.001, 0.5, 0.999, 1, 2.0])
    def test_reward_score_is_binary(self, raw_score: float) -> None:
        reward = Reward(score=raw_score)
        _assert_binary_score(reward.score, f"Reward(score={raw_score})")

    def test_reward_breakdown_is_clamped_into_repo_band(self) -> None:
        reward = Reward(score=0.5, breakdown={"a": -0.5, "b": 0, "c": 1, "d": 1.5})
        _assert_breakdown_band(reward.breakdown, "Reward")


class TestTask1Grading:
    task_data = TASKS["task_1"]

    @pytest.mark.parametrize(
        "actions",
        [
            [],
            [CLASSIFY_SUPPORT],
            [CLASSIFY_WRONG],
            [CLASSIFY_SUPPORT, CLASSIFY_SUPPORT],
            [CLASSIFY_WRONG, CLASSIFY_SUPPORT],
            [NOOP] * 8,
            [CLASSIFY_WRONG, CLASSIFY_WRONG, CLASSIFY_WRONG],
        ],
    )
    def test_score_bounds(self, actions) -> None:
        reward, _ = grade_task_1(actions, self.task_data)
        _assert_binary_score(reward.score, "task_1")
        _assert_breakdown_band(reward.breakdown, "task_1")


class TestTask2Grading:
    task_data = TASKS["task_2"]

    @pytest.mark.parametrize(
        "actions",
        [
            [],
            [CLASSIFY_SALES],
            [CLASSIFY_SALES, ROUTE_SALES],
            [CLASSIFY_SALES, ROUTE_SALES, DRAFT_GOOD_T2],
            [CLASSIFY_WRONG, ROUTE_WRONG, DRAFT_BAD],
            [CLASSIFY_WRONG, ROUTE_WRONG, DRAFT_EMPTY],
            [CLASSIFY_SALES, ROUTE_SALES, DRAFT_GOOD_T2, NOOP, NOOP],
            [NOOP] * 8,
            [CLASSIFY_SALES, CLASSIFY_SALES, ROUTE_SALES, ROUTE_SALES, DRAFT_GOOD_T2],
        ],
    )
    def test_score_bounds(self, actions) -> None:
        reward, _ = grade_task_2(actions, self.task_data)
        _assert_binary_score(reward.score, "task_2")
        _assert_breakdown_band(reward.breakdown, "task_2")


class TestTask3Grading:
    task_data = TASKS["task_3"]

    @pytest.mark.parametrize(
        "actions",
        [
            [],
            [CLASSIFY_SUPPORT],
            [CLASSIFY_SUPPORT, ROUTE_ENGINEERING],
            [CLASSIFY_SUPPORT, ROUTE_ENGINEERING, ESCALATE],
            [CLASSIFY_SUPPORT, ROUTE_ENGINEERING, ESCALATE, DRAFT_GOOD_T3],
            [CLASSIFY_WRONG, ROUTE_WRONG, DRAFT_BAD],
            [CLASSIFY_WRONG, ROUTE_WRONG, DRAFT_EMPTY],
            [NOOP] * 8,
            [CLASSIFY_SUPPORT, ROUTE_ENGINEERING, DRAFT_GOOD_T3],
            [CLASSIFY_WRONG, CLASSIFY_WRONG, ROUTE_WRONG, ROUTE_WRONG, ESCALATE, ESCALATE],
        ],
    )
    def test_score_bounds(self, actions) -> None:
        reward, _ = grade_task_3(actions, self.task_data)
        _assert_binary_score(reward.score, "task_3")
        _assert_breakdown_band(reward.breakdown, "task_3")


class TestCalculateReward:
    def test_invalid_task_id_stays_inside_score_band(self) -> None:
        reward, _ = calculate_reward("task_999", [], {})
        _assert_binary_score(reward.score, "invalid_task_id")
        _assert_breakdown_band(reward.breakdown, "invalid_task_id")


class TestStress:
    all_actions = [
        CLASSIFY_SUPPORT,
        CLASSIFY_SALES,
        CLASSIFY_WRONG,
        ROUTE_SALES,
        ROUTE_ENGINEERING,
        ROUTE_WRONG,
        ESCALATE,
        DRAFT_GOOD_T2,
        DRAFT_GOOD_T3,
        DRAFT_BAD,
        DRAFT_EMPTY,
        NOOP,
        SPAM,
    ]

    @pytest.mark.parametrize("task_id", ["task_1", "task_2", "task_3"])
    def test_single_actions(self, task_id: str) -> None:
        task_data = TASKS[task_id]
        for action in self.all_actions:
            reward, _ = calculate_reward(task_id, [action], task_data)
            _assert_binary_score(reward.score, f"{task_id} single {action.action_type}")
            _assert_breakdown_band(reward.breakdown, f"{task_id} single {action.action_type}")

    @pytest.mark.parametrize("task_id", ["task_1", "task_2", "task_3"])
    def test_long_noop_sequence(self, task_id: str) -> None:
        task_data = TASKS[task_id]
        reward, _ = calculate_reward(task_id, [NOOP] * 20, task_data)
        _assert_binary_score(reward.score, f"{task_id} noop")
        _assert_breakdown_band(reward.breakdown, f"{task_id} noop")

    @pytest.mark.parametrize("task_id", ["task_1", "task_2", "task_3"])
    def test_all_wrong_sequence(self, task_id: str) -> None:
        task_data = TASKS[task_id]
        reward, _ = calculate_reward(
            task_id, [CLASSIFY_WRONG, ROUTE_WRONG, DRAFT_BAD, DRAFT_EMPTY] * 3, task_data
        )
        _assert_binary_score(reward.score, f"{task_id} all_wrong")
        _assert_breakdown_band(reward.breakdown, f"{task_id} all_wrong")
