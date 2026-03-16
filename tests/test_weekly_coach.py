"""
Unit and integration tests for Weekly Coach Agent feature.

Test Cases:
- TC-01: Normal weekly routine generation
- TC-02: Focus areas distribution validation
- TC-03: @model_validator validation (duration sum, phase presence, day consistency)
- TC-04: API endpoint integration tests
"""

import pytest
from pydantic import ValidationError
from fastapi.testclient import TestClient

from src.main import app
from src.models.weekly_schema import DailyPlan, WeeklyDrill, WeeklyRoutineResponse


@pytest.fixture
def test_client():
    """FastAPI test client fixture."""
    with TestClient(app) as client:
        yield client


class TestWeeklySchemaValidation:
    """Unit tests for @model_validator logic in weekly schemas."""

    def _make_drill(self, phase, duration_min=5, **kwargs):
        """Helper to create a WeeklyDrill instance."""
        defaults = {
            "phase": phase,
            "drill_id": f"test-{phase}-1",
            "name": f"Test {phase} drill",
            "duration_min": duration_min,
            "description": "Test description",
            "coaching_tip": "Test tip",
            "is_custom": False,
        }
        defaults.update(kwargs)
        return WeeklyDrill(**defaults)

    def test_tc03_daily_plan_valid(self):
        """
        TC-03: DailyPlan 정상 검증
        drill duration 합계 == total_duration_min, 3 phase 모두 존재
        """
        drills = [
            self._make_drill("warmup", duration_min=5),
            self._make_drill("main", duration_min=15),
            self._make_drill("cooldown", duration_min=5),
        ]

        plan = DailyPlan(
            day_number=1,
            day_label="Day 1 - Test",
            focus_areas=["shooting"],
            total_duration_min=25,
            drills=drills,
        )

        assert plan.total_duration_min == 25
        assert len(plan.drills) == 3

    def test_tc03_daily_plan_duration_mismatch(self):
        """
        TC-03: drill duration 합계 != total_duration_min 시 ValidationError
        """
        drills = [
            self._make_drill("warmup", duration_min=5),
            self._make_drill("main", duration_min=15),
            self._make_drill("cooldown", duration_min=5),
        ]

        with pytest.raises(ValidationError, match="Sum of drill durations"):
            DailyPlan(
                day_number=1,
                day_label="Day 1",
                focus_areas=["shooting"],
                total_duration_min=30,  # actual sum is 25
                drills=drills,
            )

    def test_tc03_daily_plan_missing_phase(self):
        """
        TC-03: 필수 phase(warmup/main/cooldown) 중 하나 누락 시 ValidationError
        """
        drills = [
            self._make_drill("warmup", duration_min=10),
            self._make_drill("main", duration_min=10),
            # cooldown missing
        ]

        with pytest.raises(ValidationError, match="Missing required phases"):
            DailyPlan(
                day_number=1,
                day_label="Day 1",
                focus_areas=["dribble"],
                total_duration_min=20,
                drills=drills,
            )

    def test_tc03_weekly_response_valid(self):
        """
        TC-03: WeeklyRoutineResponse 정상 검증
        total_days == len(days), day_number 중복 없음
        """
        drills = [
            self._make_drill("warmup", duration_min=5),
            self._make_drill("main", duration_min=15),
            self._make_drill("cooldown", duration_min=5),
        ]

        days = [
            DailyPlan(
                day_number=i,
                day_label=f"Day {i}",
                focus_areas=["shooting"],
                total_duration_min=25,
                drills=drills,
            )
            for i in range(1, 4)
        ]

        response = WeeklyRoutineResponse(
            weekly_title="Test Week",
            coach_overview="Test overview",
            total_days=3,
            days=days,
        )

        assert response.total_days == 3
        assert len(response.days) == 3

    def test_tc03_weekly_response_total_days_mismatch(self):
        """
        TC-03: total_days != len(days) 시 ValidationError
        """
        drills = [
            self._make_drill("warmup", duration_min=5),
            self._make_drill("main", duration_min=15),
            self._make_drill("cooldown", duration_min=5),
        ]

        days = [
            DailyPlan(
                day_number=1,
                day_label="Day 1",
                focus_areas=["shooting"],
                total_duration_min=25,
                drills=drills,
            )
        ]

        with pytest.raises(ValidationError, match="total_days"):
            WeeklyRoutineResponse(
                weekly_title="Test Week",
                coach_overview="Test overview",
                total_days=3,  # but only 1 day provided
                days=days,
            )

    def test_tc03_weekly_response_duplicate_day_numbers(self):
        """
        TC-03: day_number 중복 시 ValidationError
        """
        drills = [
            self._make_drill("warmup", duration_min=5),
            self._make_drill("main", duration_min=15),
            self._make_drill("cooldown", duration_min=5),
        ]

        days = [
            DailyPlan(
                day_number=1,
                day_label=f"Day {label}",
                focus_areas=["shooting"],
                total_duration_min=25,
                drills=drills,
            )
            for label in ["A", "B"]  # both day_number=1
        ]

        with pytest.raises(ValidationError, match="Duplicate day_number"):
            WeeklyRoutineResponse(
                weekly_title="Test Week",
                coach_overview="Test overview",
                total_days=2,
                days=days,
            )

    def test_tc03_weekly_response_day_number_range_mismatch(self):
        """
        TC-03: day_number 값이 expected range [1..total_days]와 불일치 시 ValidationError
        """
        drills = [
            self._make_drill("warmup", duration_min=5),
            self._make_drill("main", duration_min=15),
            self._make_drill("cooldown", duration_min=5),
        ]

        days = [
            DailyPlan(
                day_number=2,
                day_label="Day 2",
                focus_areas=["shooting"],
                total_duration_min=25,
                drills=drills,
            ),
            DailyPlan(
                day_number=3,
                day_label="Day 3",
                focus_areas=["dribble"],
                total_duration_min=25,
                drills=drills,
            ),
        ]

        with pytest.raises(ValidationError, match="day_number values"):
            WeeklyRoutineResponse(
                weekly_title="Test Week",
                coach_overview="Test overview",
                total_days=2,  # expects [1, 2] but got [2, 3]
                days=days,
            )


class TestWeeklyCoachAPI:
    """Integration tests for Weekly Coach API endpoint."""

    def test_tc01_normal_weekly_routine(self, test_client):
        """
        TC-01 통합: 정상 주간 루틴 생성 (POST /api/v1/skill/weekly)
        """
        payload = {
            "skill_level": "intermediate",
            "training_days": 3,
            "focus_areas": ["shooting", "dribble"],
            "available_time_per_day_min": 30,
            "equipment": ["ball", "hoop", "cones"],
        }

        response = test_client.post("/api/v1/skill/weekly", json=payload)
        assert response.status_code == 200, f"Response: {response.text}"

        data = response.json()
        assert "data" in data

        weekly = data["data"]
        assert "weekly_title" in weekly
        assert "coach_overview" in weekly
        assert "total_days" in weekly
        assert "days" in weekly

        assert weekly["total_days"] == 3
        assert len(weekly["days"]) == 3

        for day in weekly["days"]:
            assert "day_number" in day
            assert "day_label" in day
            assert "focus_areas" in day
            assert "total_duration_min" in day
            assert "drills" in day
            assert 1 <= day["day_number"] <= 3

            # Verify each day has all 3 phases
            phases = {drill["phase"] for drill in day["drills"]}
            assert "warmup" in phases, f"Day {day['day_number']} missing warmup"
            assert "main" in phases, f"Day {day['day_number']} missing main"
            assert "cooldown" in phases, f"Day {day['day_number']} missing cooldown"

            # Verify duration sum matches
            drill_sum = sum(drill["duration_min"] for drill in day["drills"])
            assert drill_sum == day["total_duration_min"], (
                f"Day {day['day_number']}: drill sum {drill_sum} != "
                f"total_duration_min {day['total_duration_min']}"
            )

    def test_tc02_focus_areas_distribution(self, test_client):
        """
        TC-02 통합: focus_areas가 training_days에 분배되는지 검증
        """
        payload = {
            "skill_level": "beginner",
            "training_days": 4,
            "focus_areas": ["shooting", "dribble", "defense", "conditioning"],
            "available_time_per_day_min": 30,
            "equipment": ["ball"],
        }

        response = test_client.post("/api/v1/skill/weekly", json=payload)
        assert response.status_code == 200

        weekly = response.json()["data"]
        assert weekly["total_days"] == 4

        # Collect all focus areas across all days
        all_focus = []
        for day in weekly["days"]:
            all_focus.extend(day["focus_areas"])

        # Each requested focus area should appear at least once
        for area in payload["focus_areas"]:
            assert area in all_focus, (
                f"Focus area '{area}' not found in any day's plan"
            )

    def test_tc04_validation_error_missing_focus_areas(self, test_client):
        """
        TC-04 통합: focus_areas 누락 시 422 반환
        """
        payload = {
            "skill_level": "beginner",
            "training_days": 3,
            "available_time_per_day_min": 30,
        }

        response = test_client.post("/api/v1/skill/weekly", json=payload)
        assert response.status_code == 422

    def test_tc04_validation_error_empty_focus_areas(self, test_client):
        """
        TC-04 통합: focus_areas 빈 리스트 시 422 반환
        """
        payload = {
            "skill_level": "beginner",
            "training_days": 3,
            "focus_areas": [],
            "available_time_per_day_min": 30,
        }

        response = test_client.post("/api/v1/skill/weekly", json=payload)
        assert response.status_code == 422

    def test_tc04_validation_error_invalid_training_days(self, test_client):
        """
        TC-04 통합: training_days 범위 초과 시 422 반환
        """
        payload = {
            "skill_level": "beginner",
            "training_days": 8,  # max is 7
            "focus_areas": ["shooting"],
            "available_time_per_day_min": 30,
        }

        response = test_client.post("/api/v1/skill/weekly", json=payload)
        assert response.status_code == 422

    def test_tc04_validation_error_time_too_low(self, test_client):
        """
        TC-04 통합: available_time_per_day_min이 최소값 미만 시 422 반환
        """
        payload = {
            "skill_level": "beginner",
            "training_days": 3,
            "focus_areas": ["shooting"],
            "available_time_per_day_min": 2,  # min is 3
        }

        response = test_client.post("/api/v1/skill/weekly", json=payload)
        assert response.status_code == 422

    def test_tc04_minimal_input(self, test_client):
        """
        TC-04 통합: 최소 입력으로 주간 루틴 생성
        """
        payload = {
            "skill_level": "beginner",
            "training_days": 2,
            "focus_areas": ["dribble"],
            "available_time_per_day_min": 20,
        }

        response = test_client.post("/api/v1/skill/weekly", json=payload)
        assert response.status_code == 200

        weekly = response.json()["data"]
        assert weekly["total_days"] == 2
        assert len(weekly["days"]) == 2
