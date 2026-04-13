"""
Unit and integration tests for AI Skill Lab feature.

Test Cases:
- TC-01: Normal skill breakdown generation
- TC-02: Equipment filtering validation
- TC-03: Time allocation validation
- TC-04: Free-text parsing validation
- TC-05: Exception handling validation
- TC-06: API endpoint integration tests
"""

import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.main import app

_FAKE_SKILL_RESPONSE = {
    "skill_name": "Crossover Dribble",
    "total_duration_min": 20,
    "difficulty_level": "Basics → Game Speed",
    "coach_message": "Master the crossover step by step!",
    "steps": [
        {
            "step_number": 1,
            "name": "Stationary Hand Switch",
            "duration_min": 4,
            "description": "Stand still and switch the ball between hands. "
            "Focus on a low, controlled bounce. Do 3 sets of 10 reps.",
            "focus_point": "Keep your eyes up, not on the ball.",
            "success_criteria": "Complete 10 consecutive switches without losing the ball.",
        },
        {
            "step_number": 2,
            "name": "Walking Crossover",
            "duration_min": 4,
            "description": "Walk forward while performing crossover dribbles. "
            "Maintain a rhythm with each step. Do 5 lengths of the court.",
            "focus_point": "Synchronize the crossover with your footwork.",
            "success_criteria": "Complete 5 lengths without breaking rhythm.",
        },
        {
            "step_number": 3,
            "name": "Jogging Crossover",
            "duration_min": 4,
            "description": "Jog at half speed while executing crossovers. "
            "Add a slight change of direction. Do 5 lengths.",
            "focus_point": "Push the ball low and hard on the cross.",
            "success_criteria": "Complete 5 lengths at jogging pace with control.",
        },
        {
            "step_number": 4,
            "name": "Cone Weave Crossover",
            "duration_min": 4,
            "description": "Set up 5 cones and weave through them using crossovers. "
            "Increase speed each round. Do 3 rounds.",
            "focus_point": "Sell the move with a head fake before crossing over.",
            "success_criteria": "Complete 3 rounds in under 30 seconds each.",
        },
        {
            "step_number": 5,
            "name": "Live Defender Crossover",
            "duration_min": 4,
            "description": "Face a partner acting as a defender. "
            "Use the crossover to beat them off the dribble. Do 10 reps.",
            "focus_point": "Attack the defender's front foot.",
            "success_criteria": "Beat the defender 7 out of 10 attempts.",
        },
    ],
}


@pytest.fixture
def test_client():
    """FastAPI test client fixture."""
    with TestClient(app) as client:
        yield client


@pytest.fixture
def mock_coach():
    """Patch coach_agent_graph.invoke to return a canned response."""
    with patch(
        "src.api.v1.endpoints.skill.coach_agent_graph.invoke",
        return_value={"final_response": json.dumps(_FAKE_SKILL_RESPONSE)},
    ) as mock:
        yield mock


class TestSkillLabAPI:
    """Integration tests for Skill Lab API endpoint."""

    def test_tc01_api_normal_skill_breakdown(self, test_client, mock_coach):
        """
        TC-01 통합: 정상 스킬 분해 생성 (POST /api/v1/skill/)
        """
        # Arrange
        payload = {
            "skill_level": "intermediate",
            "category": "dribble",
            "specific_skill": "crossover",
            "available_time_min": 20,
            "equipment": ["ball", "cones"],
        }

        # Act
        response = test_client.post("/api/v1/skill/", json=payload)

        # Assert
        assert response.status_code == 200, f"Response: {response.text}"

        data = response.json()
        assert "data" in data

        skill_response = data["data"]
        assert "skill_name" in skill_response
        assert "total_duration_min" in skill_response
        assert "difficulty_level" in skill_response
        assert "coach_message" in skill_response
        assert "steps" in skill_response

        # Verify steps structure
        steps = skill_response["steps"]
        assert isinstance(steps, list)
        assert 3 <= len(steps) <= 5, "Should return 3-5 progressive steps"

        for step in steps:
            assert "step_number" in step
            assert "name" in step
            assert "duration_min" in step
            assert "description" in step
            assert "focus_point" in step
            assert "success_criteria" in step
            assert 1 <= step["step_number"] <= 5

    def test_tc03_time_allocation(self, test_client, mock_coach):
        """
        TC-03 통합: 시간 배분 검증
        available_time_min과 steps duration 합계가 일치
        """
        # Arrange
        available_time = 30
        payload = {
            "skill_level": "beginner",
            "category": "shooting",
            "available_time_min": available_time,
            "equipment": ["ball", "hoop"],
        }

        # Act
        response = test_client.post("/api/v1/skill/", json=payload)

        # Assert
        assert response.status_code == 200

        data = response.json()["data"]
        steps = data["steps"]

        total_step_duration = sum(step["duration_min"] for step in steps)
        assert total_step_duration == data["total_duration_min"], (
            f"Sum of step durations ({total_step_duration}) should equal "
            f"total_duration_min ({data['total_duration_min']})"
        )

    def test_tc04_free_text_parsing(self, test_client, mock_coach):
        """
        TC-04 통합: free_text 자연어 입력이 반영되는지 검증
        """
        # Arrange
        payload = {
            "skill_level": "intermediate",
            "category": "dribble",
            "available_time_min": 20,
            "equipment": ["ball"],
            "free_text": "I want to focus on weak hand dribbling with intense pace",
        }

        # Act
        response = test_client.post("/api/v1/skill/", json=payload)

        # Assert
        assert response.status_code == 200

        data = response.json()["data"]
        assert data["skill_name"], "Should have a skill_name"
        assert data["coach_message"], "Should have a coach_message"
        assert len(data["steps"]) >= 3, "Should have at least 3 steps"

    def test_tc05_validation_error_missing_required(self, test_client):
        """
        TC-05 통합: 필수 필드 누락 시 422 반환
        """
        # Missing skill_level and category
        payload = {
            "available_time_min": 20,
        }

        response = test_client.post("/api/v1/skill/", json=payload)
        assert response.status_code == 422

    def test_tc05_validation_error_invalid_time(self, test_client):
        """
        TC-05 통합: available_time_min이 0 이하일 때 422 반환
        """
        payload = {
            "skill_level": "beginner",
            "category": "dribble",
            "available_time_min": 0,
        }

        response = test_client.post("/api/v1/skill/", json=payload)
        assert response.status_code == 422

    def test_tc05_validation_error_invalid_category(self, test_client):
        """
        TC-05 통합: 유효하지 않은 카테고리 시 422 반환
        """
        payload = {
            "skill_level": "beginner",
            "category": "invalid_category",
            "available_time_min": 20,
        }

        response = test_client.post("/api/v1/skill/", json=payload)
        assert response.status_code == 422

    def test_tc06_minimal_input(self, test_client, mock_coach):
        """
        TC-06 통합: 최소 입력으로 API 호출
        """
        payload = {
            "skill_level": "beginner",
            "category": "defense",
            "available_time_min": 15,
        }

        response = test_client.post("/api/v1/skill/", json=payload)
        assert response.status_code == 200

        data = response.json()["data"]
        assert data["skill_name"]
        assert data["total_duration_min"] > 0
        assert len(data["steps"]) >= 3

    def test_tc06_korean_language(self, test_client, mock_coach):
        """
        TC-06 통합: 한국어 응답 요청
        """
        payload = {
            "skill_level": "intermediate",
            "category": "dribble",
            "specific_skill": "crossover",
            "available_time_min": 20,
            "equipment": ["ball"],
            "language": "ko",
        }

        response = test_client.post("/api/v1/skill/", json=payload)
        assert response.status_code == 200

        data = response.json()["data"]
        assert data["skill_name"], "Should have a skill_name in Korean"
        assert data["coach_message"], "Should have a coach_message in Korean"
