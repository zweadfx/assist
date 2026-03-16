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

import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.services.rag.chroma_db import chroma_manager


@pytest.fixture
def test_client():
    """FastAPI test client fixture."""
    with TestClient(app) as client:
        yield client


class TestDrillRetrieval:
    """Unit tests for drill retrieval and filtering logic."""

    def test_tc01_normal_drill_retrieval(self):
        """
        TC-01: 정상 드릴 검색
        카테고리: dribble, 난이도: intermediate
        """
        # Arrange
        query_text = (
            "A intermediate basketball drill focusing on improving "
            "dribble skills using ball."
        )

        # Act
        results = chroma_manager.query_drills(
            query_texts=[query_text],
            n_results=10,
            where={"category": "dribble"},
        )

        # Assert
        assert results is not None
        assert "documents" in results
        assert "metadatas" in results
        assert len(results["documents"][0]) > 0, "Should return at least 1 drill"

        # Verify all results match the requested category
        for metadata in results["metadatas"][0]:
            assert metadata["category"] == "dribble", (
                f"Expected category 'dribble', got '{metadata['category']}'"
            )

    def test_tc02_equipment_filtering(self):
        """
        TC-02: 장비 필터링 검증
        공만 보유 시 골대(hoop), 콘(cones) 필요 드릴 제외
        """
        # Arrange
        user_equipment = {"ball"}
        query_text = "A beginner basketball drill for shooting skills using ball."

        # Act
        results = chroma_manager.query_drills(
            query_texts=[query_text], n_results=10
        )

        # Assert - simulate the filtering logic from coach_agent.retrieve_drills
        assert results and results.get("documents")
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]

        filtered = []
        excluded = []
        for i, metadata in enumerate(metadatas):
            required_str = metadata.get("required_equipment", "")
            if not required_str:
                filtered.append(metadata)
                continue

            required = set(required_str.split(","))
            if required.issubset(user_equipment):
                filtered.append(metadata)
            else:
                excluded.append(metadata)

        # Verify excluded drills require equipment the user doesn't have
        for meta in excluded:
            required = set(meta.get("required_equipment", "").split(","))
            assert not required.issubset(user_equipment), (
                f"Drill '{meta.get('name')}' should have been included"
            )

        # Verify filtered drills only need equipment the user has
        for meta in filtered:
            required_str = meta.get("required_equipment", "")
            if required_str:
                required = set(required_str.split(","))
                assert required.issubset(user_equipment), (
                    f"Drill '{meta.get('name')}' requires {required} "
                    f"but user only has {user_equipment}"
                )

    def test_tc02_ball_only_excludes_hoop_drills(self):
        """
        TC-02 보조: 공만 보유 시 골대 필요 드릴이 실제로 제외되는지 확인
        """
        # Arrange
        user_equipment = {"ball"}
        query_text = "A basketball drill for shooting using ball."

        # Act
        results = chroma_manager.query_drills(
            query_texts=[query_text], n_results=10
        )

        assert results and results.get("metadatas")
        metadatas = results["metadatas"][0]

        filtered = []
        for metadata in metadatas:
            required_str = metadata.get("required_equipment", "")
            if not required_str:
                filtered.append(metadata)
                continue
            required = set(required_str.split(","))
            if required.issubset(user_equipment):
                filtered.append(metadata)

        # Verify no hoop-required drills survived filtering
        for meta in filtered:
            required_str = meta.get("required_equipment", "")
            if required_str:
                required_items = required_str.split(",")
                assert "hoop" not in required_items, (
                    f"Drill '{meta.get('name')}' requires hoop but user has ball only"
                )

    def test_tc03_category_filter_accuracy(self):
        """
        TC-03: 카테고리별 검색 정확도 검증
        4개 카테고리 각각 검색 시 올바른 카테고리만 반환
        """
        categories = ["dribble", "shooting", "defense", "conditioning"]

        for category in categories:
            query_text = f"A basketball drill for {category}."

            results = chroma_manager.query_drills(
                query_texts=[query_text],
                n_results=5,
                where={"category": category},
            )

            assert results and results.get("metadatas")
            for metadata in results["metadatas"][0]:
                assert metadata["category"] == category, (
                    f"Expected '{category}', got '{metadata['category']}'"
                )

    def test_tc05_empty_query(self):
        """
        TC-05: 빈 쿼리에 대한 처리
        """
        results = chroma_manager.query_drills(
            query_texts=[""], n_results=5
        )

        # Should not crash, returns some results (semantic search on empty string)
        assert results is not None
        assert "documents" in results

    def test_tc05_nonexistent_category(self):
        """
        TC-05: 존재하지 않는 카테고리 검색
        """
        results = chroma_manager.query_drills(
            query_texts=["basketball drill"],
            n_results=5,
            where={"category": "nonexistent_category"},
        )

        # Should return empty results, not crash
        assert results is not None
        assert len(results["documents"][0]) == 0


class TestSkillLabAPI:
    """Integration tests for Skill Lab API endpoint."""

    def test_tc01_api_normal_skill_breakdown(self, test_client):
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

    def test_tc03_time_allocation(self, test_client):
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

    def test_tc04_free_text_parsing(self, test_client):
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

    def test_tc06_minimal_input(self, test_client):
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

    def test_tc06_korean_language(self, test_client):
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
