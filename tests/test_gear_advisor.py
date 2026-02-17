"""
Unit and integration tests for Gear Advisor feature.

Test Cases:
- TC-01: Normal recommendation generation
- TC-02: Sensory tag filtering validation
- TC-03: Player archetype matching validation
- TC-04: Complex condition search validation
- TC-05: Exception handling validation
"""

import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.services.rag.shoe_retrieval import ShoeRetriever


@pytest.fixture
def test_client():
    """FastAPI test client fixture."""
    return TestClient(app)


@pytest.fixture
def shoe_retriever_instance():
    """ShoeRetriever instance fixture."""
    return ShoeRetriever()


class TestShoeRetrieval:
    """Unit tests for shoe retrieval logic."""

    def test_tc01_normal_recommendation(self, shoe_retriever_instance):
        """
        TC-01: 정상 추천 생성
        감각: 쫀득한 접지, 선수: Stephen Curry
        """
        # Arrange
        sensory_keywords = ["쫀득한 접지", "가벼운 무게"]
        player = "Stephen Curry"

        # Act
        results = shoe_retriever_instance.cross_analysis_search(
            sensory_keywords=sensory_keywords,
            player_archetype=player,
            n_shoes=5,
        )

        # Assert
        assert "shoes" in results
        assert "players" in results
        assert len(results["shoes"]) > 0, "Should return at least 1 shoe"
        assert len(results["shoes"]) <= 5, "Should not exceed 5 shoes"

        # Verify player found
        if results["players"]:
            player_names = [p.metadata.get("name", "") for p in results["players"]]
            assert any("curry" in name.lower() for name in player_names), (
                "Should find Stephen Curry archetype"
            )

        # Verify shoes have required metadata
        for shoe in results["shoes"]:
            assert "brand" in shoe.metadata
            assert "model_name" in shoe.metadata
            assert "price_krw" in shoe.metadata
            assert shoe.page_content, "Document content should not be empty"

    def test_tc02_sensory_tag_filtering(self, shoe_retriever_instance):
        """
        TC-02: 감각 태그 필터링 검증
        특정 감각 태그로 검색 시 관련 신발만 반환되는지 확인
        """
        # Arrange
        sensory_keywords = ["초경량", "민첩한 무브"]

        # Act
        results = shoe_retriever_instance.search_by_sensory_preferences(
            sensory_keywords=sensory_keywords, n_results=10
        )

        # Assert
        assert isinstance(results, list)
        assert len(results) > 0, "Should find shoes matching sensory preferences"

        # Verify sensory tags are present in results
        for shoe in results:
            sensory_tags = shoe.metadata.get("sensory_tags", "")
            assert isinstance(sensory_tags, str), "Sensory tags should be string"

    def test_tc03_player_archetype_matching(self, shoe_retriever_instance):
        """
        TC-03: 선수 아키타입 매칭 검증
        특정 선수 검색 시 해당 선수의 플레이 스타일 정보 반환
        """
        # Arrange
        player_names = ["LeBron James", "Kobe Bryant", "Stephen Curry"]

        for player_name in player_names:
            # Act
            results = shoe_retriever_instance.search_by_player_archetype(
                player_name=player_name, n_results=3
            )

            # Assert
            assert isinstance(results, list)
            if results:  # If player found
                assert len(results) <= 3, "Should not exceed requested count"

                # Verify player metadata
                player = results[0]
                assert "name" in player.metadata
                assert "position" in player.metadata
                assert "play_style" in player.metadata

    def test_tc04_complex_condition_search(self, shoe_retriever_instance):
        """
        TC-04: 복합 조건 검색 검증
        감각 태그 + 선수 + 예산 + 포지션 복합 조건 검색
        """
        # Arrange
        sensory_keywords = ["쫀득한 접지", "반응성 쿠션"]
        player = "Damian Lillard"
        budget = 200000
        position = "guard"

        # Act
        results = shoe_retriever_instance.cross_analysis_search(
            sensory_keywords=sensory_keywords,
            player_archetype=player,
            budget_max_krw=budget,
            position=position,
            n_shoes=5,
        )

        # Assert
        assert "shoes" in results
        assert "players" in results

        # Verify budget constraint
        for shoe in results["shoes"]:
            price = int(shoe.metadata.get("price_krw", 0))
            assert price <= budget, f"Shoe price {price} exceeds budget {budget}"

        # Verify position matching (if applicable)
        # At least some shoes should match the guard position
        guard_shoes = []
        for shoe in results["shoes"]:
            tags = shoe.metadata.get("tags", "").split(",")
            # Clean up tags (strip whitespace)
            tags = [tag.strip() for tag in tags if tag.strip()]
            if "가드" in tags or "guard" in [t.lower() for t in tags]:
                guard_shoes.append(shoe)

        # Soft check: at least one shoe should be suitable for guards
        assert len(guard_shoes) > 0, (
            "Expected at least one guard-suitable shoe in results"
        )

    def test_tc05_exception_handling(self, shoe_retriever_instance):
        """
        TC-05: 예외 처리 검증
        잘못된 입력이나 빈 결과에 대한 적절한 처리
        """
        # Test Case 1: Empty sensory keywords
        results = shoe_retriever_instance.search_by_sensory_preferences(
            sensory_keywords=[], n_results=5
        )
        assert isinstance(results, list)  # Should return empty list, not crash

        # Test Case 2: Extremely low budget (filters out all shoes)
        results = shoe_retriever_instance.cross_analysis_search(
            sensory_keywords=["쫀득한 접지"],
            budget_max_krw=1,  # 1 KRW budget should filter out all shoes
            n_shoes=5,
        )
        assert "shoes" in results, "Results should contain 'shoes' key"
        assert results["shoes"] == [], (
            "Extremely low budget should result in empty shoe list"
        )

        # Test Case 3: Non-existent player
        results = shoe_retriever_instance.search_by_player_archetype(
            player_name="NonExistentPlayer12345", n_results=3
        )
        assert isinstance(results, list)  # Should return empty list

    def test_budget_filtering(self, shoe_retriever_instance):
        """
        추가 테스트: 예산 필터링 정확도
        """
        # Arrange
        budget = 150000
        sensory_keywords = ["쫀득한 접지"]

        # Act
        results = shoe_retriever_instance.search_by_sensory_preferences(
            sensory_keywords=sensory_keywords,
            budget_max_krw=budget,
            n_results=20,
        )

        # Assert
        for shoe in results:
            price = int(shoe.metadata.get("price_krw", 0))
            assert price <= budget, f"Price {price} exceeds budget {budget}"

    def test_signature_shoe_boosting(self, shoe_retriever_instance):
        """
        추가 테스트: 시그니처 슈즈 부스팅 로직
        """
        # Arrange
        player = "Stephen Curry"

        # Act
        results = shoe_retriever_instance.cross_analysis_search(
            sensory_keywords=["정교한 접지"],
            player_archetype=player,
            n_shoes=5,
        )

        # Assert
        if results["shoes"] and results["players"]:
            # Check if Curry signature shoes appear in results
            shoe_models = [
                shoe.metadata.get("model_name", "") for shoe in results["shoes"]
            ]
            # At least one Curry shoe should be highly ranked if available
            has_curry_shoe = any("curry" in model.lower() for model in shoe_models)

            # Verify signature shoe boosting behavior
            # Since we searched for "Stephen Curry", his signature shoes should appear
            assert has_curry_shoe, (
                f"Expected Curry signature shoes in results when searching for '{player}', "
                f"but found: {shoe_models}"
            )


class TestGearAdvisorAPI:
    """Integration tests for Gear Advisor API endpoint."""

    def test_api_endpoint_success(self, test_client):
        """
        통합 테스트: API 엔드포인트 E2E - 정상 케이스
        """
        # Arrange
        payload = {
            "sensory_preferences": ["쫀득한 접지", "가벼운 무게"],
            "player_archetype": "Stephen Curry",
            "position": "guard",
            "budget_max_krw": 250000,
        }

        # Act
        response = test_client.post("/api/v1/gear/recommend", json=payload)

        # Assert
        assert response.status_code == 200, f"Response: {response.text}"

        data = response.json()
        assert "data" in data, "Response should have 'data' field"

        gear_response = data["data"]
        assert "recommendation_title" in gear_response
        assert "user_profile_summary" in gear_response
        assert "ai_reasoning" in gear_response
        assert "shoes" in gear_response

        # Verify shoes structure
        shoes = gear_response["shoes"]
        assert isinstance(shoes, list)
        assert 1 <= len(shoes) <= 5, "Should return 1-5 shoes"

        for shoe in shoes:
            assert "shoe_id" in shoe
            assert "brand" in shoe
            assert "model_name" in shoe
            assert "price_krw" in shoe
            assert "sensory_tags" in shoe
            assert "match_score" in shoe
            assert "recommendation_reason" in shoe

            # Verify match_score range
            assert 0 <= shoe["match_score"] <= 100

    def test_api_endpoint_minimal_input(self, test_client):
        """
        통합 테스트: 최소 입력으로 API 호출
        """
        # Arrange
        payload = {"sensory_preferences": ["쫀득한 접지"]}

        # Act
        response = test_client.post("/api/v1/gear/recommend", json=payload)

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "data" in data

    def test_api_endpoint_validation_error(self, test_client):
        """
        통합 테스트: 유효성 검사 에러 (빈 sensory_preferences)
        """
        # Arrange
        payload = {"sensory_preferences": []}  # Empty list should fail validation

        # Act
        response = test_client.post("/api/v1/gear/recommend", json=payload)

        # Assert
        assert response.status_code == 422, "Should return validation error"

    def test_api_endpoint_with_all_parameters(self, test_client):
        """
        통합 테스트: 모든 파라미터 포함
        """
        # Arrange
        payload = {
            "sensory_preferences": ["쫀득한 접지", "반응성 쿠션", "가벼운 무게"],
            "player_archetype": "Damian Lillard",
            "position": "guard",
            "budget_max_krw": 200000,
        }

        # Act
        response = test_client.post("/api/v1/gear/recommend", json=payload)

        # Assert
        assert response.status_code == 200
        data = response.json()
        gear_response = data["data"]

        # Verify budget constraint in response
        for shoe in gear_response["shoes"]:
            assert shoe["price_krw"] <= 200000, "All shoes should be within budget"


class TestRAGSearchQuality:
    """Tests for RAG search quality validation."""

    def test_semantic_similarity_threshold(self, shoe_retriever_instance):
        """
        RAG 검색 품질 검증: 의미론적 유사도 70% 이상

        Note: This is a qualitative test. We verify that results are relevant
        by checking if sensory tags in results overlap with query keywords.
        """
        # Arrange
        sensory_keywords = ["쫀득한 접지", "민첩한 무브"]

        # Act
        results = shoe_retriever_instance.search_by_sensory_preferences(
            sensory_keywords=sensory_keywords, n_results=5
        )

        # Assert
        assert len(results) > 0, "Should return results"

        # Check semantic relevance (at least some overlap in tags)
        relevant_count = 0
        for shoe in results:
            sensory_tags = shoe.metadata.get("sensory_tags", "").lower()
            # Check if any query keyword appears in tags
            for keyword in sensory_keywords:
                if any(word in sensory_tags for word in keyword.split()):
                    relevant_count += 1
                    break

        # At least 60% of results should have some keyword overlap
        relevance_ratio = relevant_count / len(results)
        assert relevance_ratio >= 0.6, (
            f"Relevance ratio {relevance_ratio:.2f} below 60% threshold"
        )

    def test_player_search_accuracy(self, shoe_retriever_instance):
        """
        선수 검색 정확도: 정확한 선수명으로 검색 시 1순위 매칭
        """
        # Arrange
        exact_player_names = ["Stephen Curry", "LeBron James", "Kobe Bryant"]

        for player_name in exact_player_names:
            # Act
            results = shoe_retriever_instance.search_by_player_archetype(
                player_name=player_name, n_results=1
            )

            # Assert
            if results:  # If player exists in DB
                top_result = results[0]
                result_name = top_result.metadata.get("name", "").lower()
                query_name = player_name.lower()

                # Top result should contain the queried player name
                assert query_name in result_name or result_name in query_name, (
                    f"Expected {player_name}, got {result_name}"
                )
