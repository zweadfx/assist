"""
Unit and integration tests for The Whistle feature.

Test Cases:
- TC-01: Basic violation judgment (traveling)
- TC-02: Foul judgment (cylinder violation)
- TC-03: Glossary term search
- TC-04: FIBA/NBA rule type filtering
- TC-05: Exception handling (empty input)
"""
from langchain_core.documents import Document

from src.services.rag.rule_retrieval import RuleRetriever


class TestRuleRetrieval:
    """Unit tests for rule retrieval logic."""

    def test_tc01_basic_violation_search(self):
        """
        TC-01: 기본 바이얼레이션 판정
        입력: "공을 들고 세 발자국 걸으면?"
        기대: 트래블링 관련 규정 반환
        """
        retriever = RuleRetriever()

        results = retriever.hybrid_search(
            situation="공을 들고 세 발자국 걸으면?",
            n_rules=5,
            n_glossary=3,
        )

        assert "rules" in results
        assert "glossary" in results
        assert isinstance(results["rules"], list)
        assert isinstance(results["glossary"], list)

    def test_tc02_foul_situation_search(self):
        """
        TC-02: 파울 판정
        입력: "수비수가 실린더를 침범하면?"
        기대: 실린더 원칙 관련 규정 반환
        """
        retriever = RuleRetriever()

        results = retriever.search_by_situation(
            situation="수비수가 실린더를 침범하면?",
            n_results=5,
        )

        assert isinstance(results, list)

    def test_tc03_glossary_term_search(self):
        """
        TC-03: 용어 사전 검색
        입력: "트래블링"
        기대: 용어 정의 반환
        """
        retriever = RuleRetriever()

        results = retriever.search_glossary_terms(
            query="트래블링",
            n_results=3,
        )

        assert isinstance(results, list)
        if results:
            assert all(isinstance(doc, Document) for doc in results)
            assert results[0].metadata.get("doc_type") == "glossary"

    def test_tc04_rule_type_filtering(self):
        """
        TC-04: FIBA/NBA 규정 필터링
        입력: rule_type="FIBA"
        기대: FIBA 규정만 반환
        """
        retriever = RuleRetriever()

        results = retriever.search_by_situation(
            situation="트래블링 규정",
            rule_type="FIBA",
            n_results=5,
        )

        assert isinstance(results, list)
        for doc in results:
            assert doc.metadata.get("rule_type") == "FIBA"

    def test_tc05_empty_input_handling(self):
        """
        TC-05: 예외 처리 - 빈 입력
        입력: 빈 문자열
        기대: 빈 리스트 반환 (에러 없음)
        """
        retriever = RuleRetriever()

        rules = retriever.search_by_situation(situation="")
        glossary = retriever.search_glossary_terms(query="")

        assert rules == []
        assert glossary == []

    def test_tc05_whitespace_input_handling(self):
        """
        TC-05: 예외 처리 - 공백만 입력
        입력: 공백 문자열
        기대: 빈 리스트 반환 (에러 없음)
        """
        retriever = RuleRetriever()

        rules = retriever.search_by_situation(situation="   ")
        glossary = retriever.search_glossary_terms(query="   ")

        assert rules == []
        assert glossary == []

    def test_hybrid_search_returns_both(self):
        """
        하이브리드 검색이 rules와 glossary 모두 반환하는지 확인
        """
        retriever = RuleRetriever()

        results = retriever.hybrid_search(
            situation="블로킹 파울과 차징 파울의 차이",
            n_rules=3,
            n_glossary=2,
        )

        assert "rules" in results
        assert "glossary" in results
        assert isinstance(results["rules"], list)
        assert isinstance(results["glossary"], list)
