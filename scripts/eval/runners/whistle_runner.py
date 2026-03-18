"""Whistle (Judge) evaluation runner.

Runs each whistle case through the judge agent graph,
then extracts cited articles for citation hit rate computation.
"""

import json
import logging
import re
from pathlib import Path

from langchain_core.messages import HumanMessage

from scripts.eval.metrics import citation_hit_rate
from src.models.rule_schema import WhistleResponse
from src.services.agents.judge_agent import judge_agent_graph

logger = logging.getLogger(__name__)

DATASETS_DIR = Path(__file__).resolve().parent.parent / "datasets"


def _normalize_article(article: str) -> str:
    """Extract the core article number from various formats.

    'Art 25' → '25', '33.9' → '33', 'Article 31' → '31'
    """
    numbers = re.findall(r"\d+", article)
    return numbers[0] if numbers else article


def _run_single_whistle(case: dict) -> dict:
    """Run a single whistle case through the judge agent.

    Returns dict with predicted_articles, predicted_decision.
    """
    user_input = case["input"]
    initial_state = {
        "messages": [
            HumanMessage(content=user_input["situation_description"])
        ],
        "user_info": {
            "situation_description": user_input["situation_description"],
            "rule_type": user_input.get("rule_type", "FIBA"),
        },
    }

    final_state = judge_agent_graph.invoke(initial_state)
    raw = final_state.get("final_response", "")
    response = WhistleResponse.model_validate_json(raw)

    predicted_articles = [
        _normalize_article(ref.article) for ref in response.rule_references
    ]
    return {
        "predicted_articles": predicted_articles,
        "predicted_decision": response.decision,
    }


def run_whistle_evaluation() -> dict:
    """Run full whistle evaluation for all cases.

    Returns:
        Dict with 'citation_hit_rate', 'n_cases', and 'details' per case.
    """
    with open(DATASETS_DIR / "whistle_cases.json", encoding="utf-8") as f:
        cases = json.load(f)

    citation_results = []
    details = []

    for case in cases:
        case_id = case["id"]
        expected_articles = case["expected_articles"]
        expected_decision = case["expected_decision"]
        logger.info("Running whistle case: %s", case_id)

        try:
            result = _run_single_whistle(case)
            predicted_articles = result["predicted_articles"]
            predicted_decision = result["predicted_decision"]
        except Exception as e:
            logger.error("Whistle failed for %s: %s", case_id, e)
            predicted_articles = []
            predicted_decision = "error"

        citation_results.append({
            "predicted_articles": predicted_articles,
            "expected_articles": expected_articles,
        })

        decision_match = predicted_decision == expected_decision
        details.append({
            "id": case_id,
            "description": case["description"],
            "expected_decision": expected_decision,
            "predicted_decision": predicted_decision,
            "decision_match": decision_match,
            "expected_articles": expected_articles,
            "predicted_articles": predicted_articles,
        })

        logger.info(
            "  %s → decision: %s (expected: %s) | articles: %s (expected: %s)",
            case_id,
            predicted_decision,
            expected_decision,
            predicted_articles,
            expected_articles,
        )

    return {
        "citation_hit_rate": citation_hit_rate(citation_results),
        "n_cases": len(cases),
        "details": details,
    }
