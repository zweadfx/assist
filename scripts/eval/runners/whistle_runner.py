"""Whistle (Judge) evaluation runner.

Runs each whistle case through the judge agent graph,
then extracts cited articles for citation hit rate computation.
"""
# ruff: noqa: E501

import json
import logging
import re
from pathlib import Path

from langchain_core.messages import HumanMessage

from scripts.eval.metrics import citation_hit_rate, compute_llm_judge_metrics
from scripts.eval.runners.judge_llm_runner import evaluate_with_llm_judge
from src.core.constants import LLM_JUDGE_WHISTLE_PROMPT, LLM_JUDGE_WHISTLE_SYSTEM_PROMPT
from src.models.rule_schema import WhistleResponse
from src.services.agents.judge_agent import judge_agent_graph

logger = logging.getLogger(__name__)

DATASETS_DIR = Path(__file__).resolve().parent.parent / "datasets"


def _normalize_article(article: str) -> str:
    """Extract the core article number from various formats.

    'Art 25' → '25', '33.9' → '33', 'Article 31' → '31', 'Rule 12B' → '12B'
    """
    matches = re.findall(r"\d+[A-Za-z]*", article)
    return matches[0] if matches else article


def _run_single_whistle(case: dict) -> dict:
    """Run a single whistle case through the judge agent.

    Returns dict with predicted_articles, predicted_decision, raw_response, context.
    """
    user_input = case["input"]
    initial_state = {
        "messages": [HumanMessage(content=user_input["situation_description"])],
        "user_info": {
            "situation_description": user_input["situation_description"],
            "rule_type": user_input.get("rule_type", "FIBA"),
        },
    }

    final_state = judge_agent_graph.invoke(initial_state)
    raw = final_state.get("final_response", "")

    # Extract context text
    context_docs = final_state.get("context", [])
    context_text = (
        "\n".join([doc.page_content for doc in context_docs]) if context_docs else ""
    )

    response = WhistleResponse.model_validate_json(raw)

    predicted_articles = [
        _normalize_article(ref.article) for ref in response.rule_references
    ]
    return {
        "predicted_articles": predicted_articles,
        "predicted_decision": response.decision,
        "raw_response": raw,
        "context": context_text,
    }


def run_whistle_evaluation() -> dict:
    """Run full whistle evaluation for all cases.

    Returns:
        Dict with 'citation_hit_rate', 'n_cases', 'llm_judge_metrics', and 'details' per case.
    """
    with open(DATASETS_DIR / "whistle_cases.json", encoding="utf-8") as f:
        cases = json.load(f)

    citation_results = []
    llm_judge_results = []
    details = []

    for case in cases:
        case_id = case["id"]
        expected_articles = case["expected_articles"]
        expected_decision = case["expected_decision"]
        logger.info("Running whistle case: %s", case_id)

        agent_failed = False
        try:
            result = _run_single_whistle(case)
            predicted_articles = result["predicted_articles"]
            predicted_decision = result["predicted_decision"]
            raw_response = result["raw_response"]
            context_text = result["context"]
        except Exception as e:
            logger.error("Whistle failed for %s: %s", case_id, e)
            predicted_articles = []
            predicted_decision = "error"
            raw_response = ""
            context_text = ""
            agent_failed = True

        # Run LLM Judge
        if agent_failed:
            llm_judge_score = {
                "accuracy_score": 0,
                "consistency_score": 0,
                "citation_score": 0,
                "reasoning": "Skipped: agent failed",
            }
        else:
            try:
                expected_answer_str = (
                    f"Decision: {expected_decision}, Rationale: {case.get('rationale', '')}"
                )
                llm_judge_score = evaluate_with_llm_judge(
                    question=case["input"]["situation_description"],
                    generated_answer=raw_response,
                    expected_answer=expected_answer_str,
                    context=context_text,
                    prompt_template=LLM_JUDGE_WHISTLE_PROMPT,
                    system_prompt=LLM_JUDGE_WHISTLE_SYSTEM_PROMPT,
                )
            except Exception as e:
                logger.error("LLM Judge failed for %s: %s", case_id, e)
                llm_judge_score = {
                    "accuracy_score": 0,
                    "consistency_score": 0,
                    "citation_score": 0,
                    "reasoning": "Error",
                }

        citation_results.append(
            {
                "predicted_articles": predicted_articles,
                "expected_articles": expected_articles,
            }
        )

        llm_judge_results.append(llm_judge_score)

        decision_match = predicted_decision == expected_decision
        details.append(
            {
                "id": case_id,
                "description": case["description"],
                "expected_decision": expected_decision,
                "predicted_decision": predicted_decision,
                "decision_match": decision_match,
                "expected_articles": expected_articles,
                "predicted_articles": predicted_articles,
                "llm_judge_score": llm_judge_score,
            }
        )

        logger.info(
            "  %s → decision: %s (expected: %s) | articles: %s (expected: %s) | Judge: %s",  # noqa: E501
            case_id,
            predicted_decision,
            expected_decision,
            predicted_articles,
            expected_articles,
            f"A:{llm_judge_score.get('accuracy_score')} C:{llm_judge_score.get('consistency_score')} Cit:{llm_judge_score.get('citation_score')}"  # noqa: E501,
        )

    return {
        "citation_hit_rate": citation_hit_rate(citation_results),
        "llm_judge_metrics": compute_llm_judge_metrics(llm_judge_results),
        "n_cases": len(cases),
        "details": details,
    }
