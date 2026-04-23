"""Gear Advisor evaluation runner.

Runs each gear case through both RAG and No-RAG baseline pipelines,
then collects predicted shoe IDs for metric computation.
"""
# ruff: noqa: E501

import json
import logging
from pathlib import Path

from langchain_core.messages import HumanMessage

from scripts.eval.baseline import run_no_rag_gear
from scripts.eval.metrics import compute_gear_metrics, compute_llm_judge_metrics
from scripts.eval.runners.judge_llm_runner import evaluate_with_llm_judge
from src.core.constants import LLM_JUDGE_GEAR_PROMPT, LLM_JUDGE_GEAR_SYSTEM_PROMPT
from src.models.gear_schema import GearAdvisorResponse
from src.services.agents.gear_agent import gear_agent_graph

logger = logging.getLogger(__name__)

DATASETS_DIR = Path(__file__).resolve().parent.parent / "datasets"
DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "raw"

# model_name → shoe_id lookup (RAG pipeline doesn't pass shoe_id in context)
_MODEL_TO_ID: dict[str, str] = {}


def _get_model_to_id_map() -> dict[str, str]:
    """Build and cache model_name → shoe_id mapping from shoes.json."""
    if not _MODEL_TO_ID:
        with open(DATA_DIR / "shoes.json", encoding="utf-8") as f:
            shoes = json.load(f)
        for s in shoes:
            _MODEL_TO_ID[s["model_name"].lower()] = s["id"]
    return _MODEL_TO_ID


def _resolve_shoe_id(shoe_id_or_name: str, model_name: str) -> str:
    """Resolve to canonical shoe_id. Falls back to model_name lookup."""
    if shoe_id_or_name.startswith("shoe_"):
        return shoe_id_or_name
    lookup = _get_model_to_id_map()
    resolved = lookup.get(model_name.lower())
    if resolved:
        return resolved
    return shoe_id_or_name


def _extract_shoe_ids(response: GearAdvisorResponse) -> list[str]:
    """Extract shoe_id list from GearAdvisorResponse, resolving names."""
    return [_resolve_shoe_id(shoe.shoe_id, shoe.model_name) for shoe in response.shoes]


def _run_single_rag(case: dict) -> dict:
    """Run a single case through the RAG pipeline."""
    user_input = case["input"]
    initial_state = {
        "messages": [
            HumanMessage(
                content=(
                    "Recommend shoes for: "
                    f"{', '.join(user_input.get('sensory_preferences', []))}"
                )
            )
        ],
        "user_info": {
            "sensory_preferences": user_input.get("sensory_preferences", []),
            "player_archetype": user_input.get("player_archetype"),
            "position": user_input.get("position"),
            "budget_max_krw": user_input.get("budget_max_krw"),
            "language": user_input.get("language", "en"),
        },
    }

    final_state = gear_agent_graph.invoke(initial_state)
    raw = final_state.get("final_response", "")

    context_docs = final_state.get("context", [])
    context_text = (
        "\n".join([doc.page_content for doc in context_docs]) if context_docs else ""
    )

    try:
        response = GearAdvisorResponse.model_validate_json(raw)
        predicted_ids = _extract_shoe_ids(response)
    except Exception as e:
        logger.error("Failed to parse RAG response: %s", e)
        predicted_ids = []

    return {
        "predicted_ids": predicted_ids,
        "raw_response": raw,
        "context": context_text,
    }


def _run_single_baseline(case: dict) -> list[str]:
    """Run a single case through the No-RAG baseline."""
    response = run_no_rag_gear(case["input"])
    return _extract_shoe_ids(response)


def run_gear_evaluation() -> dict:
    """Run full gear evaluation: RAG vs baseline for all cases.

    Returns:
        Dict with 'rag_results', 'baseline_results', 'rag_metrics',
        'baseline_metrics', 'llm_judge_metrics', and 'details' per case.
    """
    with open(DATASETS_DIR / "gear_cases.json", encoding="utf-8") as f:
        cases = json.load(f)

    rag_results = []
    baseline_results = []
    llm_judge_results = []
    details = []

    for case in cases:
        case_id = case["id"]
        expected_ids = case["expected_shoe_ids"]
        logger.info("Running gear case: %s", case_id)

        # RAG
        rag_failed = False
        try:
            rag_output = _run_single_rag(case)
            rag_predicted = rag_output["predicted_ids"]
            raw_response = rag_output["raw_response"]
            context_text = rag_output["context"]
        except Exception as e:
            logger.error("RAG failed for %s: %s", case_id, e)
            rag_predicted = []
            raw_response = ""
            context_text = ""
            rag_failed = True

        # Baseline
        try:
            baseline_predicted = _run_single_baseline(case)
        except Exception as e:
            logger.error("Baseline failed for %s: %s", case_id, e)
            baseline_predicted = []

        # Run LLM Judge on RAG output
        if rag_failed:
            llm_judge_score = {
                "accuracy_score": 0,
                "consistency_score": 0,
                "citation_score": 0,
                "reasoning": "Skipped: RAG failed",
            }
        else:
            try:
                expected_answer_str = (
                    f"Expected IDs: {expected_ids}, Rationale: {case.get('rationale', '')}"
                )
                user_input = case["input"]
                prefs = ", ".join(user_input.get("sensory_preferences", []))
                question_parts = [f"Recommend shoes for: {prefs}"]
                if user_input.get("player_archetype"):
                    question_parts.append(f"archetype: {user_input['player_archetype']}")
                if user_input.get("position"):
                    question_parts.append(f"position: {user_input['position']}")
                if user_input.get("budget_max_krw"):
                    question_parts.append(f"budget: {user_input['budget_max_krw']} KRW")
                question_str = ", ".join(question_parts)
                llm_judge_score = evaluate_with_llm_judge(
                    question=question_str,
                    generated_answer=raw_response,
                    expected_answer=expected_answer_str,
                    context=context_text,
                    prompt_template=LLM_JUDGE_GEAR_PROMPT,
                    system_prompt=LLM_JUDGE_GEAR_SYSTEM_PROMPT,
                )
            except Exception as e:
                logger.error("LLM Judge failed for %s: %s", case_id, e)
                llm_judge_score = {
                    "accuracy_score": 0,
                    "consistency_score": 0,
                    "citation_score": 0,
                    "reasoning": "Error",
                }

        rag_results.append(
            {
                "predicted_ids": rag_predicted,
                "expected_ids": expected_ids,
            }
        )
        baseline_results.append(
            {
                "predicted_ids": baseline_predicted,
                "expected_ids": expected_ids,
            }
        )
        llm_judge_results.append(llm_judge_score)

        details.append(
            {
                "id": case_id,
                "description": case["description"],
                "expected_ids": expected_ids,
                "rag_predicted": rag_predicted,
                "baseline_predicted": baseline_predicted,
                "llm_judge_score": llm_judge_score,
            }
        )

        logger.info(
            "  %s → RAG: %s | Baseline: %s | Expected: %s | Judge: %s",  # noqa: E501
            case_id,
            rag_predicted,
            baseline_predicted,
            expected_ids,
            f"A:{llm_judge_score.get('accuracy_score')} C:{llm_judge_score.get('consistency_score')} Cit:{llm_judge_score.get('citation_score')}"  # noqa: E501,
        )

    return {
        "rag_metrics": compute_gear_metrics(rag_results),
        "baseline_metrics": compute_gear_metrics(baseline_results),
        "llm_judge_metrics": compute_llm_judge_metrics(llm_judge_results),
        "details": details,
    }
