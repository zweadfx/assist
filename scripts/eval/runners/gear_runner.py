"""Gear Advisor evaluation runner.

Runs each gear case through both RAG and No-RAG baseline pipelines,
then collects predicted shoe IDs for metric computation.
"""

import json
import logging
from pathlib import Path

from langchain_core.messages import HumanMessage

from scripts.eval.baseline import run_no_rag_gear
from scripts.eval.metrics import compute_gear_metrics
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
    return [
        _resolve_shoe_id(shoe.shoe_id, shoe.model_name)
        for shoe in response.shoes
    ]


def _run_single_rag(case: dict) -> list[str]:
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
    response = GearAdvisorResponse.model_validate_json(raw)
    return _extract_shoe_ids(response)


def _run_single_baseline(case: dict) -> list[str]:
    """Run a single case through the No-RAG baseline."""
    response = run_no_rag_gear(case["input"])
    return _extract_shoe_ids(response)


def run_gear_evaluation() -> dict:
    """Run full gear evaluation: RAG vs baseline for all cases.

    Returns:
        Dict with 'rag_results', 'baseline_results', 'rag_metrics',
        'baseline_metrics', and 'details' per case.
    """
    with open(DATASETS_DIR / "gear_cases.json", encoding="utf-8") as f:
        cases = json.load(f)

    rag_results = []
    baseline_results = []
    details = []

    for case in cases:
        case_id = case["id"]
        expected_ids = case["expected_shoe_ids"]
        logger.info("Running gear case: %s", case_id)

        # RAG
        try:
            rag_predicted = _run_single_rag(case)
        except Exception as e:
            logger.error("RAG failed for %s: %s", case_id, e)
            rag_predicted = []

        # Baseline
        try:
            baseline_predicted = _run_single_baseline(case)
        except Exception as e:
            logger.error("Baseline failed for %s: %s", case_id, e)
            baseline_predicted = []

        rag_results.append({
            "predicted_ids": rag_predicted,
            "expected_ids": expected_ids,
        })
        baseline_results.append({
            "predicted_ids": baseline_predicted,
            "expected_ids": expected_ids,
        })
        details.append({
            "id": case_id,
            "description": case["description"],
            "expected_ids": expected_ids,
            "rag_predicted": rag_predicted,
            "baseline_predicted": baseline_predicted,
        })

        logger.info(
            "  %s → RAG: %s | Baseline: %s | Expected: %s",
            case_id,
            rag_predicted,
            baseline_predicted,
            expected_ids,
        )

    return {
        "rag_metrics": compute_gear_metrics(rag_results),
        "baseline_metrics": compute_gear_metrics(baseline_results),
        "details": details,
    }
