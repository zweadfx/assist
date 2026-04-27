"""Skill Lab evaluation runner.

Runs each skill case through both the current LLM-only agent and the
RAG-based baseline (old pipeline), then compares constraint compliance.
"""

import json
import logging
from pathlib import Path

from langchain_core.messages import HumanMessage

from scripts.eval.baseline_skill import run_rag_baseline_skill
from scripts.eval.metrics import constraint_pass_rate
from src.models.skill_schema import SkillLabResponse
from src.services.agents.coach_agent import coach_agent_graph

logger = logging.getLogger(__name__)

DATASETS_DIR = Path(__file__).resolve().parent.parent / "datasets"

EQUIPMENT_KEYWORDS = {
    "hoop": ["hoop", "rim", "basket", "backboard"],
    "cones": ["cone", "cones", "marker", "markers"],
    "partner": ["partner", "teammate"],
    "ball": ["ball", "dribble", "shoot", "pass"],
}


def _check_equipment_constraint(
    response: SkillLabResponse,
    max_equipment: list[str],
) -> bool:
    allowed = set(max_equipment)

    texts = []
    for step in response.steps:
        texts.append(step.name.lower())
        texts.append(step.description.lower())

    combined_text = " ".join(texts)

    for equip, keywords in EQUIPMENT_KEYWORDS.items():
        if equip in allowed or equip == "ball":
            continue
        for keyword in keywords:
            if keyword in combined_text:
                logger.warning(
                    "Equipment violation: '%s' found but '%s' not in allowed %s",
                    keyword,
                    equip,
                    allowed,
                )
                return False

    return True


def _check_time_constraint(
    response: SkillLabResponse,
    target_time_min: int,
) -> bool:
    actual = sum(step.duration_min for step in response.steps)
    matches = actual == target_time_min
    if not matches:
        logger.warning(
            "Time violation: sum of steps=%d, expected=%d",
            actual,
            target_time_min,
        )
    return matches


def _run_llm_only(case: dict) -> SkillLabResponse:
    """Run a single case through the current LLM-only coach agent."""
    user_input = case["input"]
    skill_desc = (
        f"{user_input.get('skill_level', '')} "
        f"{user_input.get('category', '')} training"
    )
    initial_state = {
        "messages": [
            HumanMessage(
                content=f"Generate a micro-step skill breakdown for {skill_desc}"
            )
        ],
        "user_info": {
            "skill_level": user_input["skill_level"],
            "category": user_input["category"],
            "available_time_min": user_input["available_time_min"],
            "equipment": user_input.get("equipment", []),
            "language": user_input.get("language", "en"),
            "specific_skill": user_input.get("specific_skill"),
        },
    }

    final_state = coach_agent_graph.invoke(initial_state)
    raw = final_state.get("final_response", "")
    return SkillLabResponse.model_validate_json(raw)


def _run_rag_baseline(case: dict) -> SkillLabResponse:
    """Run a single case through the RAG-based baseline (old pipeline)."""
    user_input = case["input"]
    return run_rag_baseline_skill({
        "skill_level": user_input["skill_level"],
        "category": user_input["category"],
        "available_time_min": user_input["available_time_min"],
        "equipment": user_input.get("equipment", []),
        "language": user_input.get("language", "en"),
        "specific_skill": user_input.get("specific_skill"),
    })


def _evaluate_response(
    response: SkillLabResponse,
    constraints: dict,
) -> tuple[bool, bool, list]:
    equip_pass = _check_equipment_constraint(response, constraints["max_equipment"])
    time_pass = _check_time_constraint(response, constraints["target_time_min"])
    steps = [{"name": s.name, "duration_min": s.duration_min} for s in response.steps]
    return equip_pass, time_pass, steps


def run_skill_evaluation() -> dict:
    """Run full skill lab evaluation comparing LLM-only vs RAG baseline.

    Returns:
        Dict with constraint pass rates for both pipelines, n_cases, and per-case details.
    """
    with open(DATASETS_DIR / "skill_cases.json", encoding="utf-8") as f:
        cases = json.load(f)

    llm_results = []
    rag_results = []
    details = []

    for case in cases:
        case_id = case["id"]
        constraints = case["constraints"]
        logger.info("Running skill case: %s", case_id)

        # LLM-only (current)
        try:
            llm_response = _run_llm_only(case)
            llm_equip, llm_time, llm_steps = _evaluate_response(llm_response, constraints)
        except Exception as e:
            logger.error("LLM-only failed for %s: %s", case_id, e)
            llm_equip, llm_time, llm_steps = False, False, []

        # RAG baseline (old pipeline)
        try:
            rag_response = _run_rag_baseline(case)
            rag_equip, rag_time, rag_steps = _evaluate_response(rag_response, constraints)
        except Exception as e:
            logger.error("RAG baseline failed for %s: %s", case_id, e)
            rag_equip, rag_time, rag_steps = False, False, []

        llm_results.append({"equipment_pass": llm_equip, "time_pass": llm_time})
        rag_results.append({"equipment_pass": rag_equip, "time_pass": rag_time})

        details.append({
            "id": case_id,
            "description": case["description"],
            "llm_equipment_pass": llm_equip,
            "llm_time_pass": llm_time,
            "llm_steps": llm_steps,
            "rag_equipment_pass": rag_equip,
            "rag_time_pass": rag_time,
            "rag_steps": rag_steps,
        })

        logger.info(
            "  %s → LLM(equip:%s time:%s) | RAG(equip:%s time:%s)",
            case_id,
            "PASS" if llm_equip else "FAIL",
            "PASS" if llm_time else "FAIL",
            "PASS" if rag_equip else "FAIL",
            "PASS" if rag_time else "FAIL",
        )

    return {
        "llm_constraint_pass_rate": constraint_pass_rate(llm_results),
        "rag_constraint_pass_rate": constraint_pass_rate(rag_results),
        "n_cases": len(cases),
        "details": details,
    }
