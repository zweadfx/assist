"""Skill Lab evaluation runner.

Runs each skill case through the coach agent graph,
then verifies equipment and time constraints.
"""

import json
import logging
from pathlib import Path

from langchain_core.messages import HumanMessage

from scripts.eval.metrics import constraint_pass_rate
from src.models.skill_schema import SkillLabResponse
from src.services.agents.coach_agent import coach_agent_graph

logger = logging.getLogger(__name__)

DATASETS_DIR = Path(__file__).resolve().parent.parent / "datasets"

# Equipment keywords that indicate usage of specific equipment
EQUIPMENT_KEYWORDS = {
    "hoop": ["hoop", "rim", "basket", "backboard"],
    "cones": ["cone", "cones", "marker", "markers"],
    "partner": ["partner", "teammate"],
    "ball": ["ball", "dribble", "shoot", "pass"],
}

# Phrases that mention equipment conceptually but don't require it
SIMULATION_EXCEPTIONS = [
    "imagin", "visualiz", "pretend", "shadow", "simulate",
    "as if", "invisible",
]


def _check_equipment_constraint(
    response: SkillLabResponse,
    max_equipment: list[str],
) -> bool:
    """Check if response only uses allowed equipment.

    Scans step names and descriptions for equipment keywords
    that are NOT in the user's available equipment list.
    """
    allowed = set(max_equipment)

    # Collect all text from steps for scanning
    texts = []
    for step in response.steps:
        texts.append(step.name.lower())
        texts.append(step.description.lower())

    combined_text = " ".join(texts)

    for equip, keywords in EQUIPMENT_KEYWORDS.items():
        if equip in allowed or equip == "ball":
            # ball is almost always implicitly allowed in basketball drills
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
    """Check if total duration matches the target time."""
    actual = sum(step.duration_min for step in response.steps)
    matches = actual == target_time_min
    if not matches:
        logger.warning(
            "Time violation: sum of steps=%d, expected=%d",
            actual,
            target_time_min,
        )
    return matches


def _run_single_skill(case: dict) -> SkillLabResponse:
    """Run a single skill case through the coach agent."""
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
        },
    }

    final_state = coach_agent_graph.invoke(initial_state)
    raw = final_state.get("final_response", "")
    return SkillLabResponse.model_validate_json(raw)


def run_skill_evaluation() -> dict:
    """Run full skill lab evaluation for all cases.

    Returns:
        Dict with 'constraint_pass_rate', 'n_cases', and 'details' per case.
    """
    with open(DATASETS_DIR / "skill_cases.json", encoding="utf-8") as f:
        cases = json.load(f)

    constraint_results = []
    details = []

    for case in cases:
        case_id = case["id"]
        constraints = case["constraints"]
        logger.info("Running skill case: %s", case_id)

        try:
            response = _run_single_skill(case)
            equip_pass = _check_equipment_constraint(
                response, constraints["max_equipment"]
            )
            time_pass = _check_time_constraint(
                response, constraints["target_time_min"]
            )
            step_summary = [
                {"name": s.name, "duration_min": s.duration_min}
                for s in response.steps
            ]
        except Exception as e:
            logger.error("Skill failed for %s: %s", case_id, e)
            equip_pass = False
            time_pass = False
            step_summary = []

        constraint_results.append({
            "equipment_pass": equip_pass,
            "time_pass": time_pass,
        })
        details.append({
            "id": case_id,
            "description": case["description"],
            "equipment_pass": equip_pass,
            "time_pass": time_pass,
            "steps": step_summary,
        })

        logger.info(
            "  %s → equipment: %s | time: %s",
            case_id,
            "PASS" if equip_pass else "FAIL",
            "PASS" if time_pass else "FAIL",
        )

    return {
        "constraint_pass_rate": constraint_pass_rate(constraint_results),
        "n_cases": len(cases),
        "details": details,
    }
