"""CLI entry point for RAG evaluation.

Usage:
    uv run python scripts/evaluate.py --all
    uv run python scripts/evaluate.py --gear
    uv run python scripts/evaluate.py --whistle
    uv run python scripts/evaluate.py --skill
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path for src.* imports
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.eval.report import generate_report  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="RAG Evaluation Runner")
    parser.add_argument(
        "--all", action="store_true", help="Run all evaluations"
    )
    parser.add_argument(
        "--gear", action="store_true", help="Run Gear Advisor evaluation"
    )
    parser.add_argument(
        "--whistle", action="store_true", help="Run Whistle evaluation"
    )
    parser.add_argument(
        "--skill", action="store_true", help="Run Skill Lab evaluation"
    )
    args = parser.parse_args()

    if not any([args.all, args.gear, args.whistle, args.skill]):
        parser.print_help()
        sys.exit(1)

    gear_data = None
    whistle_data = None
    skill_data = None

    if args.all or args.gear:
        logger.info("=== Running Gear Advisor Evaluation ===")
        from scripts.eval.runners.gear_runner import run_gear_evaluation

        gear_data = run_gear_evaluation()
        logger.info(
            "Gear RAG metrics: %s", gear_data["rag_metrics"]
        )
        logger.info(
            "Gear Baseline metrics: %s", gear_data["baseline_metrics"]
        )

    if args.all or args.whistle:
        logger.info("=== Running Whistle Evaluation ===")
        from scripts.eval.runners.whistle_runner import run_whistle_evaluation

        whistle_data = run_whistle_evaluation()
        logger.info(
            "Whistle Citation Hit Rate: %.2f",
            whistle_data["citation_hit_rate"],
        )

    if args.all or args.skill:
        logger.info("=== Running Skill Lab Evaluation ===")
        from scripts.eval.runners.skill_runner import run_skill_evaluation

        skill_data = run_skill_evaluation()
        logger.info(
            "Skill Constraint Pass Rate: %s",
            skill_data["constraint_pass_rate"],
        )

    report_path = generate_report(gear_data, whistle_data, skill_data)
    logger.info("Report saved to: %s", report_path)


if __name__ == "__main__":
    main()
