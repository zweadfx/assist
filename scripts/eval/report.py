"""Markdown report generator for RAG evaluation results."""

from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _gear_section(gear_data: dict) -> str:
    """Build Gear Advisor section of the report."""
    rag = gear_data["rag_metrics"]
    baseline = gear_data["baseline_metrics"]
    details = gear_data["details"]

    lines = [
        "## Gear Advisor: RAG vs No-RAG Baseline",
        "",
        "| Metric | RAG | Baseline | Delta |",
        "|--------|-----|----------|-------|",
        f"| Hit@3 | {rag['hit_at_3']:.2f} | {baseline['hit_at_3']:.2f} "
        f"| {rag['hit_at_3'] - baseline['hit_at_3']:+.2f} |",
        f"| MRR | {rag['mrr']:.2f} | {baseline['mrr']:.2f} "
        f"| {rag['mrr'] - baseline['mrr']:+.2f} |",
        f"| Cases | {rag['n_cases']} | {baseline['n_cases']} | - |",
        "",
        "### Case Details",
        "",
        "| ID | Description | Expected | RAG Predicted | Baseline Predicted | RAG Hit | Baseline Hit |",
        "|----|-------------|----------|---------------|-------------------|---------|-------------|",
    ]

    for d in details:
        expected = ", ".join(d["expected_ids"][:3])
        if len(d["expected_ids"]) > 3:
            expected += "..."
        rag_pred = ", ".join(d["rag_predicted"][:3]) or "-"
        base_pred = ", ".join(d["baseline_predicted"][:3]) or "-"
        rag_hit = "O" if any(
            sid in d["rag_predicted"][:3] for sid in d["expected_ids"]
        ) else "X"
        base_hit = "O" if any(
            sid in d["baseline_predicted"][:3] for sid in d["expected_ids"]
        ) else "X"
        lines.append(
            f"| {d['id']} | {d['description']} | {expected} "
            f"| {rag_pred} | {base_pred} | {rag_hit} | {base_hit} |"
        )

    lines.append("")
    return "\n".join(lines)


def _whistle_section(whistle_data: dict) -> str:
    """Build Whistle section of the report."""
    rate = whistle_data["citation_hit_rate"]
    n = whistle_data["n_cases"]
    details = whistle_data["details"]

    lines = [
        "## Whistle: Citation Accuracy",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Citation Hit Rate | {rate:.2f} |",
        f"| Cases | {n} |",
        "",
        "### Case Details",
        "",
        "| ID | Description | Expected Decision | Predicted Decision | Match | Expected Articles | Predicted Articles |",
        "|----|-------------|-------------------|-------------------|-------|-------------------|-------------------|",
    ]

    for d in details:
        match = "O" if d["decision_match"] else "X"
        exp_art = ", ".join(d["expected_articles"])
        pred_art = ", ".join(d["predicted_articles"]) or "-"
        lines.append(
            f"| {d['id']} | {d['description']} | {d['expected_decision']} "
            f"| {d['predicted_decision']} | {match} | {exp_art} | {pred_art} |"
        )

    lines.append("")
    return "\n".join(lines)


def _skill_section(skill_data: dict) -> str:
    """Build Skill Lab section of the report."""
    rates = skill_data["constraint_pass_rate"]
    n = skill_data["n_cases"]
    details = skill_data["details"]

    lines = [
        "## Skill Lab: Constraint Compliance",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Equipment Pass Rate | {rates['equipment']:.2f} |",
        f"| Time Pass Rate | {rates['time']:.2f} |",
        f"| Cases | {n} |",
        "",
        "### Case Details",
        "",
        "| ID | Description | Equipment | Time | Steps |",
        "|----|-------------|-----------|------|-------|",
    ]

    for d in details:
        equip = "PASS" if d["equipment_pass"] else "FAIL"
        time = "PASS" if d["time_pass"] else "FAIL"
        if d["steps"]:
            step_summary = ", ".join(
                f"{s['name']}({s['duration_min']}m)" for s in d["steps"]
            )
        else:
            step_summary = "-"
        lines.append(
            f"| {d['id']} | {d['description']} | {equip} | {time} "
            f"| {step_summary} |"
        )

    lines.append("")
    return "\n".join(lines)


def generate_report(
    gear_data: dict | None = None,
    whistle_data: dict | None = None,
    skill_data: dict | None = None,
) -> str:
    """Generate a markdown evaluation report and save to results directory.

    Args:
        gear_data: Output from gear_runner.run_gear_evaluation()
        whistle_data: Output from whistle_runner.run_whistle_evaluation()
        skill_data: Output from skill_runner.run_skill_evaluation()

    Returns:
        Path to the generated report file.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    lines = [
        f"# RAG Evaluation Report",
        f"",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
    ]

    if gear_data:
        lines.append(_gear_section(gear_data))
        lines.append("---\n")

    if whistle_data:
        lines.append(_whistle_section(whistle_data))
        lines.append("---\n")

    if skill_data:
        lines.append(_skill_section(skill_data))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS_DIR / f"eval_report_{timestamp}.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    return str(report_path)
