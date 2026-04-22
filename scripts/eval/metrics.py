"""Evaluation metrics for RAG assessment."""

from typing import Dict, List


def hit_at_k(predicted_ids: List[str], expected_ids: List[str], k: int = 3) -> int:
    """Check if any expected ID appears in top-k predictions.

    Returns 1 if hit, 0 otherwise.
    """
    top_k = predicted_ids[:k]
    return 1 if any(eid in top_k for eid in expected_ids) else 0


def reciprocal_rank(predicted_ids: List[str], expected_ids: List[str]) -> float:
    """Compute reciprocal rank of first matching expected ID.

    Returns 1/rank if found, 0.0 otherwise.
    """
    for i, pid in enumerate(predicted_ids):
        if pid in expected_ids:
            return 1.0 / (i + 1)
    return 0.0


def compute_gear_metrics(
    results: List[dict],
) -> dict:
    """Aggregate Hit@3 and MRR across gear evaluation results.

    Each result dict must have 'predicted_ids' and 'expected_ids' keys.
    """
    n = len(results)
    if n == 0:
        return {"hit_at_3": 0.0, "mrr": 0.0, "n_cases": 0}

    total_hits = sum(
        hit_at_k(r["predicted_ids"], r["expected_ids"], k=3) for r in results
    )
    total_rr = sum(
        reciprocal_rank(r["predicted_ids"], r["expected_ids"]) for r in results
    )

    return {
        "hit_at_3": total_hits / n,
        "mrr": total_rr / n,
        "n_cases": n,
    }


def citation_hit_rate(results: List[dict]) -> float:
    """Compute fraction of cases where expected articles were cited.

    Each result dict must have 'predicted_articles' and 'expected_articles'.
    """
    if not results:
        return 0.0

    hits = 0
    for r in results:
        predicted = set(r["predicted_articles"])
        expected = set(r["expected_articles"])
        if expected.issubset(predicted):
            hits += 1

    return hits / len(results)


def constraint_pass_rate(results: List[dict]) -> dict:
    """Compute pass rates for equipment and time constraints.

    Each result dict must have 'equipment_pass' (bool) and 'time_pass' (bool).
    """
    if not results:
        return {"equipment": 0.0, "time": 0.0}

    equip_passes = sum(1 for r in results if r["equipment_pass"])
    time_passes = sum(1 for r in results if r["time_pass"])
    n = len(results)

    return {
        "equipment": equip_passes / n,
        "time": time_passes / n,
    }


def compute_llm_judge_metrics(results: List[Dict[str, any]]) -> dict:
    """Aggregate LLM Judge scores.

    Each result dict must have 'accuracy_score', 'consistency_score',
    and 'citation_score'.
    """
    if not results:
        return {"accuracy": 0.0, "consistency": 0.0, "citation": 0.0}

    valid_results = [r for r in results if r.get("accuracy_score") is not None]
    n = len(valid_results)
    if n == 0:
        return {"accuracy": 0.0, "consistency": 0.0, "citation": 0.0}

    total_accuracy = sum(r["accuracy_score"] for r in valid_results)
    total_consistency = sum(r["consistency_score"] for r in valid_results)
    total_citation = sum(r["citation_score"] for r in valid_results)

    return {
        "accuracy": total_accuracy / n,
        "consistency": total_consistency / n,
        "citation": total_citation / n,
    }
