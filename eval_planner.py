"""Evaluate dependency-guided planner JSON outputs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from planners.dependency_planner import PlannerSample
from planners.planner_utils import mean, set_comparison


def oracle_set_for_grasp(sample: PlannerSample, grasp_id: str) -> tuple[list[str], dict[str, Any]]:
    label = sample.oracle_min_by_grasp.get(grasp_id, {})
    status = str(label.get("status", ""))
    minimal = list(label.get("minimal_blocker_set") or [])
    if status in {"already_feasible", "solved_within_depth"}:
        return sorted(str(x) for x in minimal), label

    dep_edges = sample.oracle_dependencies.get(grasp_id, {})
    dep_set = sorted(object_id for object_id, edge in dep_edges.items() if float(edge.get("any", 0.0)) >= 0.5)
    if dep_set:
        fallback = dict(label)
        fallback["status"] = status or "gt_dependency_fallback"
        fallback["minimal_blocker_set"] = dep_set
        fallback["minimal_blocker_set_size"] = len(dep_set)
        return dep_set, fallback
    return sorted(str(x) for x in minimal), label


def grasp_success_after_removals(sample: PlannerSample, grasp_id: str, removed: list[str]) -> bool:
    if not grasp_id:
        return False
    removed_set = set(removed)
    label = sample.oracle_min_by_grasp.get(grasp_id, {})
    status = str(label.get("status", ""))
    minimal = set(str(x) for x in (label.get("minimal_blocker_set") or []))
    feasibility = sample.grasp_feasibility.get(grasp_id, {})

    if bool(feasibility.get("feasible", False)) or status == "already_feasible":
        return True
    if status == "solved_within_depth":
        return minimal.issubset(removed_set)

    dep_edges = sample.oracle_dependencies.get(grasp_id, {})
    dep_set = {object_id for object_id, edge in dep_edges.items() if float(edge.get("any", 0.0)) >= 0.5}
    if dep_set:
        return dep_set.issubset(removed_set)
    return False


def annotate_result_with_oracle(result: dict[str, Any], sample: PlannerSample) -> dict[str, Any]:
    out = dict(result)
    grasp_id = str(out.get("selected_grasp_id") or "")
    removed = [str(x) for x in out.get("removal_sequence", [])]
    oracle_set, oracle_label = oracle_set_for_grasp(sample, grasp_id)
    predicted_set = out.get("initial_predicted_blocker_set", out.get("predicted_blocker_set", []))
    comparison = set_comparison(predicted_set, oracle_set)
    oracle_min_removals = int(oracle_label.get("minimal_blocker_set_size", len(oracle_set)) or 0)
    num_removals = int(out.get("num_removals", len(removed)) or 0)
    success = grasp_success_after_removals(sample, grasp_id, removed)
    oracle_status = oracle_label.get("status", "") or out.get("oracle_grasp_status", "")
    out.update(
        {
            "oracle_min_blocker_set": oracle_set,
            "oracle_min_removals": oracle_min_removals,
            "oracle_grasp_status": oracle_status,
            "target_retrieval_success": bool(success),
            "selected_grasp_feasibility_after_removals": bool(success),
            "excess_removal": int(num_removals - oracle_min_removals),
            "optimality_gap": float((num_removals - oracle_min_removals) / max(float(oracle_min_removals), 1.0)),
            "blocker_set_precision": comparison["precision"],
            "blocker_set_recall": comparison["recall"],
            "blocker_set_f1": comparison["f1"],
            "blocker_set_iou": comparison["iou"],
        }
    )
    return out


def group_key(result: dict[str, Any]) -> str:
    return f"{result.get('planner_type', 'unknown')}_{result.get('mode', 'unknown')}"


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        groups[group_key(result)].append(result)

    summaries: dict[str, Any] = {}
    for key, items in sorted(groups.items()):
        summaries[key] = {
            "num_samples": len(items),
            "target_retrieval_success_rate": mean(float(bool(x.get("target_retrieval_success"))) for x in items),
            "mean_num_removals": mean(float(x.get("num_removals", 0.0)) for x in items),
            "mean_excess_removal": mean(float(x.get("excess_removal", 0.0)) for x in items),
            "mean_optimality_gap": mean(float(x.get("optimality_gap", 0.0)) for x in items),
            "mean_blocker_set_precision": mean(float(x.get("blocker_set_precision", 0.0)) for x in items),
            "mean_blocker_set_recall": mean(float(x.get("blocker_set_recall", 0.0)) for x in items),
            "mean_blocker_set_f1": mean(float(x.get("blocker_set_f1", 0.0)) for x in items),
            "mean_blocker_set_iou": mean(float(x.get("blocker_set_iou", 0.0)) for x in items),
            "selected_grasp_feasibility_after_removals": mean(
                float(bool(x.get("selected_grasp_feasibility_after_removals"))) for x in items
            ),
            "mean_planning_time_sec": mean(float(x.get("planning_time_sec", 0.0)) for x in items),
        }
    return {"overall": summarize_flat(results), "by_planner": summaries}


def summarize_flat(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {"num_results": 0}
    return {
        "num_results": len(items),
        "target_retrieval_success_rate": mean(float(bool(x.get("target_retrieval_success"))) for x in items),
        "mean_num_removals": mean(float(x.get("num_removals", 0.0)) for x in items),
        "mean_excess_removal": mean(float(x.get("excess_removal", 0.0)) for x in items),
        "mean_blocker_set_iou": mean(float(x.get("blocker_set_iou", 0.0)) for x in items),
        "mean_planning_time_sec": mean(float(x.get("planning_time_sec", 0.0)) for x in items),
    }


def summary_text(metrics: dict[str, Any]) -> str:
    lines = ["Dependency-Guided Planner Summary", ""]
    for key, values in metrics.get("by_planner", {}).items():
        lines.append(
            f"{key}: success={values['target_retrieval_success_rate']:.3f}, "
            f"removals={values['mean_num_removals']:.3f}, "
            f"excess={values['mean_excess_removal']:.3f}, "
            f"IoU={values['mean_blocker_set_iou']:.3f}, "
            f"time={values['mean_planning_time_sec']:.4f}s"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("outputs/planner_results.json"))
    parser.add_argument("--metrics", type=Path, default=Path("outputs/planner_metrics.json"))
    parser.add_argument("--summary", type=Path, default=Path("outputs/planner_summary.txt"))
    args = parser.parse_args()

    results = json.loads(args.results.read_text(encoding="utf-8"))
    metrics = summarize_results(results)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    args.summary.write_text(summary_text(metrics), encoding="utf-8")
    print(summary_text(metrics))


if __name__ == "__main__":
    main()
