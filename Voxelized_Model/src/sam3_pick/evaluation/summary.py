"""Aggregate rollout success by target and generalization group."""

from collections import Counter


def summarize(results, elapsed):
    summary = dict(
        cases=len(results),
        successes=sum((r["success"] for r in results)),
        success_rate=sum((r["success"] for r in results)) / len(results),
        elapsed_s=elapsed,
        reasons=dict(Counter((r["reason"] for r in results))),
    )
    for key in ("group", "target"):
        summary["by_" + key] = {}
        for value in sorted({r[key] for r in results}):
            current = [r for r in results if r[key] == value]
            successes = sum((r["success"] for r in current))
            summary["by_" + key][value] = dict(
                cases=len(current),
                successes=successes,
                success_rate=successes / len(current),
            )
    return summary
