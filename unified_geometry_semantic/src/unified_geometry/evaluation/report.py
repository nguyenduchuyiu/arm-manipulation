"""Summarize the current experiment and export raw prediction/GT viewers."""

import json
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from unified_geometry.paths import ROOT, write


def panels(path, cases, title, labels):
    columns = min(4, len(cases))
    rows = (len(cases) + columns - 1) // columns
    figure = make_subplots(
        rows=rows,
        cols=columns,
        specs=[[{"type": "scene"}] * columns for _ in range(rows)],
        subplot_titles=labels,
    )
    for i, (prediction, truth) in enumerate(cases):
        row, col = divmod(i, columns)
        for name, cloud, color in [
            ("GT", truth, "#299456"),
            ("Prediction", prediction, "#e67429"),
        ]:
            p = cloud * 1000
            figure.add_trace(
                go.Scatter3d(
                    x=p[:, 0],
                    y=p[:, 1],
                    z=p[:, 2],
                    mode="markers",
                    name=name,
                    legendgroup=name,
                    showlegend=i == 0,
                    marker=dict(size=3, color=color),
                ),
                row=row + 1,
                col=col + 1,
            )
        figure.update_scenes(
            aspectmode="data",
            xaxis_title="X mm",
            yaxis_title="Y mm",
            zaxis_title="Z mm",
            row=row + 1,
            col=col + 1,
        )
    figure.update_layout(title=title, height=500 * rows)
    figure.write_html(path, include_plotlyjs=True)


def main():
    canonical = json.loads((ROOT / "evaluation/summary.json").read_text())
    ood = json.loads((ROOT / "ood/summary.json").read_text())
    controls = json.loads((ROOT / "test_swap/summary.json").read_text())
    lines = [
        "# E178 results",
        "",
        "| Model/split | EMD mm | Coverage @5mm % |",
        "|---|---:|---:|",
    ]
    for model, splits in canonical["results"].items():
        for split, r in splits.items():
            lines.append(
                f"| {model}/{split} | {r['emd_mm']:.3f} | {r['coverage_5mm_percent']:.2f} |"
            )
    for title, groups in [
        ("Physical OOD", ood["results"]),
        ("Correspondence", {"features": controls["results"]}),
    ]:
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| Condition | EMD mm | Selection % |",
                "|---|---:|---:|",
            ]
        )
        for group, conditions in groups.items():
            for name, r in conditions.items():
                lines.append(
                    f"| {group}/{name} | {r['emd_mm']:.3f} | {r['target_selection_percent']:.2f} |"
                )
    lines.extend(
        [
            "",
            "One seed; ideal depth and fixed clutter. Geometry covers visible surfaces; semantics are four color tasks.",
        ]
    )
    (ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    cases = []
    labels = []
    for split in ["train", "iid"]:
        with np.load(ROOT / f"evaluation/{split}_E178_best.npz") as d:
            for scene, task in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                cases.append((d["prediction"][scene, task], d["target"][scene, task]))
                labels.append(f"{split}/{scene}/{task}")
    panels(ROOT / "evaluation/best_vs_GT.html", cases, "E178 raw XYZ, no alignment", labels)
    for group, conditions in ood["results"].items():
        cases = []
        labels = []
        for name in conditions:
            with np.load(ROOT / f"ood/{group}/{name}.npz") as d:
                cases.append((d["prediction"][0, 1], d["target"][0, 1]))
                labels.append(name)
        panels(ROOT / f"ood/{group}/best_vs_GT.html", cases, f"E178 {group}", labels)
    write(
        ROOT / "completion.json",
        dict(
            status="complete",
            checkpoint=canonical["checkpoints"]["E178_best"],
            IID=canonical["results"]["E178_best"]["iid"],
            provenance=canonical["provenance"],
        ),
    )


if __name__ == "__main__":
    main()
