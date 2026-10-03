"""Generate final tables and three mechanism figures from archived SRLA experiments."""

import argparse
import csv
import hashlib
import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_csv(path):
    with path.open(encoding="utf-8-sig") as stream:
        return [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(stream)]


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--inference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    fast = args.experiment_root / "YOLOv5-srla-fast/runs/srla-fast"
    frozen = fast / "freeze/srla-hard-3seed-final"
    soft = args.experiment_root / "YOLOv5-srla-soft/runs/srla-soft"
    control = args.experiment_root / "YOLOv5-matched-control/runs/matched-gating/freeze/matched-control-seed0-final"
    inputs = []

    def results(path):
        inputs.append(path)
        rows = read_csv(path)
        assert len(rows) == 100 and [int(r["epoch"]) for r in rows] == list(range(100))
        best = max(rows, key=lambda r: 0.1 * float(r["metrics/mAP_0.5"]) + 0.9 * float(r["metrics/mAP_0.5:0.95"]))
        return {
            "best_epoch": int(best["epoch"]) + 1,
            "map50": 100 * float(best["metrics/mAP_0.5"]),
            "map5095": 100 * float(best["metrics/mAP_0.5:0.95"]),
            "last_map5095": 100 * float(rows[-1]["metrics/mAP_0.5:0.95"]),
        }

    rows = []
    for seed in range(3):
        name = "formal100" if seed == 0 else f"formal100_seed{seed}"
        for mode in ["baseline", "fast"]:
            values = results(frozen / name / mode / "results.csv")
            status = fast / name / f"{mode}_status.json"
            inputs.append(status)
            state = json.loads(status.read_text())
            assert state["status"] == "completed" and state["completed_epochs"] == 100
            rows.append(
                dict(
                    seed=seed,
                    method="Baseline" if mode == "baseline" else "Hard-SRLA",
                    **values,
                    training_minutes=state["elapsed_seconds"] / 60,
                )
            )
    write_csv(args.out / "main_per_seed.csv", rows)
    summary = []
    for metric in ["map50", "map5095", "last_map5095", "training_minutes"]:
        b = [r[metric] for r in rows if r["method"] == "Baseline"]
        h = [r[metric] for r in rows if r["method"] == "Hard-SRLA"]
        d = [y - x for x, y in zip(b, h)]
        summary.append(
            {
                "metric": metric,
                "baseline_mean": st.mean(b),
                "baseline_sd": st.stdev(b),
                "hard_mean": st.mean(h),
                "hard_sd": st.stdev(h),
                "paired_mean": st.mean(d),
                "paired_sd": st.stdev(d),
            }
        )
    write_csv(args.out / "main_summary.csv", summary)
    ablation = [
        {"method": r["method"], "map5095": r["map5095"], "best_epoch": r["best_epoch"]} for r in rows if r["seed"] == 0
    ]
    for method, path in [
        ("Matched Random", control / "formal100_seed0/control/results.csv"),
        ("Soft-SRLA", soft / "formal100_seed0/soft/results.csv"),
    ]:
        r = results(path)
        ablation.append({"method": method, "map5095": r["map5095"], "best_epoch": r["best_epoch"]})
    order = ["Baseline", "Matched Random", "Hard-SRLA", "Soft-SRLA"]
    ablation.sort(key=lambda r: order.index(r["method"]))
    write_csv(args.out / "ablations.csv", ablation)
    path = fast / "formal100/postrun_analysis/epoch_assignment_candidates.csv"
    inputs.append(path)
    raw = read_csv(path)
    assignment = []
    for level in ["P3", "P4", "P5"]:
        sub = [r for r in raw if r["level"] == level]
        assert len(sub) == 100
        before = sum(int(r["original_candidates"]) for r in sub)
        kept = sum(int(r["kept_candidates"]) for r in sub)
        assignment.append(
            {
                "level": level,
                "original": before,
                "kept": kept,
                "removed": before - kept,
                "removal_percent": 100 * (before - kept) / before,
            }
        )
    write_csv(args.out / "assignment.csv", assignment)
    path = soft / "hard_mechanism/size_comparison.csv"
    inputs.append(path)
    raw = read_csv(path)
    sizes = []
    for frame in ["original_pixels", "input640_pixels"]:
        for size in ["small", "medium", "large"]:
            sub = [r for r in raw if r["frame"] == frame and r["size_group"] == size]
            counts = {int(r["gt_count"]) for r in sub}
            assert len(sub) == 6 and len(counts) == 1
            row = {"frame": frame, "size": size, "gt_count": counts.pop()}
            for metric in ["AP50_95", "AR50_95_at300"]:
                for mode in ["baseline", "fast"]:
                    values = [100 * float(r[metric]) for r in sub if r["mode"] == mode]
                    row[f"{mode}_{metric}"] = st.mean(values)
                    row[f"{mode}_{metric}_sd"] = st.stdev(values)
            sizes.append(row)
    write_csv(args.out / "size_summary.csv", sizes)
    inputs.append(args.inference)
    inf = json.loads(args.inference.read_text())
    efficiency = [
        dict(
            method=k,
            params=v["parameters_unfused"],
            fused_params=v["parameters_fused"],
            gflops=v["gflops"],
            pt_bytes=v["checkpoint_bytes"],
            pt_mib=v["checkpoint_bytes"] / 2**20,
            **v["summary"],
        )
        for k, v in inf["models"].items()
    ]
    write_csv(args.out / "efficiency.csv", efficiency)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

    def save(fig, name):
        fig.tight_layout()
        for ext in ["png", "pdf"]:
            fig.savefig(args.out / f"{name}.{ext}", dpi=180, bbox_inches="tight")
        plt.close(fig)

    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(7, 4))
    kept = np.array([r["kept"] / 1e6 for r in assignment])
    ax.bar(x, kept, label="Retained", color="#3274A1")
    ax.bar(x, [r["removed"] / 1e6 for r in assignment], bottom=kept, label="Removed", color="#E1812C")
    for i, r in enumerate(assignment):
        ax.text(i, r["original"] / 1e6 + 0.4, f"{r['removal_percent']:.2f}% removed", ha="center")
    ax.set(
        xticks=x,
        xticklabels=["P3", "P4", "P5"],
        ylabel="GT-anchor candidates (millions)",
        title="Hard-SRLA: seed0, 100 epochs",
        ylim=(0, 45),
    )
    ax.legend(loc="upper right")
    save(fig, "mechanism_assignment")
    original = [r for r in sizes if r["frame"] == "original_pixels"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric, title in zip(axes, ["AP50_95", "AR50_95_at300"], ["AP50:95 (%)", "AR50:95 @300 (%)"]):
        for offset, mode, label, color in [
            (-0.18, "baseline", "Baseline", "#3274A1"),
            (0.18, "fast", "Hard-SRLA", "#E1812C"),
        ]:
            ax.bar(
                x + offset,
                [r[f"{mode}_{metric}"] for r in original],
                0.36,
                yerr=[r[f"{mode}_{metric}_sd"] for r in original],
                capsize=3,
                label=label,
                color=color,
            )
        ax.set(xticks=x, xticklabels=[f"{r['size']}\nn={r['gt_count']:,}" for r in original], ylabel=title)
    axes[0].legend()
    fig.suptitle("Original-pixel sizes; mean +/- sample SD, 3 seeds")
    save(fig, "mechanism_sizes")
    fig, ax = plt.subplots(figsize=(7, 4))
    values = [r["map5095"] for r in ablation[:3]]
    ax.bar(x, values, color=["#3274A1", "#999999", "#E1812C"])
    for i, v in enumerate(values):
        ax.text(i, v + 0.15, f"{v:.3f}", ha="center")
    ax.set(
        xticks=x,
        xticklabels=order[:3],
        ylabel="mAP50:95 (%)",
        ylim=(0, 13),
        title="Seed0: matched sparsity, different selection",
    )
    save(fig, "mechanism_control")
    lines = [
        "# SRLA final results",
        "",
        "Converted VisDrone subset: train 1499 images; validation 300 images / 20328 GT. Not the official VisDrone evaluation protocol.",
        "Best epoch: max(0.1*mAP50 + 0.9*mAP50:95), earliest CSV tie. Rounded CSV cannot establish exact best.pt epoch on close ties.",
        "All +/- values are sample SD (ddof=1), not confidence intervals.",
        "",
        "## Main experiment (seeds 0/1/2)",
        "",
        "| Metric | Baseline | Hard-SRLA | Paired delta |",
        "|---|---:|---:|---:|",
    ]
    for r in summary:
        lines.append(
            f"| {r['metric']} | {r['baseline_mean']:.3f} +/- {r['baseline_sd']:.3f} | {r['hard_mean']:.3f} +/- {r['hard_sd']:.3f} | {r['paired_mean']:+.3f} +/- {r['paired_sd']:.3f} |"
        )
    lines += [
        "",
        "Accuracy in percent; accuracy deltas in percentage points. Training minutes include initialization, validation and saving.",
        f"Training ratio of mean wall times: {summary[-1]['hard_mean'] / summary[-1]['baseline_mean']:.3f}x.",
        "",
        "## Seed0 ablations",
        "",
        "| Method | mAP50:95 (%) | Best epoch | Purpose |",
        "|---|---:|---:|---|",
    ]
    for r, purpose in zip(
        ablation,
        ["Anchor-only", "Control for generic pruning", "Main spectral method", "Single fixed soft rule, rejected"],
    ):
        lines.append(f"| {r['method']} | {r['map5095']:.3f} | {r['best_epoch']} | {purpose} |")
    lines += [
        "",
        "Hard exceeds Control by +0.555 pp; Control exceeds Baseline by +0.177 pp. One Control seed supports spectral selection beyond generic pruning, but is not statistical proof.",
        "Control matches GT-anchor candidate sparsity before neighbor expansion, not size-specific counts or expanded positive cells.",
        "",
        "## Inference efficiency",
        "",
        inf["protocol"],
        "GPU: "
        + inf["gpu"]
        + ". Same 32 validation images, 50 warmups / 200 forwards per round, 5 alternating rounds; CPU threads=1.",
        "GFLOPs = THOP MACs x2. FPS = 1000 / mean batch=1 latency. PyTorch memory includes the resident 32-image tensor, model and caches; it is not total device usage.",
        "",
        "| Model | Params | GFLOPs | .pt MiB | Mean ms +/- round SD | Median ms | FPS | Peak allocated MiB | Peak reserved MiB | Forward increment MiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for e in efficiency:
        lines.append(
            f"| {e['method']} | {e['params']:,} | {e['gflops']:.3f} | {e['pt_mib']:.6f} | {e['mean_ms']:.3f} +/- {e['round_mean_std_ms']:.3f} | {e['median_ms']:.3f} | {e['fps']:.2f} | {e['peak_allocated_mib']:.2f} | {e['peak_reserved_mib']:.2f} | {e['forward_increment_mib']:.2f} |"
        )
    lines += [
        "",
        "Same inference architecture; no SRLA code in forward. Measured means differ by about 4.3%, but medians are nearly identical and round variation is substantial. Do not claim exact equal speed or acceleration.",
        "Initial uncontrolled-thread measurements remain under inference/. Final measurements use inference_threads1/. Both retained; no favorable timing selection.",
        "",
        "## Mechanism figures",
        "",
        "![Assignment](mechanism_assignment.png)",
        "",
        "Seed0 counts summed over 100 epochs before neighbor expansion, not unique positive cells.",
        "",
        "![Sizes](mechanism_sizes.png)",
        "",
        "COCO-style AP/AR, maxDet=300, on converted labels; distinct from YOLO training CSV mAP. Original-pixel bands use 32^2 and 96^2 area boundaries and pycocotools conventions.",
        "Original-pixel small/medium/large GT counts: 13135/6199/994. Large-object AR decreases; effects are mixed.",
        "Input640 uses actual loader scaling: 19013/1287/28 GT. Its large-bin +13.254 pp is supplementary with only 28 GT, not a main conclusion; see size_summary.csv.",
        "",
        "![Control](mechanism_control.png)",
        "",
        "## Provenance",
        "",
        "Hard tag: srla-hard-3seed-final (a2f51f4b). Matched Control: matched-control-seed0-final (b1f037e0). Soft: 6ed85509.",
        "Input hashes: input_sha256.json. Original diagnostics and slow reference remain in experiment archives. No new training or tuning.",
    ]
    (args.out / "README.md").write_text("\n".join(lines) + "\n")
    (args.out / "input_sha256.json").write_text(
        json.dumps({str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}, indent=2)
    )
    print(args.out / "README.md")


if __name__ == "__main__":
    main()
