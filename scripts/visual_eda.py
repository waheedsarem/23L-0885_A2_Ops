"""Task 1.3: reproducible sample grid and per-class image histograms."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

CLASS_ORDER = ["AI-generated", "real"]
COLORS = {"AI-generated": "#D1495B", "real": "#247BA0"}


def load_inputs(audit_csv: Path, summary_json: Path):
    frame = pd.read_csv(audit_csv)
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    required = {"relative_path", "class", "width", "height", "file_size_bytes", "sha256", "status"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Audit CSV is missing columns: {sorted(missing)}")
    frame = frame[frame["status"] == "ok"].copy()
    if set(frame["class"]) != set(CLASS_ORDER):
        raise ValueError(f"Expected classes {CLASS_ORDER}; found {sorted(frame['class'].unique())}")
    root = Path(summary["dataset_root"])
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    return frame, summary, root


def select_samples(frame: pd.DataFrame, per_class: int, seed: int):
    chosen = []
    for index, label in enumerate(CLASS_ORDER):
        # Exact duplicate copies are excluded from sampling so each visible example is unique.
        candidates = frame[frame["class"] == label].drop_duplicates("sha256").sort_values("relative_path")
        if len(candidates) < per_class:
            raise ValueError(f"{label} has fewer than {per_class} unique readable images")
        rng = random.Random(seed + index)
        positions = sorted(rng.sample(range(len(candidates)), per_class))
        class_samples = candidates.iloc[positions].copy()
        class_samples.insert(0, "sample_position", range(1, per_class + 1))
        chosen.append(class_samples)
    return pd.concat(chosen, ignore_index=True)


def display_image(path: Path):
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def make_sample_grid(samples: pd.DataFrame, root: Path, output: Path, per_class: int, seed: int):
    columns = 4
    rows_per_class = int(np.ceil(per_class / columns))
    fig, axes = plt.subplots(rows_per_class * 2, columns, figsize=(11.5, 11.5), constrained_layout=True)
    for row_index, label in enumerate(CLASS_ORDER):
        subset = samples[samples["class"] == label].sort_values("sample_position")
        for sample_index, record in enumerate(subset.itertuples(index=False)):
            plot_row = row_index * rows_per_class + sample_index // columns
            column_index = sample_index % columns
            axis = axes[plot_row, column_index]
            axis.imshow(display_image(root / record.relative_path))
            axis.set_title(f"{record.width} x {record.height}", fontsize=9)
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_color(COLORS[label])
                spine.set_linewidth(2)
        center_row = row_index * rows_per_class + (rows_per_class - 1) / 2
        axes[int(center_row), 0].set_ylabel(label, fontsize=12, fontweight="bold", color=COLORS[label], labelpad=10)
    fig.suptitle(f"Reproducible random image samples by source label (seed={seed})", fontsize=15, fontweight="bold")
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def format_bytes(value, _=None):
    if value >= 1024**2:
        return f"{value / 1024**2:g} MiB"
    return f"{value / 1024:g} KiB"


def log_edges(values, bins=25):
    return np.geomspace(float(values.min()), float(values.max()), bins + 1)


def make_histograms(frame: pd.DataFrame, output: Path):
    specs = [
        ("width", "Width (pixels)", None),
        ("height", "Height (pixels)", None),
        ("file_size_bytes", "File size", FuncFormatter(format_bytes)),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.4), constrained_layout=True, sharey="col")
    for row_index, label in enumerate(CLASS_ORDER):
        subset = frame[frame["class"] == label]
        for column_index, (metric, xlabel, formatter) in enumerate(specs):
            axis = axes[row_index, column_index]
            # Shared bin edges make class panels directly comparable.
            edges = log_edges(frame[metric])
            axis.hist(subset[metric], bins=edges, color=COLORS[label], edgecolor="white", linewidth=0.45, alpha=0.9)
            median = float(subset[metric].median())
            axis.axvline(median, color="#202020", linestyle="--", linewidth=1.5,
                         label=f"Median: {median:,.0f}" if metric != "file_size_bytes" else f"Median: {median / 1024:,.1f} KiB")
            axis.set_xscale("log")
            if metric in ("width", "height"):
                ticks = [200, 500, 1000, 2000, 5000, 10000]
                axis.set_xticks([tick for tick in ticks if frame[metric].min() <= tick <= frame[metric].max()])
                axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
            else:
                ticks = [10 * 1024, 100 * 1024, 1024**2, 10 * 1024**2]
                axis.set_xticks([tick for tick in ticks if frame[metric].min() <= tick <= frame[metric].max()])
                axis.xaxis.set_major_formatter(FuncFormatter(format_bytes))
            axis.xaxis.set_minor_formatter(NullFormatter())
            axis.grid(axis="y", alpha=0.25)
            axis.set_xlabel(xlabel)
            if column_index == 0:
                axis.set_ylabel(f"{label}\nImage count", color=COLORS[label], fontweight="bold")
            axis.legend(loc="upper right", fontsize=9, frameon=True)
            axis.set_title(f"{label}: {xlabel}", fontsize=11)
    fig.suptitle("Image dimensions and file-size distributions by source label", fontsize=15, fontweight="bold")
    fig.text(0.5, -0.015, "Logarithmic x-axes and shared bin edges reveal both the main concentration and large-image tails.",
             ha="center", fontsize=10)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_sample_manifest(samples: pd.DataFrame, output: Path):
    fields = ["sample_position", "class", "relative_path", "width", "height", "file_size_bytes", "sha256"]
    samples[fields].to_csv(output, index=False, quoting=csv.QUOTE_MINIMAL)


def write_eda_summary(frame: pd.DataFrame, seed: int, per_class: int, output: Path):
    records = {}
    for label in CLASS_ORDER:
        subset = frame[frame["class"] == label]
        records[label] = {
            "count": int(len(subset)),
            "width_q1": float(subset.width.quantile(.25)), "width_median": float(subset.width.median()), "width_q3": float(subset.width.quantile(.75)),
            "height_q1": float(subset.height.quantile(.25)), "height_median": float(subset.height.median()), "height_q3": float(subset.height.quantile(.75)),
            "size_q1_bytes": float(subset.file_size_bytes.quantile(.25)), "size_median_bytes": float(subset.file_size_bytes.median()), "size_q3_bytes": float(subset.file_size_bytes.quantile(.75)),
            "width_over_3000": int((subset.width > 3000).sum()),
            "height_over_2500": int((subset.height > 2500).sum()),
            "size_over_2mib": int((subset.file_size_bytes > 2 * 1024**2).sum()),
        }
    output.write_text(json.dumps({"seed": seed, "samples_per_class": per_class, "classes": records}, indent=2), encoding="utf-8")


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-csv", type=Path, default=project / "evidence/task_1_2/image_audit.csv")
    parser.add_argument("--summary-json", type=Path, default=project / "evidence/task_1_2/summary.json")
    parser.add_argument("--output", type=Path, default=project / "evidence/task_1_3")
    parser.add_argument("--seed", type=int, default=230885)
    parser.add_argument("--samples-per-class", type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frame, _, root = load_inputs(args.audit_csv, args.summary_json)
    samples = select_samples(frame, args.samples_per_class, args.seed)
    make_sample_grid(samples, root, args.output / "sample_grid.png", args.samples_per_class, args.seed)
    make_histograms(frame, args.output / "histograms.png")
    write_sample_manifest(samples, args.output / "sample_manifest.csv")
    write_eda_summary(frame, args.seed, args.samples_per_class, args.output / "eda_summary.json")
    print(f"Task 1.3 complete: {len(samples)} samples, {len(frame)} images in histograms")
    print(f"Saved: {args.output / 'sample_grid.png'}")
    print(f"Saved: {args.output / 'histograms.png'}")


if __name__ == "__main__":
    main()
