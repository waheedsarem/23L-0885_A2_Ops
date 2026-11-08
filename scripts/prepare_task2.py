"""Create leakage-safe splits and Task 2.1 augmented-batch evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

from task2_common import (DATA_ROOT, EVIDENCE_DIR, LABELS, SEED, SPLIT_DIR,
                          configure_reproducibility, make_dataset, save_json)


def manifest_hash(frame):
    text = "\n".join(f"{row.relative_path}\t{row.sha256}\t{row.label}" for row in frame.itertuples())
    return hashlib.sha256(text.encode()).hexdigest()


def main():
    configure_reproducibility()
    audit = pd.read_csv(EVIDENCE_DIR.parent / "task_1_2" / "image_audit.csv")
    audit = audit[audit["status"] == "ok"].copy().sort_values("relative_path")
    duplicate_exclusions = audit[audit.duplicated("sha256", keep="first")].copy()
    unique = audit.drop_duplicates("sha256", keep="first").copy()
    unique["label"] = unique["class"].map(LABELS)
    duplicate_exclusions["label"] = duplicate_exclusions["class"].map(LABELS)
    if unique["label"].isna().any():
        raise ValueError("Unknown class label")
    if not all((DATA_ROOT / path).is_file() for path in unique["relative_path"]):
        raise FileNotFoundError("Audited dataset copy is incomplete")

    train, remainder = train_test_split(
        unique, test_size=0.30, random_state=SEED, stratify=unique["label"]
    )
    validation, test = train_test_split(
        remainder, test_size=0.50, random_state=SEED, stratify=remainder["label"]
    )
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    columns = ["relative_path", "class", "label", "sha256"]
    summary = {"seed": SEED, "label_mapping": LABELS, "source_readable_images": int(len(audit)),
               "unique_images_used": int(len(unique)), "excluded_exact_duplicate_copies": int(len(duplicate_exclusions)),
               "splits": {}}
    for name, frame in (("train", train), ("validation", validation), ("test", test)):
        frame = frame.sort_values("relative_path").reset_index(drop=True)
        frame[columns].to_csv(SPLIT_DIR / f"{name}.csv", index=False)
        summary["splits"][name] = {
            "total": int(len(frame)),
            "AI-generated": int((frame["class"] == "AI-generated").sum()),
            "real": int((frame["class"] == "real").sum()),
            "manifest_sha256": manifest_hash(frame),
        }
    duplicate_exclusions[columns[:-1] + ["sha256"]].to_csv(SPLIT_DIR / "excluded_duplicates.csv", index=False)
    hashes = {name: set(pd.read_csv(SPLIT_DIR / f"{name}.csv")["sha256"])
              for name in ("train", "validation", "test")}
    summary["overlap_checks"] = {
        "train_validation": len(hashes["train"] & hashes["validation"]),
        "train_test": len(hashes["train"] & hashes["test"]),
        "validation_test": len(hashes["validation"] & hashes["test"]),
    }
    if any(summary["overlap_checks"].values()):
        raise RuntimeError("Content hash leaked across splits")
    save_json(EVIDENCE_DIR / "split_summary.json", summary)

    batch_images, batch_labels = next(iter(make_dataset("train", training=True, batch_size=12, augment=True)))
    fig, axes = plt.subplots(3, 4, figsize=(10, 8), constrained_layout=True)
    names = {0: "real", 1: "AI-generated"}
    for axis, image, label in zip(axes.flat, batch_images.numpy(), batch_labels.numpy()):
        axis.imshow(image.clip(0, 255).astype("uint8"))
        axis.set_title(names[int(label)], fontsize=10)
        axis.axis("off")
    fig.suptitle("Training-only augmented batch (160 x 160, aspect ratio preserved with padding)",
                 fontsize=14, fontweight="bold")
    fig.savefig(EVIDENCE_DIR / "augmented_batch.png", dpi=180, facecolor="white")
    plt.close(fig)
    print(json.dumps(summary, indent=2))
    print(f"Saved {EVIDENCE_DIR / 'augmented_batch.png'}")


if __name__ == "__main__":
    main()
