"""Task 1.2: read-only image metadata, full decoding, and SHA-256 audit.

Example:
  python scripts/audit_dataset.py --data-root data/raw --class-folder AiArtData=AI-generated --class-folder RealArt=real
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone, timedelta
import getpass
import hashlib
import json
from pathlib import Path
import statistics
import time
import warnings

from PIL import Image, UnidentifiedImageError

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".jfif", ".avif", ".heic", ".heif", ".ico", ".ppm", ".pgm", ".pbm"}
FIELDS = ["relative_path", "class", "file_size_bytes", "width", "height", "colour_mode", "format", "frames", "sha256", "status", "error", "warnings", "duplicate_group", "cross_class_duplicate"]


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_file(path, root, class_folders):
    relative = path.relative_to(root)
    labels = {class_folders[part] for part in relative.parts[:-1] if part in class_folders}
    row = dict.fromkeys(FIELDS, "")
    row.update(relative_path=relative.as_posix(), **{"class": next(iter(labels)) if len(labels) == 1 else "UNMAPPED"}, status="ok")
    known_extension = path.suffix.lower() in IMAGE_EXTENSIONS
    try:
        row["file_size_bytes"] = path.stat().st_size
        row["sha256"] = sha256_file(path)
    except OSError as exc:
        row.update(status="file_read_error", error=f"{type(exc).__name__}: {exc}")
        return row
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            # verify checks structural integrity; reopening and load force pixel decoding.
            with Image.open(path) as image:
                row.update(width=image.width, height=image.height, colour_mode=image.mode,
                           format=image.format, frames=getattr(image, "n_frames", 1))
                image.verify()
            with Image.open(path) as image:
                for frame in range(getattr(image, "n_frames", 1)):
                    image.seek(frame)
                    image.load()
            row["warnings"] = " | ".join(sorted({str(item.message) for item in caught}))
    except Image.DecompressionBombError as exc:
        row.update(status="safety_limit", error=f"{type(exc).__name__}: {exc}")
    except UnidentifiedImageError as exc:
        row.update(status="corrupt_or_unreadable" if known_extension else "non_image",
                   error=f"{type(exc).__name__}: {exc}")
    except (OSError, ValueError, SyntaxError, EOFError) as exc:
        row.update(status="corrupt_or_unreadable", error=f"{type(exc).__name__}: {exc}")
    if row["status"] != "non_image" and len(labels) != 1:
        row["error"] = (row["error"] + " | Class folder mapping missing or ambiguous").strip(" |")
    return row


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        if row["sha256"]:
            groups[row["sha256"]].append(row)
    duplicates = [group for group in groups.values() if len(group) > 1]
    for index, group in enumerate(duplicates, 1):
        cross_class = len({row["class"] for row in group}) > 1
        for row in group:
            row["duplicate_group"] = f"D{index:05d}"
            row["cross_class_duplicate"] = cross_class
    counts, numeric = [], []
    for label in sorted({r["class"] for r in rows}) + ["ALL"]:
        subset = rows if label == "ALL" else [r for r in rows if r["class"] == label]
        valid = [r for r in subset if r["status"] == "ok"]
        hashes = Counter(r["sha256"] for r in subset if r["sha256"])
        counts.append({"class": label, "image_files": len(subset), "readable": len(valid),
                       "corrupt_or_unreadable": sum(r["status"] == "corrupt_or_unreadable" for r in subset),
                       "file_read_errors": sum(r["status"] == "file_read_error" for r in subset),
                       "safety_limit": sum(r["status"] == "safety_limit" for r in subset),
                       "duplicate_groups_within_class": sum(n > 1 for n in hashes.values()),
                       "duplicate_excess_within_class": sum(n - 1 for n in hashes.values()),
                       "files_in_global_duplicate_groups": sum(bool(r["duplicate_group"]) for r in subset)})
        for metric in ("width", "height", "file_size_bytes"):
            values = [r[metric] for r in valid]
            numeric.append({"class": label, "metric": metric, "readable_images": len(values),
                            "min": min(values) if values else "", "median": statistics.median(values) if values else "",
                            "max": max(values) if values else ""})
    return counts, numeric, duplicates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--class-folder", action="append", required=True, metavar="FOLDER=LABEL")
    parser.add_argument("--output", type=Path, default=Path("evidence/task_1_2"))
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    root, output = args.data_root.resolve(), args.output.resolve()
    if not root.is_dir():
        parser.error(f"Dataset directory does not exist: {root}")
    if output == root or root in output.parents:
        parser.error("Output must be outside the dataset directory")
    mapping = {}
    for entry in args.class_folder:
        folder, sep, label = entry.partition("=")
        if not sep or not folder or not label or folder in mapping:
            parser.error("Supply unique class folder mappings as FOLDER=LABEL")
        mapping[folder] = label
    files = sorted((p for p in root.rglob("*") if p.is_file()), key=lambda p: p.relative_to(root).as_posix())
    if not files:
        parser.error("Dataset contains no files")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    print(f"23L-0885_A2_Ops | Task 1.2 | Auditing {len(files):,} files", flush=True)
    print(f"Dataset: {root}", flush=True)
    records = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for index, row in enumerate(pool.map(lambda p: inspect_file(p, root, mapping), files), 1):
            records.append(row)
            if index % 500 == 0 or index == len(files):
                print(f"Audited {index:,}/{len(files):,} files", flush=True)
    images = [r for r in records if r["status"] != "non_image"]
    skipped = [r for r in records if r["status"] == "non_image"]
    if not images:
        raise RuntimeError("No candidate images found; check dataset root")
    counts, numeric, duplicate_groups = summarize(images)
    write_csv(output / "image_audit.csv", images, FIELDS)
    write_csv(output / "non_image_files.csv", skipped, FIELDS)
    write_csv(output / "class_counts.csv", counts, list(counts[0]))
    write_csv(output / "numeric_summary.csv", numeric, list(numeric[0]))
    write_csv(output / "unreadable_files.csv", [r for r in images if r["status"] != "ok"], FIELDS)
    write_csv(output / "exact_duplicates.csv", [r for r in images if r["duplicate_group"]], FIELDS)
    distributions = []
    for field in ("format", "colour_mode"):
        for (label, value), count in sorted(Counter((r["class"], r[field]) for r in images if r["status"] == "ok").items()):
            distributions.append({"class": label, "attribute": field, "value": value, "count": count})
    write_csv(output / "format_mode_counts.csv", distributions, ["class", "attribute", "value", "count"])
    manifest = "\n".join(f"{r['relative_path']}\t{r['sha256']}" for r in records)
    summary = {"project": "23L-0885_A2_Ops", "recorded_at": datetime.now(timezone(timedelta(hours=5))).isoformat(timespec="seconds"),
               "user": getpass.getuser(), "dataset_root": str(root), "class_folder_mapping": mapping,
               "total_files": len(records), "image_files": len(images), "non_image_files": len(skipped),
               "status_counts": dict(Counter(r["status"] for r in images)), "class_counts": counts,
               "numeric_summary": numeric, "duplicate_groups": len(duplicate_groups),
               "files_in_duplicate_groups": sum(len(g) for g in duplicate_groups),
               "duplicate_excess": sum(len(g)-1 for g in duplicate_groups),
               "cross_class_duplicate_groups": sum(len({r['class'] for r in g}) > 1 for g in duplicate_groups),
               "unmapped_images": sum(r["class"] == "UNMAPPED" for r in images),
               "manifest_sha256": hashlib.sha256(manifest.encode()).hexdigest(),
               "elapsed_seconds": round(time.perf_counter()-started, 2),
               "statistics_population": "Readable images only; duplicates retained; dimensions in stored orientation; sizes in bytes",
               "duplicate_definition": "Identical full-file SHA-256; excess counts all copies beyond one per group"}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if summary["unmapped_images"]:
        raise RuntimeError("Unmapped image classes found; fix class-folder mappings before using results")


if __name__ == "__main__":
    main()
