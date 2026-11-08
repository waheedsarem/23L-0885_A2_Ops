"""Display saved audit results in compact form for terminal evidence."""
import argparse
import getpass
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--section", choices=["counts", "statistics"], default="counts")
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
s = json.loads((root / "evidence/task_1_2/summary.json").read_text())
print("VisionGuard | 23L-0885_A2_Ops | Task 1.2")
print("User:", getpass.getuser(), "| Audit recorded:", s["recorded_at"], "(Asia/Karachi)")
print("Project:", root)
print("Source: Kaggle cashbowman/ai-generated-images-vs-real-images | version 1")
print("Folder labels: AiArtData -> AI-generated; RealArt -> real")
print("Saved results: evidence/task_1_2/summary.json")
print()
if args.section == "counts":
    print(f"{'CLASS':<16}{'FILES':>10}{'READABLE':>12}{'CORRUPT':>12}{'READ ERROR':>14}{'LIMIT':>10}")
    for row in s["class_counts"]:
        print(f"{row['class']:<16}{row['image_files']:>10,}{row['readable']:>12,}{row['corrupt_or_unreadable']:>12,}{row['file_read_errors']:>14,}{row['safety_limit']:>10,}")
    print()
    for key in ("total_files", "non_image_files", "duplicate_groups", "files_in_duplicate_groups", "duplicate_excess", "cross_class_duplicate_groups", "unmapped_images"):
        print(f"{key.replace('_', ' ').capitalize()}: {s[key]:,}")
    print()
    print("Exact duplicates = identical SHA-256 of complete file bytes.")
    print("Duplicate excess = copies beyond one per group. Source data was not changed.")
    print("Corruption check: Pillow verify(), reopen, and load() every frame.")
else:
    print("MIN / MEDIAN / MAX: READABLE IMAGES, DUPLICATES RETAINED")
    print(f"{'CLASS':<16}{'METRIC':<20}{'N':>7}{'MIN':>15}{'MEDIAN':>15}{'MAX':>15}")
    for row in s["numeric_summary"]:
        print(f"{row['class']:<16}{row['metric']:<20}{row['readable_images']:>7,}{row['min']:>15,}{row['median']:>15,}{row['max']:>15,}")
    print()
    print("Width/height: pixels in stored orientation. File sizes: bytes.")
    print("Every image row includes path, class, bytes, width, height, mode, format,")
    print("SHA-256, decode status, error text, and duplicate group.")
print()
print("Manifest SHA-256:", s["manifest_sha256"])
print("Audit elapsed seconds:", s["elapsed_seconds"])
