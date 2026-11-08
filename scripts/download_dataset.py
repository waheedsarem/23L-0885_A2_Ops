"""Download the assignment dataset, pinned to the version audited in Task 1.2."""
import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
os.environ["KAGGLEHUB_CACHE"] = str(PROJECT / "data" / "kaggle_cache")

import kagglehub

if __name__ == "__main__":
    print(kagglehub.dataset_download("cashbowman/ai-generated-images-vs-real-images/versions/1"))
