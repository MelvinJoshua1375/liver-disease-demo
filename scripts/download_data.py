"""Download the liver disease dataset from Kaggle.

Usage:
    python scripts/download_data.py

Requires:
    - kaggle package: pip install kaggle
    - Kaggle API credentials in ~/.kaggle/kaggle.json
      (or KAGGLE_USERNAME + KAGGLE_KEY environment variables)

Dataset: https://www.kaggle.com/datasets/rabieelkharoua/predict-liver-disease-1700-records-dataset
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATASET_SLUG = "rabieelkharoua/predict-liver-disease-1700-records-dataset"
TARGET_DIR = Path(__file__).resolve().parent.parent / "data"
TARGET_FILE = TARGET_DIR / "liver_disease_data.csv"


def download_via_kagglehub() -> bool:
    """Download using kagglehub (no credentials file needed if already logged in)."""
    try:
        import kagglehub

        print("Downloading via kagglehub...")
        path = kagglehub.dataset_download(DATASET_SLUG)
        src = Path(path)
        csvs = list(src.glob("*.csv"))
        if not csvs:
            print(f"  No CSV found in {src}")
            return False
        TARGET_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(csvs[0], TARGET_FILE)
        print(f"  Saved: {TARGET_FILE}")
        return True
    except Exception as e:
        print(f"  kagglehub failed: {e}")
        return False


def download_via_kaggle_api() -> bool:
    """Download using the official kaggle CLI package."""
    try:
        import kaggle

        print("Downloading via kaggle API...")
        TARGET_DIR.mkdir(parents=True, exist_ok=True)
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            DATASET_SLUG,
            path=str(TARGET_DIR),
            unzip=True,
        )
        # Rename to canonical filename if needed
        for csv in TARGET_DIR.glob("*.csv"):
            if csv.name != TARGET_FILE.name:
                csv.rename(TARGET_FILE)
        if TARGET_FILE.exists():
            print(f"  Saved: {TARGET_FILE}")
            return True
        print("  CSV not found after download.")
        return False
    except Exception as e:
        print(f"  kaggle API failed: {e}")
        return False


def main():
    if TARGET_FILE.exists():
        print(f"Dataset already exists: {TARGET_FILE}")
        print(f"  Size: {TARGET_FILE.stat().st_size / 1024:.1f} KB")
        return

    print(f"Target path: {TARGET_FILE}")
    success = download_via_kagglehub() or download_via_kaggle_api()

    if success:
        print("\nDownload complete.")
        print(f"  File: {TARGET_FILE} ({TARGET_FILE.stat().st_size / 1024:.1f} KB)")
    else:
        print("\nAutomatic download failed.")
        print("Manual steps:")
        print(f"  1. Visit: https://www.kaggle.com/datasets/{DATASET_SLUG}")
        print("  2. Download and extract the CSV")
        print(f"  3. Place it at: {TARGET_FILE}")
        sys.exit(1)


if __name__ == "__main__":
    main()
