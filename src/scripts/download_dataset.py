"""
Download the Tatoeba sentences dataset into the local data directory.

This script fetches the sentences.csv file used for language identification
experiments and stores it under:

    data/sentences.csv

Usage (from repository root):
    python -m src.download_dataset
"""

from pathlib import Path
import urllib.request


DATA_URL = "https://downloads.tatoeba.org/exports/sentences.csv"
OUTPUT_FILE = "sentences.csv"


def main():
    # Resolve project root and data directory
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    output_path = data_dir / OUTPUT_FILE

    print(f"Downloading dataset from {DATA_URL}")
    print(f"Saving to {output_path}")

    # Download file
    urllib.request.urlretrieve(DATA_URL, output_path)

    print("Download complete.")


if __name__ == "__main__":
    main()
