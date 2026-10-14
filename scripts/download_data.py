"""Download the CWRU bearing data used in this project (drive end, 12 kHz, plus normal baseline).

Source: Case Western Reserve University Bearing Data Center, via a public GitHub mirror.
"""
import sys
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR
from src.data import RECORDS

MIRROR = "https://raw.githubusercontent.com/s-whynot/CWRU-dataset/main"


def main():
    for path in RECORDS["path"]:
        target = DATA_DIR / path
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(f"{MIRROR}/{urllib.parse.quote(path)}", target)
            print(f"downloaded {path}")
    print(f"{len(RECORDS)} files ready in {DATA_DIR}")


if __name__ == "__main__":
    main()
