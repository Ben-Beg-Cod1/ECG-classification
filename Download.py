"""
download.py
-----------
Download ECG datasets:
  - MIT-BIH Arrhythmia Database (beat-level classificatie)
  - PhysioNet CinC 2017 (ritme-niveau: N / AF / Other / Noisy)
 
Gebruik:
  python data/download.py --dataset mitbih
  python data/download.py --dataset cinc2017
  python data/download.py --dataset all
"""
 
import os
import argparse
import urllib.request
import zipfile
from pathlib import Path
 
RAW_DIR = Path(__file__).parent / "raw"
 
# ------------------------------------------------------------------ MIT-BIH --
MITBIH_BASE = "https://physionet.org/files/mitdb/1.0.0/"
MITBIH_RECORDS = [
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119", "121",
    "122", "123", "124", "200", "201", "202", "203", "205", "207", "208",
    "209", "210", "212", "213", "214", "215", "217", "219", "220", "221",
    "222", "223", "228", "230", "231", "232", "233", "234",
]
MITBIH_EXTENSIONS = [".dat", ".hea", ".atr"]
 
 
def download_mitbih():
    dest = RAW_DIR / "mitbih"
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[MIT-BIH] Downloaden naar {dest} ...")
 
    for record in MITBIH_RECORDS:
        for ext in MITBIH_EXTENSIONS:
            filename = record + ext
            url = MITBIH_BASE + filename
            target = dest / filename
            if target.exists():
                print(f"  [skip] {filename}")
                continue
            print(f"  {filename} ...", end=" ", flush=True)
            try:
                urllib.request.urlretrieve(url, target)
                print("ok")
            except Exception as e:
                print(f"FOUT: {e}")
 
    print("[MIT-BIH] Klaar.\n")
 
 
# --------------------------------------------------------------- CinC 2017 --
CINC2017_URL = (
    "https://physionet.org/files/challenge-2017/1.0.0/training2017.zip"
)
CINC2017_LABELS_URL = (
    "https://physionet.org/files/challenge-2017/1.0.0/REFERENCE-v3.csv"
)
 
 
def download_cinc2017():
    dest = RAW_DIR / "cinc2017"
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / "training2017.zip"
 
    print(f"[CinC 2017] Downloaden naar {dest} ...")
 
    if not zip_path.exists():
        print("  training2017.zip ...", end=" ", flush=True)
        urllib.request.urlretrieve(CINC2017_URL, zip_path)
        print("ok")
    else:
        print("  [skip] training2017.zip")
 
    extract_dir = dest / "training2017"
    if not extract_dir.exists():
        print("  Uitpakken ...", end=" ", flush=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest)
        print("ok")
    else:
        print("  [skip] al uitgepakt")
 
    labels_path = dest / "REFERENCE-v3.csv"
    if not labels_path.exists():
        print("  REFERENCE-v3.csv ...", end=" ", flush=True)
        urllib.request.urlretrieve(CINC2017_LABELS_URL, labels_path)
        print("ok")
    else:
        print("  [skip] REFERENCE-v3.csv")
 
    print("[CinC 2017] Klaar.\n")
 
 
# -------------------------------------------------------------------  main --
def main():
    parser = argparse.ArgumentParser(description="ECG dataset downloader")
    parser.add_argument(
        "--dataset",
        choices=["mitbih", "cinc2017", "all"],
        default="all",
        help="Welke dataset downloaden (default: all)",
    )
    args = parser.parse_args()
 
    RAW_DIR.mkdir(parents=True, exist_ok=True)
 
    if args.dataset in ("mitbih", "all"):
        download_mitbih()
    if args.dataset in ("cinc2017", "all"):
        download_cinc2017()
 
 
if __name__ == "__main__":
    main()
