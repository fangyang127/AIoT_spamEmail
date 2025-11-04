import os
from pathlib import Path
import pandas as pd
import requests

RAW_URL = (
    "https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/"
    "master/Chapter03/datasets/sms_spam_no_header.csv"
)


def ensure_data_dir():
    Path("data").mkdir(parents=True, exist_ok=True)


def download_dataset(path: str = "data/sms_spam_no_header.csv") -> str:
    ensure_data_dir()
    if os.path.exists(path):
        return path
    print(f"Downloading dataset to {path}...")
    r = requests.get(RAW_URL, timeout=30)
    r.raise_for_status()
    with open(path, "wb") as f:
        f.write(r.content)
    print("Download complete.")
    return path


def load_data(path: str = "data/sms_spam_no_header.csv", download: bool = True) -> pd.DataFrame:
    if download and not os.path.exists(path):
        download_dataset(path)
    # CSV has no header; format is: label,message
    df = pd.read_csv(path, header=None, names=["label", "message"], encoding="utf-8", sep=",", quotechar='"')
    # Basic cleanup: drop NA and strip strings
    df = df.dropna().reset_index(drop=True)
    df["label"] = df["label"].astype(str).str.strip()
    df["message"] = df["message"].astype(str).str.strip()
    return df


if __name__ == "__main__":
    df = load_data()
    print(f"Loaded {len(df)} rows. Sample:")
    print(df.head())
