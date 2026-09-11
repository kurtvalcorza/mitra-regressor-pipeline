"""
Build sample datasets for the Mitra Regressor pipeline
=====================================================
Builds the two cross-sectional companion sample datasets:
1. Insurance Medical Charges (Brett Lantz / OpenML) -> insurance-medical-charges.zip
2. Ames Housing (OpenML 42165)                     -> ames-housing.zip

The build is byte-reproducible: sources are pinned to immutable revisions and
digest-checked, ZIP members carry a fixed timestamp, and the written archives are
asserted against the digests recorded in the tutorial notebook's SAMPLE_CONFIGS.

Usage:
    python examples/build_sample_datasets.py
"""
from __future__ import annotations

import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split

SEED = 42
OUT_DIR = Path(__file__).resolve().parent / "sample-data"
NETWORK_TIMEOUT_SECONDS = 60

# Fixed member timestamp so identical CSV bytes always yield identical archives.
ARCHIVE_TIMESTAMP = (2026, 9, 7, 23, 2, 46)

# Insurance source: stedy/Machine-Learning-with-R-datasets, pinned to the last commit that touched insurance.csv.
INSURANCE_SOURCE_REVISION = "d20658ec6d336af2d4ddb5fd72b6f677dd46136e"
INSURANCE_SOURCE_URL = (
    "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/"
    f"{INSURANCE_SOURCE_REVISION}/insurance.csv"
)
INSURANCE_SOURCE_SHA256 = "505c1cbc2e63d0363bac59501563df2530aadf4cdb9cfee226f4ef32f5468281"
EXPECTED_INSURANCE_SHA256 = "7e16a06e13a7cb58fabb27ee53c1e20af83575c12ac97ea874fea45f013690a2"

# Ames source: OpenML dataset id 42165 ("house_prices" version 1); OpenML dataset versions are immutable.
AMES_OPENML_DATA_ID = 42165
EXPECTED_AMES_SHA256 = "14e084b13077f6b24e1b5f67c379f9695589f05dcce3f2e26bf96bbf72379dd3"


def fetch_verified(url: str, expected_sha256: str) -> bytes:
    with urllib.request.urlopen(url, timeout=NETWORK_TIMEOUT_SECONDS) as resp:
        payload = resp.read()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected_sha256:
        raise ValueError(f"Source digest mismatch for {url}: expected {expected_sha256}, got {digest}")
    return payload


def write_archive(out_zip: Path, parts: list[tuple[str, pd.DataFrame]], expected_sha256: str) -> Path:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, part in parts:
            info = zipfile.ZipInfo(name, ARCHIVE_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, part.to_csv(index=False))
    payload = buf.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected_sha256:
        raise ValueError(f"Built {out_zip.name} does not match the recorded digest: expected {expected_sha256}, got {digest}")
    out_zip.write_bytes(payload)
    print(f"  Wrote {out_zip} ({len(payload):,} bytes, SHA-256 {digest})")
    return out_zip


def build_insurance_charges() -> Path:
    print(f"Fetching Insurance Medical Charges (pinned revision {INSURANCE_SOURCE_REVISION[:12]})...")
    df = pd.read_csv(io.BytesIO(fetch_verified(INSURANCE_SOURCE_URL, INSURANCE_SOURCE_SHA256)))

    # Split 1,338 rows: 802 train (60%), 268 val (20%), 268 test (20%)
    train_df, temp_df = train_test_split(df, train_size=802, random_state=SEED)
    val_df, test_df = train_test_split(temp_df, train_size=268, random_state=SEED)

    return write_archive(
        OUT_DIR / "insurance-medical-charges.zip",
        [("train.csv", train_df), ("val.csv", val_df), ("test.csv", test_df)],
        EXPECTED_INSURANCE_SHA256,
    )


def build_ames_housing() -> Path:
    print(f"Fetching Ames Housing (OpenML data_id={AMES_OPENML_DATA_ID})...")
    ames_raw = sklearn.datasets.fetch_openml(data_id=AMES_OPENML_DATA_ID, as_frame=True, parser="auto")
    df = ames_raw.frame.copy()

    features = [
        # Numeric
        "LotArea", "YearBuilt", "YearRemodAdd", "1stFlrSF", "2ndFlrSF", "GrLivArea",
        "FullBath", "HalfBath", "BedroomAbvGr", "TotRmsAbvGrd", "Fireplaces",
        "GarageCars", "GarageArea", "WoodDeckSF", "OpenPorchSF",
        # Categorical / Ordinal
        "MSZoning", "Neighborhood", "Condition1", "BldgType", "HouseStyle",
        "OverallQual", "OverallCond", "ExterQual", "HeatingQC", "CentralAir",
        # Target
        "SalePrice"
    ]
    df_curated = df[features].copy()

    # Split 1,460 rows: 876 train (60%), 292 val (20%), 292 test (20%)
    train_df, temp_df = train_test_split(df_curated, train_size=876, random_state=SEED)
    val_df, test_df = train_test_split(temp_df, train_size=292, random_state=SEED)

    return write_archive(
        OUT_DIR / "ames-housing.zip",
        [("train.csv", train_df), ("val.csv", val_df), ("test.csv", test_df)],
        EXPECTED_AMES_SHA256,
    )


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    build_insurance_charges()
    build_ames_housing()
