"""
Build sample datasets for the Mitra Regressor pipeline
=====================================================
Builds the two cross-sectional companion sample datasets:
1. Insurance Medical Charges (Brett Lantz / OpenML) -> insurance-medical-charges.zip
2. Ames Housing (OpenML 42165)                     -> ames-housing.zip

Usage:
    python examples/build_sample_datasets.py
"""
from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split

SEED = 42
OUT_DIR = Path(__file__).resolve().parent / "sample-data"


def build_insurance_charges() -> Path:
    print("Fetching Insurance Medical Charges from public mirror...")
    url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
    with urllib.request.urlopen(url) as resp:
        df = pd.read_csv(io.BytesIO(resp.read()))

    # Split 1,338 rows: 802 train (60%), 268 val (20%), 268 test (20%)
    train_df, temp_df = train_test_split(df, train_size=802, random_state=SEED)
    val_df, test_df = train_test_split(temp_df, train_size=268, random_state=SEED)

    out_zip = OUT_DIR / "insurance-medical-charges.zip"
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("train.csv", train_df.to_csv(index=False))
        z.writestr("val.csv", val_df.to_csv(index=False))
        z.writestr("test.csv", test_df.to_csv(index=False))
    print(f"  Wrote {out_zip} ({out_zip.stat().st_size:,} bytes)")
    return out_zip


def build_ames_housing() -> Path:
    print("Fetching Ames Housing (OpenML 42165)...")
    ames_raw = sklearn.datasets.fetch_openml("house_prices", version=1, as_frame=True, parser="auto")
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

    out_zip = OUT_DIR / "ames-housing.zip"
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("train.csv", train_df.to_csv(index=False))
        z.writestr("val.csv", val_df.to_csv(index=False))
        z.writestr("test.csv", test_df.to_csv(index=False))
    print(f"  Wrote {out_zip} ({out_zip.stat().st_size:,} bytes)")
    return out_zip


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    build_insurance_charges()
    build_ames_housing()
