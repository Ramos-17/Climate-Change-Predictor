# Script to build the final dataset for modeling by merging and preprocessing the raw data files. Run as: python scripts/build_dataset.py
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_preprocessing import ClimateDataPipeline


def main() -> None:
    pipeline = ClimateDataPipeline()

    raw_dir = ROOT / "data" / "raw"
    processed_dir = ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    df_yearly, df_monthly = pipeline.load_raw(
        raw_dir / "Dataset1.csv",
        raw_dir / "GlobalLandTemperaturesByCountry.csv",
    )

    merged = pipeline.merge_datasets_yearly(df_yearly, df_monthly)
    merged = pipeline.drop_leakage_columns(merged)

    out_path = processed_dir / "model_dataset.csv"
    merged.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
