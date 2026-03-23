# Helper script to report dataset statistics and insights before preprocessing. Run as: python scripts/report_stats.py

from pathlib import Path

import pandas as pd


def file_size_mb(p: Path) -> float:
    return p.stat().st_size / (1024 * 1024)


def main() -> None:
    raw1 = Path("data/raw/Dataset1.csv")
    raw2 = Path("data/raw/GlobalLandTemperaturesByCountry.csv")

    df1 = pd.read_csv(raw1)
    df2 = pd.read_csv(raw2)

    print("== File sizes and shapes ==")
    print(f"Dataset1: rows={len(df1)}, cols={df1.shape[1]}, size_mb={file_size_mb(raw1):.2f}")
    print(
        f"GlobalLandTemperaturesByCountry: rows={len(df2)}, cols={df2.shape[1]}, size_mb={file_size_mb(raw2):.2f}"
    )

    # Prepare merge (yearly)
    df1 = df1.rename(columns={"year": "Year", "country": "Country"})
    df1["Country"] = df1["Country"].astype(str).str.strip()
    df1["Year"] = pd.to_numeric(df1["Year"], errors="coerce")

    df1_value_cols = [c for c in df1.columns if c not in ["Year", "Country"]]
    df1 = df1.groupby(["Year", "Country"], as_index=False)[df1_value_cols].mean(
        numeric_only=True
    )

    df2["dt"] = pd.to_datetime(df2["dt"], errors="coerce")
    df2["Year"] = df2["dt"].dt.year
    df2["Country"] = df2["Country"].astype(str).str.strip()
    df2["AverageTemperature"] = pd.to_numeric(df2["AverageTemperature"], errors="coerce")
    df2["AverageTemperatureUncertainty"] = pd.to_numeric(
        df2["AverageTemperatureUncertainty"], errors="coerce"
    )

    df2_yearly = df2.groupby(["Year", "Country"], as_index=False).agg(
        AverageTemperature=("AverageTemperature", "mean"),
        AverageTemperatureUncertainty=("AverageTemperatureUncertainty", "mean"),
    )

    merged = pd.merge(df2_yearly, df1, on=["Year", "Country"], how="inner")
    merged = merged.drop(columns=["predicted_temperature_2050"], errors="ignore")

    print("\n== Merged dataset shape ==")
    print(f"rows={len(merged)}, cols={merged.shape[1]}")
    print(f"year range: {int(merged['Year'].min())} .. {int(merged['Year'].max())}")

    print("\n== Columns ==")
    print(list(merged.columns))

    print("\n== Dtypes ==")
    for c, t in merged.dtypes.astype(str).items():
        print(f"{c}: {t}")

    missing_pct = merged.isna().mean().sort_values(ascending=False) * 100
    print("\n== Missing % (top 10) ==")
    print(missing_pct.head(10).round(2))

    numeric_cols = merged.select_dtypes(include=["number"]).columns.tolist()
    num_ranges = merged[numeric_cols].agg(["min", "max"]).T
    print("\n== Numeric ranges ==")
    for c in numeric_cols:
        mn = num_ranges.loc[c, "min"]
        mx = num_ranges.loc[c, "max"]
        print(f"{c}: {mn} .. {mx}")


if __name__ == "__main__":
    main()
