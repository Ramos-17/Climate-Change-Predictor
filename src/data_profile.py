from pathlib import Path

import pandas as pd


INPUT_CSV = Path("data/processed/model_dataset.csv")
OUTPUT_MD = Path("dataset_profile_draft.md")


def iqr_outlier_stats(df: pd.DataFrame, numeric_cols: list[str]) -> tuple[dict[str, int], dict[str, float]]:
    counts: dict[str, int] = {}
    pct: dict[str, float] = {}

    for col in numeric_cols:
        series = df[col].dropna()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            outlier_count = 0
        else:
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outlier_count = int(((series < lower) | (series > upper)).sum())

        counts[col] = outlier_count
        pct[col] = round((outlier_count / len(series)) * 100, 4) if len(series) else 0.0

    return counts, pct


def build_data_dictionary(df: pd.DataFrame) -> list[str]:
    # Descriptions are tailored to the known dataset schema.
    descriptions: dict[str, str] = {
        "Year": "Observation year (temporal index).",
        "Country": "Country name (spatial group key).",
        "AverageTemperature": "Mean country temperature; main prediction target.",
        "AverageTemperatureUncertainty": "Uncertainty of average temperature estimate.",
        "global_avg_temperature": "Global average temperature indicator.",
        "temperature_anomaly": "Temperature deviation from historical baseline.",
        "max_temperature": "Maximum observed temperature indicator.",
        "min_temperature": "Minimum observed temperature indicator.",
        "co2_concentration_ppm": "Atmospheric CO2 concentration in parts per million.",
        "annual_rainfall_mm": "Annual rainfall in millimeters.",
        "sea_level_rise_mm": "Sea level rise indicator in millimeters.",
        "sea_surface_temperature": "Sea surface temperature indicator.",
        "heatwave_days": "Annual heatwave days indicator.",
        "drought_index": "Drought severity index.",
        "flood_events_count": "Flood events count indicator.",
        "forest_cover_percent": "Forest cover as a percentage.",
        "deforestation_rate": "Deforestation rate indicator.",
        "fossil_fuel_consumption": "Fossil fuel consumption indicator.",
        "renewable_energy_share": "Renewable energy share percentage.",
        "air_quality_index": "Air quality index indicator.",
        "predicted_temperature_2050": "Projected temperature indicator for 2050.",
        "climate_risk_index": "Composite climate risk index.",
    }

    lines: list[str] = []
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    for idx, col in enumerate(df.columns, start=1):
        if col in numeric_cols:
            col_min = df[col].min()
            col_max = df[col].max()
            lines.append(
                f"{idx}. {col}: {descriptions.get(col, 'Feature description not provided.')} "
                f"Range [{col_min:.4f}, {col_max:.4f}]"
            )
        else:
            n_unique = df[col].nunique()
            lines.append(
                f"{idx}. {col}: {descriptions.get(col, 'Feature description not provided.')} "
                f"Unique values: {n_unique}"
            )

    return lines


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input dataset not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    file_size_bytes = INPUT_CSV.stat().st_size
    file_size_mb = round(file_size_bytes / (1024**2), 4)
    file_size_gb = round(file_size_bytes / (1024**3), 6)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    dtypes = {col: str(df[col].dtype) for col in df.columns}

    missing_pct = (df.isna().mean() * 100).round(4).to_dict()
    outlier_counts, outlier_pct = iqr_outlier_stats(df, numeric_cols)

    country_counts = df["Country"].value_counts().sort_values(ascending=False) if "Country" in df.columns else None
    year_counts = df["Year"].value_counts().sort_index() if "Year" in df.columns else None

    lines: list[str] = []
    lines.append("## 2.1 Dataset Profile")
    lines.append("")
    lines.append("### Data Size and Shape")
    lines.append(f"- Dataset file: {INPUT_CSV}")
    lines.append(f"- Total samples: {df.shape[0]}")
    lines.append(f"- Total columns: {df.shape[1]}")
    lines.append(f"- File size: {file_size_bytes} bytes ({file_size_mb} MB, {file_size_gb} GB)")

    if "Year" in df.columns:
        lines.append(f"- Year span: {int(df['Year'].min())} to {int(df['Year'].max())}")
    if "Country" in df.columns:
        lines.append(f"- Unique countries: {int(df['Country'].nunique())}")

    lines.append("")
    lines.append("### Data Types")
    for col, dtype in dtypes.items():
        lines.append(f"- {col}: {dtype}")

    lines.append("")
    lines.append("### Feature Descriptions (Data Dictionary)")
    lines.extend(build_data_dictionary(df))

    lines.append("")
    lines.append("### Missing Values and Anomalies")
    lines.append("- Missing values (% by feature):")
    for col, pct in missing_pct.items():
        lines.append(f"  - {col}: {pct}%")

    lines.append("- Outliers (IQR rule):")
    for col in numeric_cols:
        lines.append(f"  - {col}: {outlier_counts[col]} ({outlier_pct[col]}%)")

    lines.append("- Handling strategy:")
    lines.append("  - Current preprocessing drops rows with missing target/features.")
    lines.append("  - For future updates, use robust scaling and domain validation before removing outliers.")

    lines.append("")
    lines.append("### Class Distribution (if applicable)")
    lines.append("- This dataset is currently used for regression (continuous target), so class balance is not directly applicable.")

    if country_counts is not None:
        lines.append("- Sample balance by country:")
        for country, count in country_counts.items():
            lines.append(f"  - {country}: {int(count)}")
        min_count = int(country_counts.min())
        max_count = int(country_counts.max())
        ratio = round(max_count / min_count, 4) if min_count else 0
        lines.append(f"  - Min/Max ratio: {min_count}/{max_count} ({ratio})")

    if year_counts is not None:
        min_count = int(year_counts.min())
        max_count = int(year_counts.max())
        ratio = round(max_count / min_count, 4) if min_count else 0
        lines.append(f"- Year coverage min/max records: {min_count}/{max_count} ({ratio})")

    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved profile report: {OUTPUT_MD}")


if __name__ == "__main__":
    main()
