# Climate Change Predictor - Data Preprocessing Pipeline

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


@dataclass
class ClimateDataPipeline:

    scaler: StandardScaler = StandardScaler()
    country_encoder: LabelEncoder = LabelEncoder()

    def load_raw(
        self,
        dataset1_path: Path,
        temperatures_path: Path,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df_yearly = pd.read_csv(dataset1_path)
        df_monthly = pd.read_csv(temperatures_path)
        return df_yearly, df_monthly

    def merge_datasets_yearly(
        self,
        yearly_df: pd.DataFrame,
        monthly_df: pd.DataFrame,
        year_col: str = "Year",
        country_col: str = "Country",
    ) -> pd.DataFrame:
        
        #Merge yearly and monthly datasets on year and country, averaging monthly values to yearly
        
        yearly = yearly_df.copy()
        monthly = monthly_df.copy()

        yearly = yearly.rename(columns={"year": year_col, "country": country_col})
        yearly[country_col] = yearly[country_col].astype(str).str.strip()
        yearly[year_col] = pd.to_numeric(yearly[year_col], errors="coerce")

        yearly_value_cols = [c for c in yearly.columns if c not in [year_col, country_col]]
        yearly = yearly.groupby([year_col, country_col], as_index=False)[
            yearly_value_cols
        ].mean(numeric_only=True)

        monthly["dt"] = pd.to_datetime(monthly["dt"], errors="coerce")
        monthly[year_col] = monthly["dt"].dt.year
        monthly[country_col] = monthly[country_col].astype(str).str.strip()
        monthly["AverageTemperature"] = pd.to_numeric(
            monthly["AverageTemperature"], errors="coerce"
        )
        monthly["AverageTemperatureUncertainty"] = pd.to_numeric(
            monthly["AverageTemperatureUncertainty"], errors="coerce"
        )

        monthly_yearly = monthly.groupby([year_col, country_col], as_index=False).agg(
            AverageTemperature=("AverageTemperature", "mean"),
            AverageTemperatureUncertainty=("AverageTemperatureUncertainty", "mean"),
        )

        merged = pd.merge(
            monthly_yearly, yearly, on=[year_col, country_col], how="inner"
        )
        return merged

    def handle_missing_values(
        self, df: pd.DataFrame, numeric_cols: Optional[Iterable[str]] = None
    ) -> pd.DataFrame:
        
        # Missing values in numeric columns are filled using linear interpolation. Categorical columns are left as is (could be handled differently if needed).
        out = df.copy()
        if numeric_cols is None:
            numeric_cols = out.select_dtypes(include=["number"]).columns.tolist()

        out[numeric_cols] = out[numeric_cols].interpolate(
            method="linear", limit_direction="both"
        )
        return out

    def temporal_split(
        self,
        df: pd.DataFrame,
        time_col: str,
        train_size: float = 0.70,
        val_size: float = 0.15,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        
        #Chronological split
        
        if time_col not in df.columns:
            raise ValueError(f"{time_col} column not found for temporal split")

        ordered = df.sort_values(time_col).reset_index(drop=True)
        n = len(ordered)
        train_end = int(n * train_size)
        val_end = int(n * (train_size + val_size))

        train = ordered.iloc[:train_end].reset_index(drop=True)
        val = ordered.iloc[train_end:val_end].reset_index(drop=True)
        test = ordered.iloc[val_end:].reset_index(drop=True)
        return train, val, test

    def scale_features(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_cols: Iterable[str],
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
       
        #Z-score scaling
        train = train_df.copy()
        val = val_df.copy()
        test = test_df.copy()

        cols = list(feature_cols)
        self.scaler.fit(train[cols])
        train[cols] = self.scaler.transform(train[cols])
        val[cols] = self.scaler.transform(val[cols])
        test[cols] = self.scaler.transform(test[cols])
        return train, val, test

    def fit_country_encoder(
        self, df: pd.DataFrame, country_col: str = "Country"
    ) -> pd.DataFrame:
       
        out = df.copy()
        out[country_col] = self.country_encoder.fit_transform(
            out[country_col].astype(str)
        )
        return out

    def encode_countries(
        self, df: pd.DataFrame, country_col: str = "Country"
    ) -> pd.DataFrame:
        
        #Transform country to integer labels using the fitted encoder
        out = df.copy()
        out[country_col] = self.country_encoder.transform(out[country_col].astype(str))
        return out

    def drop_leakage_columns(
        self, df: pd.DataFrame, cols: Optional[Iterable[str]] = None
    ) -> pd.DataFrame:
        
        #Drop columns that would leak future information (e.g. predicted_temperature_2050)
        if cols is None:
            cols = ["predicted_temperature_2050"]
        return df.drop(columns=[c for c in cols if c in df.columns])

    def prepare_for_training(
        self,
        df: pd.DataFrame,
        target_col: str = "AverageTemperature",
        country_col: str = "Country",
        time_col: str = "Year",
        train_size: float = 0.70,
        val_size: float = 0.15,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        
        df = self.drop_leakage_columns(df)
        df = self.handle_missing_values(df)

        train, val, test = self.temporal_split(
            df, time_col=time_col, train_size=train_size, val_size=val_size
        )

        train = self.fit_country_encoder(train, country_col=country_col)
        val = self.encode_countries(val, country_col=country_col)
        test = self.encode_countries(test, country_col=country_col)

        feature_cols = [
            c
            for c in train.columns
            if c not in [target_col, country_col, time_col]
        ]
        train, val, test = self.scale_features(
            train, val, test, feature_cols=feature_cols
        )

        return train, val, test
