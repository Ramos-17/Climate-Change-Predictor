## 2.1 Dataset Profile

### Data Size and Shape
- Dataset file: data/processed/model_dataset.csv
- Total samples: 252
- Total columns: 22
- File size: 57,308 bytes (0.0547 MB, 0.000053 GB)
- Data structure: tabular panel time-series (Country x Year records)
- Year span: 1980 to 2013
- Unique countries: 8 (Australia, Brazil, Canada, China, France, Germany, India, Pakistan)

### Data Types
- Temporal discrete feature:
  - Year (int64)
- Categorical nominal feature:
  - Country (string)
- Numerical continuous features (float64):
  - AverageTemperature
  - AverageTemperatureUncertainty
  - global_avg_temperature
  - temperature_anomaly
  - max_temperature
  - min_temperature
  - co2_concentration_ppm
  - annual_rainfall_mm
  - sea_level_rise_mm
  - sea_surface_temperature
  - heatwave_days (aggregated count-like, stored as continuous)
  - drought_index
  - flood_events_count (aggregated count-like, stored as continuous)
  - forest_cover_percent
  - deforestation_rate
  - fossil_fuel_consumption
  - renewable_energy_share
  - air_quality_index
  - predicted_temperature_2050
  - climate_risk_index

### Feature Descriptions (Data Dictionary)
1. Year: observation year, 1980-2013, unit year, key temporal index.
2. Country: country name, 8 categories, unit none, key spatial/group index.
3. AverageTemperature: mean country temperature, -5.875 to 25.959, likely degC, main prediction target.
4. AverageTemperatureUncertainty: uncertainty of temperature estimate, 0.136 to 0.534, likely degC, measures target reliability.
5. global_avg_temperature: global mean temperature indicator, 12.780 to 16.100, likely degC, global climate context.
6. temperature_anomaly: deviation from baseline temperature, 0.310 to 2.017, likely degC, warming signal.
7. max_temperature: maximum temperature indicator, 18.620 to 47.070, likely degC, extreme heat context.
8. min_temperature: minimum temperature indicator, -3.700 to 13.280, likely degC, cold-end variability.
9. co2_concentration_ppm: atmospheric CO2 concentration, 331.310 to 486.530, ppm, greenhouse gas driver.
10. annual_rainfall_mm: annual precipitation, 396.130 to 1729.790, mm, hydroclimate driver.
11. sea_level_rise_mm: sea level change indicator, 0.575 to 6.450, mm, climate impact signal.
12. sea_surface_temperature: sea surface temperature, 11.320 to 20.880, likely degC, ocean-atmosphere coupling.
13. heatwave_days: annual heatwave day indicator, 0.000 to 58.000, days/aggregated, extreme heat intensity.
14. drought_index: drought severity indicator, 0.170 to 4.990, index, moisture stress signal.
15. flood_events_count: flood event indicator, 0.000 to 14.000, events/aggregated, extreme precipitation impact.
16. forest_cover_percent: forest cover share, 12.110 to 69.355, percent, land-use and carbon sink proxy.
17. deforestation_rate: forest loss rate, 0.210 to 3.300, likely percent/year, anthropogenic land-use pressure.
18. fossil_fuel_consumption: fossil energy use index, 20.450 to 89.535, unit unspecified/index-like, emission driver.
19. renewable_energy_share: renewable energy proportion, 5.190 to 59.850, percent, mitigation indicator.
20. air_quality_index: air pollution index, 40.000 to 197.000, AQI/index, environmental stress proxy.
21. predicted_temperature_2050: projected future temperature indicator, 1.040 to 3.290, likely degC projection/index, forward-looking signal.
22. climate_risk_index: climate risk score, 3.420 to 97.090, index, vulnerability/impact proxy.

### Missing Values and Anomalies
- Missing values:
  - All columns have 0.0% missing in the current processed dataset.
  - Reason: preprocessing already removed rows with missing target/features before export.
- Current handling strategy in pipeline:
  - Drop rows with missing values in target and selected features.
- Recommended strategy for future robustness:
  - Keep current drop strategy for small missingness.
  - If missingness increases, use country-wise median/temporal interpolation for numeric features.
  - Preserve missingness flags for high-signal fields if needed.
- Outliers (IQR rule) detected in several numeric columns:
  - AverageTemperatureUncertainty: 12 (4.7619%)
  - sea_surface_temperature: 11 (4.3651%)
  - renewable_energy_share: 9 (3.5714%)
  - co2_concentration_ppm: 7 (2.7778%)
  - sea_level_rise_mm: 7 (2.7778%)
  - annual_rainfall_mm: 6 (2.3810%)
  - min_temperature: 6 (2.3810%)
  - predicted_temperature_2050: 5 (1.9841%)
- Planned outlier handling:
  - Validate domain plausibility first (do not auto-drop climate extremes).
  - Use robust scaling or winsorization for high-leverage models.
  - Consider log transform for skewed positive features such as rainfall and consumption proxies.

### Class Distribution (if applicable)
- This is currently a regression setup (target is continuous AverageTemperature), so class balance does not directly apply.
- Balance check of sample coverage:
  - Country counts range from 30 to 34 records (max/min ratio 1.13, fairly balanced).
  - Year counts range from 6 to 8 records (max/min ratio 1.33, moderately even).
- If later reframed as classification:
  - Use stratified splitting.
  - Use class weights and/or over/under-sampling.
  - Consider focal loss only for strongly imbalanced multi-class setups.
