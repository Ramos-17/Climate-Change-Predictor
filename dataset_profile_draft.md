## 2.1 Dataset Profile

### Data Size and Shape
- Dataset file: data\processed\model_dataset.csv
- Total samples: 252
- Total columns: 22
- File size: 115731 bytes (0.1104 MB, 0.000108 GB)
- Year span: 1980 to 2013

### Data Types
- Year: int64
-  Country  : str
-  AverageTemperature : float64
-  AverageTemperatureUncertainty: float64
-  global_avg_temperature: float64
-  temperature_anomaly: float64
-  max_temperature   : float64
-  min_temperature     : float64
-  co2_concentration_ppm: float64
-  annual_rainfall_mm: float64
-  sea_level_rise_mm : float64
-  sea_surface_temperature: float64
-  heatwave_days     : float64
-  drought_index     : float64
-  flood_events_count : float64
-  forest_cover_percent: float64
-  deforestation_rate : float64
-  fossil_fuel_consumption: float64
-  renewable_energy_share: float64
-  air_quality_index  : float64
-  predicted_temperature_2050: float64
-  climate_risk_index: float64

### Feature Descriptions (Data Dictionary)
1. Year: Observation year (temporal index). Range [1980.0000, 2013.0000]
2.  Country  : Feature description not provided. Unique values: 8
3.  AverageTemperature : Feature description not provided. Range [-5.8752, 25.9587]
4.  AverageTemperatureUncertainty: Feature description not provided. Range [0.1358, 0.5337]
5.  global_avg_temperature: Feature description not provided. Range [12.7800, 16.1000]
6.  temperature_anomaly: Feature description not provided. Range [0.3100, 2.0167]
7.  max_temperature   : Feature description not provided. Range [18.6200, 47.0700]
8.  min_temperature     : Feature description not provided. Range [-3.7000, 13.2800]
9.  co2_concentration_ppm: Feature description not provided. Range [331.3100, 486.5300]
10.  annual_rainfall_mm: Feature description not provided. Range [396.1300, 1729.7900]
11.  sea_level_rise_mm : Feature description not provided. Range [0.5750, 6.4500]
12.  sea_surface_temperature: Feature description not provided. Range [11.3200, 20.8800]
13.  heatwave_days     : Feature description not provided. Range [0.0000, 58.0000]
14.  drought_index     : Feature description not provided. Range [0.1700, 4.9900]
15.  flood_events_count : Feature description not provided. Range [0.0000, 14.0000]
16.  forest_cover_percent: Feature description not provided. Range [12.1100, 69.3550]
17.  deforestation_rate : Feature description not provided. Range [0.2100, 3.3000]
18.  fossil_fuel_consumption: Feature description not provided. Range [20.4500, 89.5350]
19.  renewable_energy_share: Feature description not provided. Range [5.1900, 59.8500]
20.  air_quality_index  : Feature description not provided. Range [40.0000, 197.0000]
21.  predicted_temperature_2050: Feature description not provided. Range [1.0400, 3.2900]
22.  climate_risk_index: Feature description not provided. Range [3.4200, 97.0900]

### Missing Values and Anomalies
- Missing values (% by feature):
  - Year: 0.0%
  -  Country  : 0.0%
  -  AverageTemperature : 0.0%
  -  AverageTemperatureUncertainty: 0.0%
  -  global_avg_temperature: 0.0%
  -  temperature_anomaly: 0.0%
  -  max_temperature   : 0.0%
  -  min_temperature     : 0.0%
  -  co2_concentration_ppm: 0.0%
  -  annual_rainfall_mm: 0.0%
  -  sea_level_rise_mm : 0.0%
  -  sea_surface_temperature: 0.0%
  -  heatwave_days     : 0.0%
  -  drought_index     : 0.0%
  -  flood_events_count : 0.0%
  -  forest_cover_percent: 0.0%
  -  deforestation_rate : 0.0%
  -  fossil_fuel_consumption: 0.0%
  -  renewable_energy_share: 0.0%
  -  air_quality_index  : 0.0%
  -  predicted_temperature_2050: 0.0%
  -  climate_risk_index: 0.0%
- Outliers (IQR rule):
  - Year: 0 (0.0%)
  -  AverageTemperature : 0 (0.0%)
  -  AverageTemperatureUncertainty: 12 (4.7619%)
  -  global_avg_temperature: 3 (1.1905%)
  -  temperature_anomaly: 2 (0.7937%)
  -  max_temperature   : 3 (1.1905%)
  -  min_temperature     : 6 (2.381%)
  -  co2_concentration_ppm: 7 (2.7778%)
  -  annual_rainfall_mm: 6 (2.381%)
  -  sea_level_rise_mm : 7 (2.7778%)
  -  sea_surface_temperature: 11 (4.3651%)
  -  heatwave_days     : 0 (0.0%)
  -  drought_index     : 0 (0.0%)
  -  flood_events_count : 0 (0.0%)
  -  forest_cover_percent: 0 (0.0%)
  -  deforestation_rate : 0 (0.0%)
  -  fossil_fuel_consumption: 2 (0.7937%)
  -  renewable_energy_share: 9 (3.5714%)
  -  air_quality_index  : 0 (0.0%)
  -  predicted_temperature_2050: 5 (1.9841%)
  -  climate_risk_index: 0 (0.0%)
- Handling strategy:
  - Current preprocessing drops rows with missing target/features.
  - For future updates, use robust scaling and domain validation before removing outliers.

### Class Distribution (if applicable)
- This dataset is currently used for regression (continuous target), so class balance is not directly applicable.
- Year coverage min/max records: 6/8 (1.3333)