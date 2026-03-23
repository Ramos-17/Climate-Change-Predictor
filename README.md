The raw datasets are not included due to Github file size limits.

Download datasets from shared drive and place them in:
data/raw/

## Setup

1. Use Python 3.13+.
2. Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Requirements

This project uses the packages listed in requirements.txt:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Run Data Profiling

Generate the dataset profile report from the processed CSV:

```powershell
python src/data_profile.py
```

Output file:

- dataset_profile_draft.md

## Notes

- If data/processed/model_dataset.csv is missing, run your preprocessing workflow first.
- If your Python command is not recognized, try using py instead of python.
