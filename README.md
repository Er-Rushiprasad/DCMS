# Reactor Parameter Modeling and Data Cleaning

## How did we train our models?
- **Data cleaning:** Parsed the original CSV with dual headers (group and parameter rows), trimmed extra unnamed columns, and applied strict parameter range checks (PC 0–100; P1 200–300 m³/hr; P2/P3 105–135 °C; P4 90–110 °C; P5 as present). Out-of-range values were set to NaN, and all rows were preserved. The main script for this is `clean_data.py`, which outputs `New_PC_Without_Constraints_Cleaned_v9.csv`.
- **Model training:** For each group, a separate scikit-learn gradient boosting model was trained to predict P1, P2, P3, P4, and P5 from PC. Multi-output regression was used where needed. Training scripts include `train_model.py`, `train_pc_model.py`, and `train_pc_predictor.py`. Models are saved for later prediction.
- **Missing data:** No imputation or row dropping was performed; NaNs are left in place for invalid or missing values.

## What are the predicted output values of the parameters based on the input %conversion (PC)?
- For a given %conversion (PC), the trained models predict P1–P5 for each group. Predictions are group-specific and reflect the cleaned data and model training.
- **How to predict:**

```powershell
python predict_p1_to_p5_from_pc.py --pc 75
```

- This will output the predicted P1–P5 for all groups at PC=75. Other scripts (e.g., `predict_p1.py`) can be used for single-parameter predictions.
- **Output:** Results are printed or saved as CSVs, depending on the script. For authoritative predictions, use the latest models and cleaned data.

## Notes / Summary of work to date
- Parsed and cleaned the original data, removing extra columns and setting out-of-range values to NaN.
- Developed and iteratively improved the cleaning logic, especially for P5 and dtype handling.
- Trained per-group ML models (scikit-learn gradient boosting) to predict reactor parameters from PC.
- Created scripts for prediction, validation, and daily aggregation.
- Ensured all outputs preserve row counts and data integrity.
- No GEKKO or linear fits were used in the final workflow; all results are from scikit-learn models.

**Next steps (optional):**
- Add imputation or row-dropping for missing critical values.
- Organize model artifacts and add a CLI for batch predictions.

**Key scripts:**
- Data cleaning: `clean_data.py`
- Model training: `train_model.py`, `train_pc_model.py`, `train_pc_predictor.py`
- Prediction: `predict_p1_to_p5_from_pc.py`, `predict_p1.py`
- Helpers: `fill_p1_all_groups.py`, `fill_missing_grp28.py`

