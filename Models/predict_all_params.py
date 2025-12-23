import pickle
import numpy as np
import pandas as pd

MODEL_FILE = 'pc_to_params_model_20251223_154625.pkl'
SCALER_FILE = 'pc_to_params_scaler_20251223_154625.pkl'
METADATA_FILE = 'pc_to_params_metadata_20251223_154625.pkl'
PC_VALUE = 84.0

with open(MODEL_FILE, 'rb') as f:
    model = pickle.load(f)
with open(SCALER_FILE, 'rb') as f:
    scaler = pickle.load(f)
with open(METADATA_FILE, 'rb') as f:
    metadata = pickle.load(f)

num_pc_features = metadata['pc_feature_count']
input_pc = np.full((1, num_pc_features), PC_VALUE)
input_pc_scaled = scaler.transform(input_pc)
pred = model.predict(input_pc_scaled)[0]

output_columns = metadata['output_columns']
results = []
for (group, param, col_idx), value in zip(output_columns, pred):
    results.append({'Group': group, 'Parameter': param, 'PredictedValue': value})

results_df = pd.DataFrame(results)
results_df.to_csv('predictions_p1_to_p5_PC84_all_groups.csv', index=False)
print('Predictions saved to predictions_p1_to_p5_PC84_all_groups.csv')
