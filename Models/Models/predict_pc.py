# predict_pc.py - Updated with correct units
import pickle
import numpy as np

# ------------------- CONFIG -------------------
PC_VALUE = 85.0          # Change this to any desired Percentage Conversion (%)

# Update this to your latest trained model file
model_file = 'C:\\DataCleaning\\Models\\Models\\pc_predictor_model_20251222_163752.pkl'
# ------------------------------------------------

with open(model_file, 'rb') as f:
    data = pickle.load(f)

model = data['model']
scaler = data['scaler']
output_columns = data['output_columns']  # ['P1', 'P2', 'P3', 'P4', 'P5']

# Known training data range (from your dataset)
TRAIN_PC_MIN = 78.0
TRAIN_PC_MAX = 80.0
TRAIN_PC_TYPICAL = 79.0

print(f"Model loaded: {model_file}")
print(f"Predicting process values for PC = {PC_VALUE}% (Percentage Conversion)\n")

if PC_VALUE < TRAIN_PC_MIN or PC_VALUE > TRAIN_PC_MAX:
    print("WARNING: Requested PC is outside historical range (78–80%)")
    print("   → Prediction is extrapolation and may not be accurate.\n")
else:
    print("✓ PC within historical operating range.\n")

# Prepare and predict
pc_array = np.array([[PC_VALUE]])
pc_scaled = scaler.transform(pc_array)
prediction = model.predict(pc_scaled)[0]

# Display results with correct units
print("PREDICTED VALUES")
print("-" * 50)
for col, val in zip(output_columns, prediction):
    if col == 'P1':
        print(f"{col} → {val:7.2f} m³/hr   (Flow rate)")
    else:
        print(f"{col} → {val:7.2f} °C      (Temperature)")
print("-" * 50)
print(f"Input: PC = {PC_VALUE}%")