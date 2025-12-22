import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
from datetime import datetime

# CONFIG
input_file = 'single_group_reactor.csv'
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_file = f'C:\\DataCleaning\\Models\\pc_predictor_model_{timestamp}.pkl'

print("Loading data...")

# Read headers
with open(input_file, 'r', encoding='latin-1') as f:
    header_lines = [f.readline().strip(), f.readline().strip()]

data_df = pd.read_csv(input_file, skiprows=2, encoding='latin-1')

# Parse parameter names
params = [p.strip().upper() for p in header_lines[1].split(',')]

print(f"Columns found: {params}")

# Find indices
pc_idx = params.index('PC') if 'PC' in params else None
p_cols = ['P1', 'P2', 'P3', 'P4', 'P5']
output_idxs = [params.index(p) for p in p_cols if p in params]

if pc_idx is None:
    raise ValueError("PC column not found!")
if len(output_idxs) < 5:
    raise ValueError(f"Missing some P columns. Found: {[params[i] for i in output_idxs]}")

# Select only PC + P1 to P5
selected_cols = [pc_idx] + output_idxs
df = data_df.iloc[:, selected_cols].copy()
df.columns = ['PC', 'P1', 'P2', 'P3', 'P4', 'P5']

print(f"Initial rows: {len(df)}")

# Convert to numeric
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

print("\nFirst 10 rows after cleaning:")
print(df.head(10))

# Filter valid PC
df = df[df['PC'].notna()]
df = df[(df['PC'] >= 0) & (df['PC'] <= 100)]
print(f"After valid PC: {len(df)} rows")

# Temperature validity ranges (customized based on your data)
# P1: 150–200°C (melt temp), others lower
valid_ranges = {
    'P1': (180, 300),
    'P2': (90, 130),
    'P3': (90, 130),
    'P4': (80, 110),
    'P5': (60, 90)   # Clearly around 72–73°C
}

temp_cols = ['P1', 'P2', 'P3', 'P4', 'P5']
for col in temp_cols:
    min_t, max_t = valid_ranges[col]
    invalid = ~((df[col] >= min_t) & (df[col] <= max_t))
    df.loc[invalid, col] = np.nan
    print(f"{col}: {invalid.sum()} values marked NaN (outside {min_t}–{max_t}°C)")

# Keep rows with at least 4 valid temperatures (strict but safe)
before = len(df)
df['valid_count'] = df[temp_cols].count(axis=1)
df = df[df['valid_count'] >= 4]
df.drop(columns=['valid_count'], inplace=True)
print(f"After requiring ≥4 valid temps: {len(df)} rows (dropped {before - len(df)})")

# Impute remaining missing values with median
print("\nImputing missing values:")
for col in temp_cols:
    na_count = df[col].isna().sum()
    if na_count > 0:
        median = df[col].median()
        df[col].fillna(median, inplace=True)
        print(f"  {col}: imputed {na_count} → {median:.2f}°C")

final_rows = len(df)
print(f"\nFINAL training rows: {final_rows}")

if final_rows < 20:
    raise ValueError("Not enough clean data!")

# Train
X = df[['PC']].values
y = df[temp_cols].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

base = HistGradientBoostingRegressor(
    max_iter=1000,
    learning_rate=0.05,
    max_depth=10,
    min_samples_leaf=3,
    random_state=42
)
model = MultiOutputRegressor(base, n_jobs=-1)
model.fit(X_train, y_train)

train_r2 = model.score(X_train, y_train)
test_r2 = model.score(X_test, y_test)

print(f"\nTRAINING COMPLETE!")
print(f"Train R²: {train_r2:.4f}")
print(f"Test  R²: {test_r2:.4f}")

# Save
save_data = {
    'model': model,
    'scaler': scaler,
    'output_columns': temp_cols,
    'valid_ranges': valid_ranges,
    'note': 'Single reactor: predicts P1-P5 from PC (PF ignored)'
}

with open(model_file, 'wb') as f:
    pickle.dump(save_data, f)

print(f"\nModel saved: {model_file}")
print("   → Predicts: P1, P2, P3, P4, P5")