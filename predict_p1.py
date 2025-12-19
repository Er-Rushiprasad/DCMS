import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Load data (prefer filled P3 if available)
try:
    df = pd.read_csv('grp28_data_filled.csv')
except FileNotFoundError:
    df = pd.read_csv('grp28_data.csv', skiprows=1, header=0)
    df.columns = ['P1','P2','P3','P4','P5','PF','PC']

# Ensure numeric
for col in ['P1','P2','P3','P4','P5','PC']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

features = ['P2','P3','P4','P5','PC']

# Rows available for training: P1 present and all selected features present
train_df = df[df['P1'].notna()].dropna(subset=features+['P1']).copy()
print(f"Training rows: {len(train_df)}")
if len(train_df) < 20:
    print("Warning: very few training samples.")

X = train_df[features].values
y = train_df['P1'].values

models = {
    'LinearRegression': Pipeline([('scaler', StandardScaler()), ('lr', LinearRegression())]),
    'RandomForest': RandomForestRegressor(n_estimators=200, random_state=0)
}

scores = {}
for name, model in models.items():
    try:
        cv = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
        scores[name] = -cv.mean()
        print(f"{name} MAE (CV): { -cv.mean():.3f } (+/- {cv.std():.3f})")
    except Exception as e:
        print(f"{name} failed: {e}")

# Choose best (lowest MAE)
best_name = min(scores, key=scores.get)
best_model = models[best_name]
print(f"\nSelected model: {best_name}")

# Fit on all training data
best_model.fit(X, y)

# Predict missing P1 where features present
mask_missing = df['P1'].isna() & df[features].notna().all(axis=1)
print(f"Rows with missing P1 but available features: {mask_missing.sum()}")
if mask_missing.any():
    preds = best_model.predict(df.loc[mask_missing, features].values)
    df.loc[mask_missing, 'P1'] = preds
    print(f"Filled {len(preds)} P1 values")

# Save
out = 'grp28_P1_filled.csv'
df.to_csv(out, index=False)
print(f"Saved filled file to {out}")

# Quick feature importances if RF
if best_name == 'RandomForest':
    importances = best_model.feature_importances_
    for f, imp in zip(features, importances):
        print(f"{f}: {imp:.3f}")
else:
    # coeffs from linear
    if best_name == 'LinearRegression':
        coefs = best_model.named_steps['lr'].coef_
        intercept = best_model.named_steps['lr'].intercept_
        for f, c in zip(features, coefs):
            print(f"{f}: {c:.4f}")
        print(f"Intercept: {intercept:.4f}")
