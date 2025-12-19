import pandas as pd
import numpy as np
import re

# Input and output files
input_file = 'New_PC_Without_Constraints.csv'
output_file = 'New_PC_Without_Constraints_Cleaned_v9.csv'

# Read header rows
with open(input_file, 'r', encoding='latin-1') as f:
    header_lines = [f.readline().strip(), f.readline().strip()]

# Parse parameter row
params_line = header_lines[1]
params = [p.strip() for p in params_line.split(',')]

# Parse group row
groups_line = header_lines[0]
groups = [g.strip() for g in groups_line.split(',')]

# Find group boundaries
group_ranges = {}
current_group = None
group_start = None

for i, group in enumerate(groups):
    if group and group.upper() not in ['GROUP NO']:
        if current_group != group:
            if current_group is not None and group_start is not None:
                group_ranges[current_group] = (group_start, i - 1)
            current_group = group
            group_start = i
    elif group == '':
        pass

if current_group is not None:
    group_ranges[current_group] = (group_start, len(groups) - 1)

print(f"Found {len(group_ranges)} groups")

# Load data
data_df = pd.read_csv(input_file, header=None, skiprows=2, encoding='latin-1')

# Trim to the number of params to remove extra unnamed columns
data_df = data_df.iloc[:, :len(params)]

# Set column names
data_df.columns = params

cleaned_df = data_df.copy()

for group_name, (start_col, end_col) in group_ranges.items():
    print(f"Processing group: {group_name}")

    # Find indices for P1, P2, P3, P4, P5, PC within the group
    p_indices = {}
    for col_idx in range(start_col, end_col + 1):
        if col_idx < len(params):
            param = params[col_idx].upper()
            if param in ['P1', 'P2', 'P3', 'P4', 'P5', 'PC']:
                p_indices[param] = col_idx

    required = {'P1', 'P2', 'P3', 'P4', 'PC'}
    if not required.issubset(set(p_indices.keys())):
        print(f"Skipping group {group_name}: missing required params")
        continue

    # Select columns for this group
    cols = [p_indices[param] for param in ['PC', 'P1', 'P2', 'P3', 'P4']]
    if 'P5' in p_indices:
        cols.append(p_indices['P5'])

    group_df = data_df.iloc[:, cols].copy()
    group_df.columns = ['PC', 'P1', 'P2', 'P3', 'P4'] + (['P5'] if 'P5' in p_indices else [])

    print(f"Group {group_name}: initial rows {len(group_df)}")

    # Convert PC and set NaN if out of range
    group_df['PC'] = pd.to_numeric(group_df['PC'], errors='coerce')
    group_df.loc[~((group_df['PC'] >= 0) & (group_df['PC'] <= 100)), 'PC'] = np.nan

    # -------- RANGE & IMPUTATION SECTION (CHANGED) --------
    temp_cols = ['P1', 'P2', 'P3', 'P4'] + (['P5'] if 'P5' in group_df.columns else [])
    for col in temp_cols:
        group_df[col] = pd.to_numeric(group_df[col], errors='coerce')

    # Reasonable ranges:
    # P1: 200-300 m³/hr
    group_df.loc[~((group_df['P1'] >= 200) & (group_df['P1'] <= 300)), 'P1'] = np.nan

    # P2, P3: 105–135 °C
    for col in ['P2', 'P3']:
        if col in group_df.columns:
            group_df.loc[~((group_df[col] >= 105) & (group_df[col] <= 135)), col] = np.nan

    # P4: approx 100, say 90-110
    group_df.loc[~((group_df['P4'] >= 90) & (group_df['P4'] <= 110)), 'P4'] = np.nan

    print(f"After range checks: P1 NaN={group_df['P1'].isna().sum()}, "
          f"P2 NaN={group_df['P2'].isna().sum()}, "
          f"P3 NaN={group_df['P3'].isna().sum()}, "
          f"P4 NaN={group_df['P4'].isna().sum()}, "
          f"P5 NaN={group_df['P5'].isna().sum() if 'P5' in group_df.columns else 'NA'}")

    # For now, just keep all rows with NaNs set
    # No dropping or imputation
    # -------- END CHANGES --------

    # Update cleaned_df with the modified group_df
    cleaned_df.iloc[:, cols] = group_df.astype(object).values

# Save the cleaned DataFrame
with open(output_file, 'w', encoding='latin-1') as f:
    f.write(header_lines[0] + '\n')
    f.write(header_lines[1] + '\n')
cleaned_df.to_csv(output_file, mode='a', index=False, encoding='latin-1')

print(f"Cleaned data saved to {output_file}")
print(f"Original rows: {len(data_df)}, Cleaned rows: {len(cleaned_df)}")