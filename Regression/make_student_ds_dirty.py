import pandas as pd
import numpy as np

# Load the clean dataset
df = pd.read_csv('../Datasets/student_performance.csv')

# --- Introduce Missing Values ---
# Randomly set 10% of the 'Hours Studied' to NaN
np.random.seed(42)
mask = np.random.rand(len(df)) < 0.1
df.loc[mask, 'Hours Studied'] = np.nan

# Randomly set 5% of the 'Sleep Hours' to NaN
mask = np.random.rand(len(df)) < 0.05
df.loc[mask, 'Sleep Hours'] = np.nan

# --- Change Data Types ---
# Convert 'Previous Scores' to string by mistake
df['Previous Scores'] = df['Previous Scores'].astype(str)

# --- Introduce Duplicates ---
# Append 5 duplicate rows to simulate duplicate entries
df_duplicates = df.sample(5, random_state=42)
df = pd.concat([df, df_duplicates], ignore_index=True)

# --- Add Irrelevant Column ---
# Add a column with random textual notes that are not needed for regression
df['Notes'] = np.where(np.random.rand(len(df)) < 0.5, 'review', 'excellent')

# --- Introduce Outliers ---
# For the 'Sample Question Papers Practiced' column, set a few extreme values
outlier_indices = np.random.choice(df.index, size=3, replace=False)
df.loc[outlier_indices, 'Sample Question Papers Practiced'] = df['Sample Question Papers Practiced'].max() * 3

# Save the dirty dataset
df.to_csv('student_performance_dirty.csv', index=False)
print("Dirty dataset saved as 'student_performance_dirty.csv'")

