import pandas as pd
import numpy as np

# Read data
df = pd.read_csv('dataset_phishing.csv')

# Check the number of negative values for each column
negative_counts = {}
for column in df.columns:
    # Convert to numeric, set to NaN if conversion fails
    col_numeric = pd.to_numeric(df[column], errors='coerce')
    negative_count = (col_numeric < 0).sum()
    if negative_count > 0:
        negative_counts[column] = negative_count

# Print result
print("\n음수 값을 포함한 컬럼과 음수 값의 수:")
for column, count in negative_counts.items():
    print(f"{column}: {count}개")


