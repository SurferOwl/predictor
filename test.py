import pandas as pd
from pathlib import Path

for f in Path("cleaned_data").glob("*.csv"):
    df = pd.read_csv(f, nrows=5)
    if "name" in df.columns:
        print(f"\n{f.name}:")
        print(df["name"].tolist())