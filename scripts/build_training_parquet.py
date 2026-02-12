# save this as scripts/build_training_parquet.py and run once
import pandas as pd
from pathlib import Path

raw = Path("data/raw")
out = Path("data/raw")

wq = pd.read_csv(raw/"water_quality_training_dataset.csv")
ls = pd.read_csv(raw/"landsat_features_training.csv")
tc = pd.read_csv(raw/"terraclimate_features_training.csv")

# example join keys – adapt to your actual schema
keys = ["Sample Date"]

df = wq.merge(ls, on=keys, how="left").merge(tc, on=keys, how="left")

out.mkdir(parents=True, exist_ok=True)
df.to_parquet(out/"training_data.parquet", index=False)
print("Wrote data/raw/training_data.parquet")