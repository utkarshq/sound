import pandas as pd
import json

df = pd.read_csv("opensmile.csv", sep=";")
features = df[["F0", "loudness", "spectralFlux"]].mean().tolist()
with open("opensmile.json", "w") as f:
    json.dump({"opensmile_features": features}, f)