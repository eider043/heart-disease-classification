import pandas as pd
import numpy as np

def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    cols = ["age", "sex", "cp", "trestbps", "chol", "fbs",
            "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
    
    df = pd.read_csv(url, names=cols, na_values="?")
    df.dropna(inplace=True)
    df["target"] = (df["target"] > 0).astype(int)
    df.to_csv("../data/heart_disease.csv", index=False)
    
    print(f"Dataset cargado: {df.shape}")
    print(f"Distribucion target:\n{df['target'].value_counts()}")
    return df