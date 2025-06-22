import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter

REGISTRY = {
    "abalone": ("rodolfomendes/abalone-dataset/versions/3", "abalone.csv"),
    "adult": ("uciml/adult-census-income/versions/3", "adult.csv"),
}

def load_dataset(name: str) -> pd.DataFrame:
    h, p = REGISTRY[name]
    
    return kagglehub.load_dataset(
        adapter = KaggleDatasetAdapter.PANDAS,
        handle = h,
        path = p,
    )