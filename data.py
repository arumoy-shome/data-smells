import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter

REGISTRY = {
    "abalone": ("rodolfomendes/abalone-dataset/versions/3", "abalone.csv"),
}

def load_dataset(name: str) -> pd.DataFrame:
    h, p = REGISTRY[name]
    
    return kagglehub.load_dataset(
        adapter = KaggleDatasetAdapter.PANDAS,
        handle = h,
        path = p,
    )