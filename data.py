import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter

REGISTRY = {
    "abalone": ("rodolfomendes/abalone-dataset/versions/3", "abalone.csv"),
    "adult": ("uciml/adult-census-income/versions/3", "adult.csv"),
    "airbnb": ("dgomonov/new-york-city-airbnb-open-data/versions/3", "AB_NYC_2019.csv"),
    "avocado": ("neuromusic/avocado-prices/versions/1", "avocado.csv"),
}

def load_dataset(name: str) -> pd.DataFrame:
    h, p = REGISTRY[name]
    
    return kagglehub.load_dataset(
        adapter = KaggleDatasetAdapter.PANDAS,
        handle = h,
        path = p,
    )