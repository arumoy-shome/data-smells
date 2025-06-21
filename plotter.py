import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
import pandas as pd


def corr(corr: pd.DataFrame) -> None:
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        fig, ax = plt.subplots()
        sns.heatmap(corr, mask=mask, square=True, ax=ax)
        fig.tight_layout()
