import pandas as pd
from sklearn.metrics import silhouette_score

def evaluate_clusters(scores, labels):
    return {
        "silhouette_score": float(silhouette_score(scores, labels)),
        "cluster_count": int(len(set(labels)))
    }

def factor_summary(pca):
    ratios = pca.explained_variance_ratio_
    return pd.DataFrame({
        "component": [f"PC{i+1}" for i in range(len(ratios))],
        "explained_variance": ratios,
        "cumulative_variance": ratios.cumsum()
    })
