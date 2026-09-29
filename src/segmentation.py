import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def select_k(scores, k_values=range(2,7)):
    rows = []
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = model.fit_predict(scores)
        rows.append({"k": k, "inertia": model.inertia_,
                     "silhouette": silhouette_score(scores, labels)})
    return pd.DataFrame(rows)

def fit_clusters(scores, k):
    model = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = model.fit_predict(scores)
    return model, labels
