import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.preprocessing import StandardScaler

def fit_pca(X_scaled, max_components=None):
    n = max_components or min(X_scaled.shape)
    pca = PCA(n_components=n, random_state=42)
    scores = pca.fit_transform(X_scaled)
    return pca, scores

def choose_components(pca, threshold=0.80):
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    return int(np.argmax(cumulative >= threshold) + 1)

def fit_factor_analysis(X_scaled, n_factors):
    model = FactorAnalysis(n_components=n_factors, random_state=42)
    scores = model.fit_transform(X_scaled)
    loadings = pd.DataFrame(
        model.components_.T,
        columns=[f"Factor_{i+1}" for i in range(n_factors)]
    )
    return model, scores, loadings

def label_factors(loadings):
    labels = {}
    for col in loadings.columns:
        top = loadings[col].abs().sort_values(ascending=False).head(4).index.tolist()
        text = " + ".join(top[:3])
        labels[col] = text
    return labels
