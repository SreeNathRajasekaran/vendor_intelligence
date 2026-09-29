from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ProductVendorMatcher:
    """Lightweight semantic-search baseline with TF-IDF and cosine similarity.
    Can be replaced by Sentence Transformers without changing the UI contract.
    """
    def __init__(self, products):
        self.products = products.copy()
        text = (
            self.products["product_name"].fillna("") + " " +
            self.products["category"].fillna("") + " " +
            self.products["subcategory"].fillna("") + " " +
            self.products["description"].fillna("")
        )
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2), min_df=1)
        self.matrix = self.vectorizer.fit_transform(text)

    def search(self, query, top_k=5):
        q = self.vectorizer.transform([query])
        scores = cosine_similarity(q, self.matrix).ravel()
        idx = scores.argsort()[::-1][:top_k]
        out = self.products.iloc[idx].copy()
        out["semantic_similarity"] = scores[idx]
        return out.reset_index(drop=True)
