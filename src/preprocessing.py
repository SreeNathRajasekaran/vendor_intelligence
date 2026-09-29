from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "fulfilment_rate", "cancellation_rate", "return_rate", "stock_availability",
    "avg_delivery_days", "response_time_hours", "order_count", "gmv",
    "average_order_value", "repeat_customer_rate", "conversion_rate",
    "customer_engagement", "category_coverage", "product_count", "average_rating",
    "price_index"
]

def load_vendor_data(data_dir="data"):
    return pd.read_csv(Path(data_dir) / "vendors.csv")

def prepare_features(vendors):
    X = vendors[FEATURES].copy()
    X["gmv_log"] = __import__("numpy").log1p(X["gmv"])
    X["order_count_log"] = __import__("numpy").log1p(X["order_count"])
    X = X.drop(columns=["gmv", "order_count"])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X, X_scaled, scaler
