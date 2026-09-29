from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parents[1]

def test_vendor_data():
    df = pd.read_csv(ROOT / "data/vendors.csv")
    assert len(df) >= 80
    assert {"vendor_id","gmv","fulfilment_rate"}.issubset(df.columns)

def test_products():
    df = pd.read_csv(ROOT / "data/products.csv")
    assert len(df) >= 800
    assert {"product_id","vendor_id","description"}.issubset(df.columns)

def test_latent_scores():
    df = pd.read_csv(ROOT / "data/vendor_scores.csv")
    assert "Strategic Marketplace Value" in df.columns
    assert df["Strategic Marketplace Value"].between(0,100).all()
