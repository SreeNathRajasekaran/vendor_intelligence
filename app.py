import os
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from src.recommendation import ProductVendorMatcher

ROOT = Path(__file__).parent
DATA = ROOT / "data"

st.set_page_config(page_title="EarthBased Vendor Intelligence", page_icon="◈", layout="wide")

@st.cache_data
def load_data():
    vendors = pd.read_csv(DATA/"vendors.csv")
    products = pd.read_csv(DATA/"products.csv")
    scores = pd.read_csv(DATA/"vendor_scores.csv")
    loadings = pd.read_csv(DATA/"factor_loadings.csv")
    cluster_eval = pd.read_csv(DATA/"cluster_evaluation.csv")
    pca_summary = pd.read_csv(DATA/"pca_summary.csv")
    return vendors, products, scores, loadings, cluster_eval, pca_summary

@st.cache_resource
def get_matcher(products):
    return ProductVendorMatcher(products)

vendors, products, scores, loadings, cluster_eval, pca_summary = load_data()
matcher = get_matcher(products)

st.sidebar.title("EarthBased")
st.sidebar.caption("Vendor Intelligence Prototype")
page = st.sidebar.radio("Navigate", [
    "Executive Overview", "Vendor Intelligence", "Latent Factors",
    "Vendor Segmentation", "Product Matching", "AI Vendor Copilot", "Model Evaluation"
])

st.sidebar.markdown("---")
st.sidebar.caption("Synthetic/anonymized data. Prototype inspired by marketplace product-management challenges.")

latent_cols = [c for c in [
    "Operational Reliability", "Commercial Potential",
    "Catalogue-Market Fit", "Customer/Product Fit",
    "Strategic Marketplace Value"
] if c in scores.columns]

if page == "Executive Overview":
    st.title("Vendor Intelligence & Recommendation Engine")
    st.write("A decision-support prototype for inferring hidden marketplace dimensions from observable vendor behaviour.")
    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("Vendors", len(vendors))
    k2.metric("Products", len(products))
    k3.metric("Orders", f"{pd.read_csv(DATA/'orders.csv').shape[0]:,}")
    k4.metric("GMV", f"₹{vendors.gmv.sum()/1e6:.1f}M")
    k5.metric("Avg. Rating", f"{vendors.average_rating.mean():.2f}")

    if "Commercial Potential" in scores and "Operational Reliability" in scores:
        fig = px.scatter(
            scores, x="Commercial Potential", y="Operational Reliability",
            size="gmv", color=scores["vendor_cluster"].astype(str),
            hover_name="vendor_name", hover_data=["category","city"],
            labels={"color":"Vendor Cluster"},
            title="Vendor Landscape: Commercial Potential vs Operational Reliability"
        )
        st.plotly_chart(fig, use_container_width=True)

    c1,c2 = st.columns(2)
    with c1:
        seg = scores["vendor_cluster"].value_counts().sort_index().reset_index()
        seg.columns = ["Cluster","Vendors"]
        st.plotly_chart(px.bar(seg, x="Cluster", y="Vendors", title="Vendor Segment Distribution"), use_container_width=True)
    with c2:
        cat = vendors.groupby("category").agg(GMV=("gmv","sum"), Vendors=("vendor_id","nunique")).reset_index()
        st.plotly_chart(px.bar(cat, x="category", y="GMV", title="GMV by Category"), use_container_width=True)

elif page == "Vendor Intelligence":
    st.title("Vendor Intelligence")
    selected = st.selectbox("Select vendor", scores.vendor_id.tolist())
    row = scores[scores.vendor_id == selected].iloc[0]
    st.subheader(row.vendor_name)
    st.caption(f"{row.category} • {row.city}")
    cols = st.columns(5)
    for c, metric in zip(cols, ["gmv","average_rating","fulfilment_rate","cancellation_rate","return_rate"]):
        val = row[metric]
        label = metric.replace("_"," ").title()
        if metric == "gmv":
            text = f"₹{val:,.0f}"
        elif metric in ["fulfilment_rate","cancellation_rate","return_rate"]:
            text = f"{val:.1%}"
        else:
            text = f"{val:.2f}"
        c.metric(label, text)

    st.subheader("Latent Vendor Profile")
    available = [c for c in latent_cols if c != "Strategic Marketplace Value"]
    vals = pd.DataFrame({"Factor": available, "Score": [row[c] for c in available]})
    st.plotly_chart(px.bar(vals, x="Score", y="Factor", orientation="h", range_x=[0,100], title="Inferred latent dimensions"), use_container_width=True)

    st.info("These are inferred dimensions from observed marketplace signals; they are not directly observed attributes.")

elif page == "Latent Factors":
    st.title("Latent Factor Analysis")
    st.write("The system uses observable marketplace signals to uncover lower-dimensional business dimensions.")
    st.subheader("Explained Variance")
    st.plotly_chart(px.line(pca_summary, x="component", y="cumulative_variance", markers=True, title="Cumulative PCA variance"), use_container_width=True)
    st.subheader("Factor Loadings")
    fl = loadings.set_index("feature")
    st.dataframe(fl.style.format("{:.3f}"), use_container_width=True)
    st.caption("Factor labels are assigned from the strongest statistically observed loadings and then translated into business language.")
    st.subheader("Business Interpretation")
    for factor in [c for c in latent_cols if c != "Strategic Marketplace Value"]:
        if factor in fl.columns:
            top = fl[factor].abs().sort_values(ascending=False).head(4).index.tolist()
            st.markdown(f"**{factor}** — strongest contributing signals: " + ", ".join(top))

elif page == "Vendor Segmentation":
    st.title("Vendor Segmentation")
    st.plotly_chart(px.line(cluster_eval, x="k", y="silhouette", markers=True, title="Silhouette score by cluster count"), use_container_width=True)
    best_k = int(cluster_eval.loc[cluster_eval.silhouette.idxmax(),"k"])
    st.success(f"Selected cluster count based on highest silhouette score: {best_k}")
    x = "Commercial Potential" if "Commercial Potential" in scores else latent_cols[0]
    y = "Operational Reliability" if "Operational Reliability" in scores else latent_cols[1]
    st.plotly_chart(px.scatter(scores, x=x, y=y, color=scores.vendor_cluster.astype(str),
                               hover_name="vendor_name", title="Latent vendor segments"), use_container_width=True)
    st.dataframe(scores.groupby("vendor_cluster")[latent_cols].mean().round(1), use_container_width=True)

elif page == "Product Matching":
    st.title("Product / Vendor Matching")
    query = st.text_area("Describe the requirement", "Premium sustainable packaging with reliable fulfilment for restaurant orders.")
    top_k = st.slider("Number of matches", 3, 10, 5)
    results = matcher.search(query, top_k)
    enriched = results.merge(scores, on="vendor_id", how="left")
    cols = ["vendor_name","product_name","category","subcategory","semantic_similarity"] + [c for c in latent_cols if c in enriched.columns]
    st.dataframe(enriched[cols].round(3), use_container_width=True)
    st.caption("Semantic similarity is a transparent TF-IDF/cosine baseline. The architecture can be upgraded to Sentence Transformers without changing the product workflow.")

elif page == "AI Vendor Copilot":
    st.title("AI Vendor Copilot")
    st.write("Evidence-grounded prototype for vendor intelligence questions.")
    question = st.text_input("Ask a marketplace question", "Which vendors have strong commercial potential but weaker operational reliability?")
    if st.button("Analyze"):
        q = question.lower()
        df = scores.copy()
        evidence = ""
        if "commercial" in q and ("operational" in q or "reliability" in q):
            df = df.sort_values(["Commercial Potential","Operational Reliability"], ascending=[False,True]).head(8)
            evidence = "Filtered for high commercial potential and comparatively lower operational reliability."
        elif "priorit" in q or "strategic" in q:
            df = df.sort_values("Strategic Marketplace Value", ascending=False).head(8)
            evidence = "Ranked by the transparent average of the available latent dimensions."
        elif "catalog" in q or "assortment" in q:
            df = df.sort_values("Catalogue-Market Fit", ascending=False).head(8)
            evidence = "Ranked by inferred catalogue-market fit."
        else:
            df = df.sort_values("Strategic Marketplace Value", ascending=False).head(8)
            evidence = "Defaulted to strategic marketplace value because the question did not map to a supported analytical intent."
        st.write(evidence)
        st.dataframe(df[["vendor_name","category"] + latent_cols].round(1), use_container_width=True)
        st.info("Evidence-first design: the prototype surfaces the underlying records rather than inventing unsupported vendor claims.")

elif page == "Model Evaluation":
    st.title("Model Evaluation")
    c1,c2,c3 = st.columns(3)
    best = cluster_eval.loc[cluster_eval.silhouette.idxmax()]
    c1.metric("Selected K", int(best.k))
    c2.metric("Best Silhouette", f"{best.silhouette:.3f}")
    c3.metric("PCA Components", len(pca_summary))
    st.subheader("Cluster Evaluation")
    st.dataframe(cluster_eval.round(4), use_container_width=True)
    st.subheader("Production KPIs to Track")
    st.markdown("""
    - Vendor shortlisting time
    - Recommendation acceptance rate
    - Catalogue coverage
    - Supplier response rate
    - Conversion rate
    - Repeat purchase rate
    - Operational issue rate
    - Recommendation latency and inference cost
    """)
    st.warning("These are proposed production KPIs, not measured outcomes from this prototype.")
