# EarthBased Vendor Intelligence & Recommendation Engine

An AI/ML decision-support prototype for multi-vendor marketplace management.

## Why this project?

The project is inspired by vendor onboarding, assortment and marketplace-scaling challenges encountered while working on a multi-vendor e-commerce platform.

The central question is:

> Can latent vendor characteristics be inferred from observable marketplace behaviour to improve vendor prioritisation, assortment decisions and product/vendor matching?

The prototype combines:

- Exploratory marketplace analytics
- PCA / Factor Analysis
- Latent-variable discovery
- K-Means vendor segmentation
- Semantic product/vendor matching
- Vector-search-ready architecture
- Evidence-grounded AI copilot
- Model evaluation
- Streamlit product dashboard

### Important data note

The repository contains **synthetic/anonymized marketplace data**. It does not contain proprietary company, customer, vendor or transaction data and should not be represented as a production EarthBased system.

## Architecture

```text
Marketplace Signals
        |
        v
Data Cleaning & Feature Engineering
        |
        v
PCA / Factor Analysis
        |
        v
Latent Vendor Dimensions
        |
        v
Vendor Segmentation
        |
        +-------------------+
        |                   |
        v                   v
Vendor Prioritisation   Product Matching
                            |
                            v
                    Evidence / Retrieval
                            |
                            v
                    Vendor Copilot
```

## Latent variables

The system attempts to infer business dimensions such as:

- Operational Reliability
- Commercial Potential
- Catalogue-Market Fit
- Customer/Product Fit

A transparent aggregate, Strategic Marketplace Value, is calculated only after the latent dimensions are estimated.

The factor labels are based on observed factor loadings rather than arbitrary weights.

## ML methodology

1. Generate realistic synthetic marketplace data.
2. Clean and engineer vendor-level features.
3. Standardise variables.
4. Use PCA to inspect dimensionality.
5. Use Factor Analysis to estimate latent scores.
6. Interpret factors from their strongest loadings.
7. Use K-Means to segment vendors in latent space.
8. Use TF-IDF/cosine similarity as a lightweight semantic matching baseline.
9. Surface evidence behind recommendations.

## Dashboard

The Streamlit application contains:

1. Executive Overview
2. Vendor Intelligence
3. Latent Factors
4. Vendor Segmentation
5. Product Matching
6. AI Vendor Copilot
7. Model Evaluation

## Run locally

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501`.

## Repository structure

```text
earthbased-vendor-intelligence/
├── app.py
├── requirements.txt
├── README.md
├── data/
├── notebooks/
├── src/
├── tests/
└── .streamlit/
```

## Deployment

This application is designed for GitHub + Streamlit Community Cloud deployment.

1. Push the repository to GitHub.
2. Create a Streamlit Community Cloud app.
3. Select the `main` branch.
4. Set the main file to `app.py`.
5. Deploy.

No external API key is required for the core analytical dashboard.

## Limitations

- The dataset is synthetic.
- Vendor labels are inferred, not ground-truth business attributes.
- TF-IDF is used as a transparent semantic-search baseline.
- The AI Copilot intentionally avoids unsupported claims.
- Production deployment would require governed marketplace data, human feedback and outcome labels.

## Future improvements

- Sentence Transformer embeddings
- FAISS/Chroma vector database
- Ground-truth vendor outcome labels
- Learning-to-rank recommendation model
- Human-in-the-loop feedback
- Model drift monitoring
- LLM-based RAG with evaluation
- Recommendation acceptance tracking
- Cost/latency monitoring
