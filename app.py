import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Product Recommendation System",
    layout="wide"
)

# ---------------- GLOBAL CSS (DESIGN ONLY) ----------------
st.markdown("""
<style>
.main {
    background-color: #0e1117;
}

/* Card style */
.card {
    background: #111827;
    padding: 22px;
    border-radius: 14px;
    margin-bottom: 20px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.35);
}

/* Section title */
.section-title {
    font-size: 26px;
    font-weight: 700;
    margin-bottom: 10px;
}

/* Helper text */
.helper {
    color: #9ca3af;
    font-size: 14px;
    margin-top: -5px;
    margin-bottom: 10px;
}

/* Highlight box */
.highlight {
    background: linear-gradient(90deg,#064e3b,#065f46);
    padding: 14px;
    border-radius: 10px;
    color: #ecfdf5;
    font-weight: 600;
    margin-top: 10px;
}

/* Buttons */
.stButton>button {
    border-radius: 10px;
    padding: 0.5rem 1.2rem;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
<div class="card" style="text-align:center;">
    <h1 style="color:white;">Product Recommendation System</h1>
    <p style="color:#d1d5db;">
        SVD-Based Recommendation Engine with User & Product Clustering
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------- LOAD FILES ----------------
@st.cache_data
def load_files():
    U = joblib.load("models/user_latent_features.joblib")
    Vt = joblib.load("models/product_latent_features.joblib")
    sigma = joblib.load("models/sigma_matrix.joblib")

    user_clusters_df = pd.read_csv("user_clusters.csv")
    product_clusters_df = pd.read_csv("product_clusters.csv")

    return U, Vt, sigma, user_clusters_df, product_clusters_df

U, Vt, sigma, user_clusters_df, product_clusters_df = load_files()

# ---------------- SAFE INDEX MAP ----------------
user_id_to_index = dict(
    zip(user_clusters_df["userid"], range(len(user_clusters_df)))
)

product_id_to_index = dict(
    zip(product_clusters_df["productid"], range(len(product_clusters_df)))
)

# =================================================
# SIDE BY SIDE LAYOUT
# =================================================
col1, col2 = st.columns(2)

# ================= LEFT COLUMN ===================
with col1:
    st.markdown("""
    <div class="card">
        <div class="section-title">👤 User → Product Recommendation</div>
    </div>
    """, unsafe_allow_html=True)

    user_id = st.selectbox(
        "Select User ID",
        user_clusters_df["userid"]
    )
    st.markdown(
        '<div class="helper">Choose a user to get personalized product recommendations</div>',
        unsafe_allow_html=True
    )

    top_n_products = st.slider(
        "Number of Product Recommendations",
        5, 20, 10
    )

    if st.button("Recommend Products", key="btn_products"):
        user_index = user_id_to_index[user_id]

        user_vector = U[user_index].reshape(1, -1)
        scores = (user_vector @ sigma @ Vt).flatten()

        top_idx = np.argsort(scores)[::-1][:top_n_products]

        output = pd.DataFrame({
            "Product ID": product_clusters_df.iloc[top_idx]["productid"].values,
            "Product Cluster": product_clusters_df.iloc[top_idx]["product_cluster"].values
        })

        st.markdown(
            f'<div class="highlight">✅ User ID: {user_id} | Cluster: '
            f'{user_clusters_df.loc[user_clusters_df["userid"]==user_id,"cluster"].values[0]}</div>',
            unsafe_allow_html=True
        )
        st.dataframe(output, use_container_width=True)

# ================= RIGHT COLUMN ==================
with col2:
    st.markdown("""
    <div class="card">
        <div class="section-title">📦 Product → User Recommendation</div>
    </div>
    """, unsafe_allow_html=True)

    product_id = st.selectbox(
        "Select Product ID",
        product_clusters_df["productid"]
    )
    st.markdown(
        '<div class="helper">Find users who belong to the same product cluster</div>',
        unsafe_allow_html=True
    )

    top_n_users = st.slider(
        "Number of User Recommendations",
        5, 20, 10,
        key="user_slider"
    )

    if st.button("Recommend Users", key="btn_users"):
        product_cluster = product_clusters_df.loc[
            product_clusters_df["productid"] == product_id,
            "product_cluster"
        ].values[0]

        matched_users = user_clusters_df[
            user_clusters_df["cluster"] == product_cluster
        ].head(top_n_users)

        st.markdown(
            f'<div class="highlight">📌 Product ID: {product_id} | Cluster: {product_cluster}</div>',
            unsafe_allow_html=True
        )
        st.dataframe(matched_users, use_container_width=True)
        st.caption("Users shown belong to the same product cluster.")
