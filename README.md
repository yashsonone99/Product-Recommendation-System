# Product-Recommendation-System
End-to-end Product Recommendation System using Collaborative Filtering, SVD, and Clustering with EDA, evaluation, and web deployment.
# 🛒 Product Recommendation System

An end-to-end **E-commerce Product Recommendation System** built using Machine Learning techniques including **Item-Based Collaborative Filtering, SVD Matrix Factorization, and K-Means Clustering**.  
The project covers the complete pipeline from **EDA to deployment**.

---

## 📌 Project Overview
E-commerce platforms face challenges in recommending relevant products due to large catalogs and sparse user interactions.  
This project aims to solve that problem by building a **personalized recommendation system** based on user-product rating behavior.

---

## 📊 Dataset Details
- **Source:** User-product rating dataset
- **Attributes:**
  - `userId` – Unique user identifier
  - `productId` – Unique product identifier
  - `rating` – Rating given by user
  - `timestamp` – Time of rating (ignored)
- Dataset used for collaborative filtering and clustering

---

## 🔍 Exploratory Data Analysis (EDA)
- Rating distribution analysis
- Most active users and most rated products
- Data sparsity analysis
- User–Item interaction heatmap
- User & product behavior insights

---

## 🤖 Models Implemented
### 1. Item-Based Collaborative Filtering
- Similarity-based recommendation using user rating patterns

### 2. SVD Matrix Factorization
- Latent feature extraction
- Rating prediction for sparse data

### 3. K-Means Clustering
- User segmentation based on rating behavior
- Product clustering using latent features
- Hierarchical clustering validation

---

## 📈 Model Evaluation
- Precision@K
- Precision@10
- Hit@K
- Ranking-based evaluation to measure recommendation relevance

---

## 🚀 Deployment
- Web application built using **Flask**
- Real-time recommendation generation
- User-friendly interface for product suggestions

📽️ **Demo Video:** Available in the `demo/` folder

---

## 🛠️ Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- Flask
- Matplotlib & Seaborn
- Jupyter Notebook

---

## 📁 Project Structure
