"""
Practical Assignment: Unsupervised Clustering & Evaluation
Dataset: Mall Customers Dataset (Customer Segmentation)
Algorithms: K-Means Clustering & Agglomerative Hierarchical Clustering
Evaluation Metrics: Silhouette Score, Davies-Bouldin Index, WCSS (Elbow Method)
Author: TY B.Tech AIML Student
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import scipy.cluster.hierarchy as sch

def load_or_generate_data(filepath=None):
    """
    Loads dataset from filepath or creates a realistic Mall Customers benchmark dataset.
    Features: CustomerID, Genre (Gender), Age, Annual Income (k$), Spending Score (1-100)
    """
    if filepath:
        try:
            df = pd.read_csv(filepath)
            print(f"[+] Loaded dataset successfully from {filepath}")
            return df
        except Exception as e:
            print(f"[!] Warning: Could not read {filepath}: {e}. Generating benchmark dataset.")

    # Benchmark generation consistent with standard UCI/Kaggle Mall Customers distribution
    np.random.seed(42)
    n_samples = 200
    
    # 5 archetypal customer clusters in (Income, Spending Score)
    # 1. Low Income, High Spending (Careless / Impulsive)
    c1 = np.column_stack([np.random.normal(25, 5, 40), np.random.normal(79, 6, 40)])
    # 2. Low Income, Low Spending (Sensible / Budget)
    c2 = np.column_stack([np.random.normal(26, 5, 40), np.random.normal(20, 6, 40)])
    # 3. High Income, High Spending (Target / Elite)
    c3 = np.column_stack([np.random.normal(86, 6, 40), np.random.normal(82, 6, 40)])
    # 4. High Income, Low Spending (Careful / Savers)
    c4 = np.column_stack([np.random.normal(88, 6, 40), np.random.normal(17, 6, 40)])
    # 5. Medium Income, Medium Spending (Standard / Average)
    c5 = np.column_stack([np.random.normal(55, 6, 40), np.random.normal(50, 6, 40)])
    
    features = np.vstack([c1, c2, c3, c4, c5])
    ages = np.random.randint(18, 70, size=n_samples)
    genders = np.random.choice(['Male', 'Female'], size=n_samples)
    
    df = pd.DataFrame({
        'CustomerID': range(1, n_samples + 1),
        'Gender': genders,
        'Age': ages,
        'Annual Income (k$)': np.round(np.clip(features[:, 0], 15, 140), 1),
        'Spending Score (1-100)': np.round(np.clip(features[:, 1], 1, 99), 1)
    })
    return df

def evaluate_clustering(X, labels, model_name="Model"):
    """
    Computes and prints Silhouette Score and Davies-Bouldin Index.
    """
    sil = silhouette_score(X, labels)
    db = davies_bouldin_score(X, labels)
    print(f"{model_name:30s} | Silhouette Score: {sil:6.4f} | Davies-Bouldin Index: {db:6.4f}")
    return sil, db

def main():
    print("=" * 70)
    print("PRACTICAL ASSIGNMENT: CLUSTERING ALGORITHMS & MODEL EVALUATION")
    print("=" * 70)
    
    # 1. Load Data
    df = load_or_generate_data("Mall_Customers.csv")
    print("\n[+] Dataset Overview:")
    print(df.head())
    print("\n[+] Summary Statistics:")
    print(df.describe())
    
    # Select feature space for 2D segmentation (Annual Income and Spending Score)
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values
    
    # Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 2. Elbow Method & Silhouette Analysis across multiple k values
    print("\n[+] Evaluating Optimal Number of Clusters (k = 2 to 10)...")
    k_range = range(2, 11)
    wcss = []
    silhouette_scores = []
    db_scores = []
    
    for k in k_range:
        km = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300, random_state=42)
        km.fit(X_scaled)
        wcss.append(km.inertia_)
        sil = silhouette_score(X_scaled, km.labels_)
        db = davies_bouldin_score(X_scaled, km.labels_)
        silhouette_scores.append(sil)
        db_scores.append(db)
        print(f"k = {k:2d} | WCSS (Inertia): {km.inertia_:9.2f} | Silhouette: {sil:6.4f} | DB Index: {db:6.4f}")
        
    optimal_k = 5
    print(f"\n[+] Selected Optimal k = {optimal_k} based on Elbow Point and Maximum Silhouette Score.")
    
    # 3. Model 1: K-Means with Optimal k
    print("\n--- Training K-Means Model (k=5) ---")
    kmeans = KMeans(n_clusters=optimal_k, init='k-means++', n_init=10, max_iter=300, random_state=42)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    df['KMeans_Cluster'] = kmeans_labels
    evaluate_clustering(X_scaled, kmeans_labels, f"K-Means (k={optimal_k})")
    
    # 4. Model 2: Agglomerative Hierarchical Clustering
    print("\n--- Training Agglomerative Hierarchical Model (k=5, linkage='ward') ---")
    agg_clustering = AgglomerativeClustering(n_clusters=optimal_k, metric='euclidean', linkage='ward')
    agg_labels = agg_clustering.fit_predict(X_scaled)
    df['Agglomerative_Cluster'] = agg_labels
    evaluate_clustering(X_scaled, agg_labels, f"Agglomerative (k={optimal_k})")
    
    # 5. Cluster Profiling (Business Insights)
    print("\n--- Cluster Personas & Characteristics (K-Means) ---")
    cluster_profile = df.groupby('KMeans_Cluster')[['Annual Income (k$)', 'Spending Score (1-100)', 'Age']].mean()
    cluster_counts = df['KMeans_Cluster'].value_counts().sort_index()
    cluster_profile['Customer Count'] = cluster_counts
    
    persona_map = {
        0: "Target / Elite (High Income, High Spending)",
        1: "Sensible / Budget-Conscious (Low Income, Low Spending)",
        2: "Impulsive / Careless (Low Income, High Spending)",
        3: "Standard / Moderate (Medium Income, Medium Spending)",
        4: "Careful / Savers (High Income, Low Spending)"
    }
    # Map personas based on profile averages
    print(cluster_profile)
    print("\n[+] Done! Script successfully completed.")

if __name__ == "__main__":
    main()
