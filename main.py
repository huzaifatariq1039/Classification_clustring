import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans, DBSCAN
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import classification_report, silhouette_score
import matplotlib.pyplot as plt

# Load Iris Dataset for Classification
iris_data = pd.read_csv("iris_dataset.csv")
X_iris = iris_data.drop(columns=["target"])
y_iris = iris_data["target"]

# Split Classification Data
X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(X_iris, y_iris, test_size=0.2, random_state=42)

# Load Mall Customers Dataset for Clustering
mall_features = pd.read_csv("mall_features_scaled.csv")

# CLASSIFICATION TASK IMPLEMENTATION
classification_models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(),
    "Random Forest": RandomForestClassifier()
}

classification_results = {}

for model_name, model in classification_models.items():
    model.fit(X_train_iris, y_train_iris)
    y_pred = model.predict(X_test_iris)
    classification_results[model_name] = classification_report(y_test_iris, y_pred, output_dict=True)

# CLUSTERING TASK IMPLEMENTATION
clustering_models = {
    "K-Means": KMeans(n_clusters=3, random_state=42),
    "Hierarchical": None,  # Handled separately
    "DBSCAN": DBSCAN(eps=1.5, min_samples=5)
}

clustering_results = {}

# K-Means Clustering
kmeans_labels = clustering_models["K-Means"].fit_predict(mall_features)
clustering_results["K-Means"] = silhouette_score(mall_features, kmeans_labels)

# Hierarchical Clustering
hierarchical_linkage = linkage(mall_features, method='ward')
hierarchical_labels = fcluster(hierarchical_linkage, t=3, criterion='maxclust')
clustering_results["Hierarchical"] = silhouette_score(mall_features, hierarchical_labels)

# DBSCAN Clustering
dbscan_labels = clustering_models["DBSCAN"].fit_predict(mall_features)
if len(set(dbscan_labels)) > 1:
    clustering_results["DBSCAN"] = silhouette_score(mall_features, dbscan_labels)
else:
    clustering_results["DBSCAN"] = "Insufficient clusters"

# VISUALIZATION OF CLUSTERING
plt.figure(figsize=(10, 5))

# K-Means Visualization
plt.subplot(1, 3, 1)
plt.scatter(mall_features.iloc[:, 0], mall_features.iloc[:, 1], c=kmeans_labels, cmap='viridis')
plt.title('K-Means Clustering')

# Hierarchical Visualization
plt.subplot(1, 3, 2)
plt.scatter(mall_features.iloc[:, 0], mall_features.iloc[:, 1], c=hierarchical_labels, cmap='viridis')
plt.title('Hierarchical Clustering')

# DBSCAN Visualization
plt.subplot(1, 3, 3)
plt.scatter(mall_features.iloc[:, 0], mall_features.iloc[:, 1], c=dbscan_labels, cmap='viridis')
plt.title('DBSCAN Clustering')

plt.tight_layout()
plt.savefig("clustering_visualization.png")
plt.show()

# OUTPUT RESULTS
classification_results_df = pd.DataFrame.from_dict(classification_results, orient='index')
clustering_results_df = pd.DataFrame.from_dict(clustering_results, orient='index', columns=['Silhouette Score'])

classification_results_df.to_csv("classification_results.csv")
clustering_results_df.to_csv("clustering_results.csv")
