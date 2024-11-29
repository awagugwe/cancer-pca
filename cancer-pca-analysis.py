import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the breast cancer dataset
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

# Create a DataFrame with feature names
df = pd.DataFrame(X, columns=cancer.feature_names)

# 1. Data Preprocessing
# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. PCA Implementation
# Initialize PCA with 2 components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Calculate explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

# Create DataFrame with PCA results
pca_df = pd.DataFrame(
    X_pca, 
    columns=['PC1', 'PC2']
)
pca_df['target'] = y

# Get feature importance
components_df = pd.DataFrame(
    pca.components_.T,
    columns=['PC1', 'PC2'],
    index=cancer.feature_names
)

# 3. Visualization Functions
def plot_variance_explained():
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 3), cumulative_variance_ratio, 'bo-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('Explained Variance Ratio vs Number of Components')
    plt.grid(True)
    plt.show()

def plot_pca_scatter():
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, 
                         cmap='viridis', alpha=0.6)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('PCA of Breast Cancer Dataset')
    plt.colorbar(scatter)
    plt.grid(True)
    plt.show()

def plot_feature_importance():
    plt.figure(figsize=(12, 6))
    components_df['PC1'].sort_values(ascending=True).plot(kind='barh')
    plt.title('Feature Importance in First Principal Component')
    plt.xlabel('Coefficient Value')
    plt.tight_layout()
    plt.show()

# 4. Logistic Regression Implementation (Bonus)
def implement_logistic_regression():
    # Split the PCA-transformed data
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=0.2, random_state=42
    )
    
    # Train logistic regression model
    lr = LogisticRegression(random_state=42)
    lr.fit(X_train, y_train)
    
    # Make predictions
    y_pred = lr.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Generate classification report
    report = classification_report(y_test, y_pred)
    
    return accuracy, report

# 5. Main Analysis
def main():
    # Print explained variance ratio
    print("Explained Variance Ratio:")
    print(f"PC1: {explained_variance_ratio[0]:.4f}")
    print(f"PC2: {explained_variance_ratio[1]:.4f}")
    print(f"Total: {sum(explained_variance_ratio):.4f}\n")
    
    # Display top features contributing to PC1
    print("Top 5 Features Contributing to PC1:")
    pc1_contributions = abs(components_df['PC1']).sort_values(ascending=False)
    print(pc1_contributions.head())
    print("\n")
    
    # Generate visualizations
    plot_variance_explained()
    plot_pca_scatter()
    plot_feature_importance()
    
    # Implement logistic regression and print results
    accuracy, report = implement_logistic_regression()
    print("Logistic Regression Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)

if __name__ == "__main__":
    main()
