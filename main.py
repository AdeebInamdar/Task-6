import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from matplotlib.colors import ListedColormap

# --- 1. Data Loading and Preparation ---

# Load the dataset
# Assuming the file 'Iris.csv' is in the current directory
df = pd.read_csv('Iris.csv')

# Drop the 'Id' column as it's not a feature
df = df.drop('Id', axis=1)

# Define features (X) and target (y)
X = df.drop('Species', axis=1).values
y = df['Species'].values

# Split the data into training and testing sets (e.g., 70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)

# --- 2. Feature Normalization (Scaling) ---

# Normalization is important for KNN as it relies on distance metrics [cite: 17]
scaler = StandardScaler()

# Fit the scaler on the training data and transform both training and testing data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# --- 3. Implement K-Nearest Neighbors (KNN) ---

# Initialize the KNN classifier with K=5 as a starting point [cite: 9]
K_OPTIMAL = 5 # Starting value for K
knn = KNeighborsClassifier(n_neighbors=K_OPTIMAL)

# Train the model
knn.fit(X_train_scaled, y_train)

# Make predictions on the scaled test data
y_pred = knn.predict(X_test_scaled)


# --- 4. Model Evaluation for K=5 ---

print(f"--- Evaluation for K = {K_OPTIMAL} ---")

# Calculate Accuracy [cite: 10]
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Confusion Matrix [cite: 10]
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Classification Report
report = classification_report(y_test, y_pred)
print("\nClassification Report:")
print(report)

print("-" * 50)


# --- 5. Experiment with Different K Values (Elbow Method) ---

# Determine the optimal K by plotting the error rate for different K values
error_rate = []
max_k = 25

# Experiment with odd values of K (1, 3, 5, ..., max_k) to avoid ties
for i in range(1, max_k + 1, 2):
    knn_i = KNeighborsClassifier(n_neighbors=i)
    knn_i.fit(X_train_scaled, y_train)
    y_pred_i = knn_i.predict(X_test_scaled)
    # Calculate the mean error rate (1 - accuracy)
    error_rate.append(np.mean(y_pred_i != y_test))

# Plot the error rate vs. K
plt.figure(figsize=(10, 6))
plt.plot(range(1, max_k + 1, 2), error_rate, color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value (Elbow Plot)')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.grid(True)
plt.show()

# Based on the plot, select the K corresponding to the lowest error rate.
# Let's say we find K=7 is better (or another value). We can re-evaluate with the chosen K.


# --- 6. Visualize Decision Boundaries (2D Projection) ---

# KNN decision boundaries are typically visualized using only 2 features[cite: 11].
# We'll use PetalLengthCm and PetalWidthCm as they usually provide the clearest separation.

# Extract the two features for visualization (PetalLengthCm, PetalWidthCm are columns 2 and 3)
X_viz = X_train[:, [2, 3]]
y_viz = y_train

# Re-scale only these two features
scaler_viz = StandardScaler()
X_viz_scaled = scaler_viz.fit_transform(X_viz)

# Train a KNN model on the 2D data using the optimal K (e.g., K=5)
knn_viz = KNeighborsClassifier(n_neighbors=K_OPTIMAL)
knn_viz.fit(X_viz_scaled, y_viz)

# Create a mesh grid for plotting
h = .02  # step size in the mesh
x_min, x_max = X_viz_scaled[:, 0].min() - 1, X_viz_scaled[:, 0].max() + 1
y_min, y_max = X_viz_scaled[:, 1].min() - 1, X_viz_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Predict class for each point in the mesh
Z = knn_viz.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Convert string labels to numeric for plotting contours
label_mapping = {label: idx for idx, label in enumerate(np.unique(y_viz))}
Z_numeric = np.array([label_mapping[label] for label in Z.ravel()]).reshape(Z.shape)

# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

plt.figure(figsize=(10, 7))

# Plot the decision boundaries (colored area)
plt.contourf(xx, yy, Z_numeric, cmap=cmap_light, alpha=0.8)

# Plot the training points (using the original string labels for color)
for name, color in zip(label_mapping.keys(), cmap_bold.colors):
    idx = np.where(y_viz == name)
    plt.scatter(X_viz_scaled[idx, 0], X_viz_scaled[idx, 1],
                c=color, label=name, edgecolor='k', s=20)

plt.title(f"KNN Decision Boundary (K={K_OPTIMAL}) on Petal Features")
plt.xlabel("Petal Length (Scaled)")
plt.ylabel("Petal Width (Scaled)")
plt.legend()
plt.show()








