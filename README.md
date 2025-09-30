# AI & ML Internship - Task 6: K-Nearest Neighbors (KNN) Classification

## 🎯 Objective
This task involved implementing the **K-Nearest Neighbors (KNN)** classification algorithm on the classic **Iris Dataset**. The goal was to understand KNN's mechanics, perform essential data preprocessing (normalization), find an optimal hyperparameter, and evaluate the model's performance.

---

## 🛠️ Project Details

### Tools & Libraries
* **Python**
* **Scikit-learn (sklearn):** Used for the `KNeighborsClassifier`, data splitting, scaling, and evaluation metrics.
* **Pandas:** For data loading and manipulation.
* **Matplotlib:** For visualization (Elbow Plot and Decision Boundaries).

### Dataset
The project utilizes the **Iris Dataset**.

| Feature | Description |
| :--- | :--- |
| `SepalLengthCm` | Sepal length in cm |
| `SepalWidthCm` | Sepal width in cm |
| `PetalLengthCm` | Petal length in cm |
| `PetalWidthCm` | Petal width in cm |
| `Species` | The target class (`Iris-setosa`, `Iris-versicolor`, `Iris-virginica`) |

---

## 🚀 Implementation Steps

1.  **Data Preparation:** Loaded the `Iris.csv` file and performed necessary cleaning (e.g., dropping the `Id` column). The data was split into features (X) and the target (y).
2.  **Feature Normalization:** Applied **Normalization** (`StandardScaler` or similar) to the feature set. This step is crucial for KNN as it relies on distance calculations.
3.  **K Selection (Elbow Method):** Experimented with different values of and plotted the error rate versus $K$ to identify the optimal value that minimizes the classification error on the test set.
4.  **Model Training & Evaluation:** Trained the final `KNeighborsClassifier` using the optimal $K$. Performance was evaluated using **Accuracy** and a **Confusion Matrix**.
5.  **Visualization:** Created a plot to visualize the model's **decision boundaries** using two features (Petal Length and Petal Width) to demonstrate how the classifier separates the classes.

---

## 🧠 Key Concepts & Interview Prep

### Key Learning Points
* **Instance-based learning**: KNN is a lazy learning algorithm that defers generalization until a query is made.
* **Euclidean distance**: The standard metric used to find the "nearest" neighbors.
* **K selection**: Choosing the right $K$ balances the trade-off between model bias and variance.

### Interview Questions

| Question | Answer Summary |
| :--- | :--- |
| **How does the KNN algorithm work?** | It classifies a new point by finding the **$K$ nearest training data points** (via distance) and assigning the class by **majority vote**. |
| **How do you choose the right K?** | By **experimenting** with different values, often using an **Elbow Plot** to visualize the trade-off between error rate and complexity. |
| **Why is normalization important in KNN?** | KNN is distance-based, so normalization ensures that features with larger numerical ranges do not **dominate** the distance calculation. |
| **What is the time complexity of KNN?** | **Prediction time** is the bottleneck: $O(n \cdot d)$, where $n$ is training samples and $d$ is features, as it calculates distance to every point. |
| **What are pros and cons of KNN?** | **Pros:** Simple, effective for non-linear data, no training time. **Cons:** Slow prediction for large data, sensitive to scale and noise. |
| **Is KNN sensitive to noise?** | **Yes**, especially with a small $K$. Outliers can heavily skew the majority vote for nearby test points. |
| **How does KNN handle multi-class problems?** | By taking the **majority vote** among the $K$ nearest neighbors to assign the class, regardless of the number of available classes. |
| **What's the role of distance metrics in KNN?** | They quantify the dissimilarity between data points (e.g., Euclidean or Manhattan) and are essential for determining the "nearest" neighbors. |
