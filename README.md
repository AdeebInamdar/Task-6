# AI & ML Internship - Task 6: K-Nearest Neighbors (KNN) Classification

## 🎯 Objective
[cite_start]This task involved implementing the **K-Nearest Neighbors (KNN)** classification algorithm [cite: 4, 5] [cite_start]on the classic **Iris Dataset**[cite: 12]. [cite_start]The goal was to understand KNN's mechanics, perform essential data preprocessing (normalization), find an optimal hyperparameter $K$ [cite: 9][cite_start], and evaluate the model's performance[cite: 10].

---

## 🛠️ Project Details

### Tools & Libraries
* **Python**
* [cite_start]**Scikit-learn (sklearn):** Used for the `KNeighborsClassifier` [cite: 8][cite_start], data splitting, scaling, and evaluation metrics[cite: 10].
* [cite_start]**Pandas:** For data loading and manipulation[cite: 5].
* [cite_start]**Matplotlib:** For visualization (Elbow Plot and Decision Boundaries)[cite: 5, 11].

### Dataset
[cite_start]The project utilizes the **Iris Dataset**[cite: 12].

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
2.  [cite_start]**Feature Normalization:** Applied **Normalization** (`StandardScaler` or similar) to the feature set[cite: 7]. This step is crucial for KNN as it relies on distance calculations.
3.  [cite_start]**K Selection (Elbow Method):** Experimented with different values of $K$ [cite: 9] and plotted the error rate versus $K$ to identify the optimal value that minimizes the classification error on the test set.
4.  **Model Training & Evaluation:** Trained the final `KNeighborsClassifier` using the optimal $K$. [cite_start]Performance was evaluated using **Accuracy** and a **Confusion Matrix**[cite: 10].
5.  [cite_start]**Visualization:** Created a plot to visualize the model's **decision boundaries** [cite: 11] using two features (Petal Length and Petal Width) to demonstrate how the classifier separates the classes.

---

## 🧠 Key Concepts & Interview Prep

### Key Learning Points
* [cite_start]**Instance-based learning**[cite: 13]: KNN is a lazy learning algorithm that defers generalization until a query is made.
* [cite_start]**Euclidean distance**[cite: 13]: The standard metric used to find the "nearest" neighbors.
* [cite_start]**K selection**[cite: 13]: Choosing the right $K$ balances the trade-off between model bias and variance.

### Interview Questions

| Question | Answer Summary |
| :--- | :--- |
| [cite_start]**How does the KNN algorithm work?** [cite: 15] | [cite_start]It classifies a new point by finding the **$K$ nearest training data points** (via distance) and assigning the class by **majority vote**[cite: 15]. |
| [cite_start]**How do you choose the right K?** [cite: 16] | [cite_start]By **experimenting** with different values, often using an **Elbow Plot** to visualize the trade-off between error rate and complexity[cite: 9]. |
| [cite_start]**Why is normalization important in KNN?** [cite: 17] | [cite_start]KNN is distance-based, so normalization ensures that features with larger numerical ranges do not **dominate** the distance calculation[cite: 17]. |
| [cite_start]**What is the time complexity of KNN?** [cite: 18] | [cite_start]**Prediction time** is the bottleneck: $O(n \cdot d)$, where $n$ is training samples and $d$ is features, as it calculates distance to every point[cite: 18]. |
| [cite_start]**What are pros and cons of KNN?** [cite: 19] | **Pros:** Simple, effective for non-linear data, no training time. [cite_start]**Cons:** Slow prediction for large data, sensitive to scale and noise[cite: 19]. |
| [cite_start]**Is KNN sensitive to noise?** [cite: 20] | **Yes**, especially with a small $K$. [cite_start]Outliers can heavily skew the majority vote for nearby test points[cite: 20]. |
| [cite_start]**How does KNN handle multi-class problems?** [cite: 21] | [cite_start]By taking the **majority vote** among the $K$ nearest neighbors to assign the class, regardless of the number of available classes[cite: 21]. |
| [cite_start]**What's the role of distance metrics in KNN?** [cite: 22] | [cite_start]They quantify the dissimilarity between data points (e.g., Euclidean or Manhattan) and are essential for determining the "nearest" neighbors[cite: 22]. |
