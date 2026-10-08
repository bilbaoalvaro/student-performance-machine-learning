# Student Performance — Machine Learning

Machine learning project focused on analysing and predicting academic performance in higher education.

The project studies three complementary problems:

- **Multiclass classification:** predicting students' academic status (`abandono`, `graduado`, `matriculado`).
- **Regression:** predicting second-semester academic performance.
- **Unsupervised learning:** exploring groups of students using PCA and K-Means.

## Dataset

The dataset contains information about students' academic background, personal characteristics and first-semester performance.

The classification task uses **4,403 students** and three target classes.

## Methods

### Classification

Two approaches are compared:

- K-Nearest Neighbors (KNN)
- Multiclass Logistic Regression

Models are evaluated using accuracy, macro F1-score and a confusion matrix. KNN is tuned using cross-validation, with **k = 20** selected for the final configuration.

### Regression

The regression task predicts students' second-semester average grade.

The project compares:

- KNN Regression
- Linear Regression
- Polynomial Regression
- Other regression variants explored during model selection

Cross-validation is used during model selection, and the final models are evaluated on a separate test set.

### Unsupervised Learning

The project also explores student profiles without using the target variable.

This part includes:

- Principal Component Analysis (PCA)
- K-Means clustering
- Silhouette analysis
- Inertia / elbow analysis

PCA and K-Means are implemented within the project using NumPy-based code.

## Results

### Classification

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| KNN (k = 20) | 0.741 | 0.641 |
| Multiclass Logistic Regression | 0.734 | 0.587 |

KNN gives the stronger result on the test set, particularly in macro F1-score.

### Regression

| Model | R² | RMSE | MAE |
|---|---:|---:|---:|
| KNN Regression (k = 27) | 0.644 | 2.577 | 1.422 |
| Polynomial Regression | 0.669 | 2.486 | 1.468 |

Polynomial regression achieves the highest R² and lowest RMSE among the final models compared in the project.

### Clustering

The clustering analysis tests values of k from 2 to 6. The solution with **5 clusters** obtains the highest silhouette score among the tested values (`0.265`) and is selected as the final clustering solution.

## Implementation

The repository contains course implementations and adaptations of the main algorithms and evaluation metrics used in the project, including:

- KNN classification and regression
- Multiclass logistic regression
- Linear regression
- Accuracy and macro F1
- MAE, RMSE and R²
- PCA
- K-Means
- Silhouette score

## Repository structure

```text
student-performance-machine-learning/
├── data/
│   ├── README.md
│   └── rendimiento_estudiantes.csv
├── docs/
│   └── project_report.pdf
├── src/
│   ├── __init__.py
│   ├── knn.py
│   ├── knn_regression.py
│   ├── logistic_regression.py
│   ├── metrics.py
│   └── regression.py
├── student_performance_analysis.ipynb
├── .gitignore
├── README.md
└── requirements.txt
```

## How to run

1. Install the dependencies:

```bash
pip install -r requirements.txt
```

2. Open `student_performance_analysis.ipynb` from the repository root.

3. Run the notebook cells in order.

## Course

**Machine Learning / Aprendizaje Automático**  
Comillas ICAI · 2025/26
