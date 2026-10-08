# Student Performance — Machine Learning

Machine learning project focused on analysing and predicting academic
performance in higher education.

The project combines supervised and unsupervised learning to study three
different problems:

- Multiclass classification of students' academic status
- Regression to predict second-semester academic performance
- Unsupervised learning to identify groups of students

## Project Overview

The dataset contains information about students' academic background,
personal characteristics and first-semester performance.

Before modelling, the data is cleaned and checked for missing values,
duplicates and potential inconsistencies. The classification target contains
three classes:

- Graduated
- Enrolled
- Dropout

The classification dataset contains 4,403 students.

## Methods

### Classification

Two approaches were compared:

- K-Nearest Neighbors (KNN)
- Multiclass Logistic Regression

The models were evaluated using accuracy, macro F1-score and a confusion
matrix.

KNN was tuned using cross-validation, with **k = 20** selected as the final
configuration.

### Regression

The goal is to predict students' second-semester average grade.

The following approaches were compared:

- KNN Regression
- Linear Regression
- Polynomial Regression
- Spline Regression
- Interaction-based regression

Model selection was performed using cross-validation and the final models
were evaluated on a separate test set.

### Unsupervised Learning

The project also explores student profiles without using the target variable.

The analysis includes:

- Principal Component Analysis (PCA)
- K-Means clustering
- Silhouette analysis
- Inertia analysis

PCA was implemented manually and the number of components was selected based
on explained variance.

## Results

### Classification

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| KNN (k = 20) | 0.741 | 0.641 |
| Multiclass Logistic Regression | 0.734 | 0.587 |

KNN achieved the best overall classification performance, particularly in
macro F1-score.

### Regression

| Model | R² | RMSE | MAE |
|---|---:|---:|---:|
| KNN Regression (k = 27) | 0.644 | 2.577 | 1.422 |
| Polynomial Regression | 0.669 | 2.486 | 1.468 |

Polynomial regression achieved the highest R² and lowest RMSE on the test set.

### Clustering

PCA reduced the original feature space to the number of components required
to explain at least 80% of the variance.

The best clustering solution among the tested values of k was **5 clusters**,
selected using silhouette score as the main criterion and inertia as supporting
evidence.

## Implementation

An important part of the project was implementing several algorithms and
evaluation metrics directly rather than relying entirely on machine learning
libraries.

The repository includes implementations of:

- KNN classification
- KNN regression
- Multiclass logistic regression
- Linear regression
- Accuracy
- Macro F1-score
- MAE
- RMSE
- R²
- PCA
- K-Means
- Silhouette score

## Technologies

**Python · NumPy · Pandas · Matplotlib · Jupyter Notebook**

## Course

**Machine Learning / Aprendizaje Automático**  
Comillas ICAI — 2025/26
