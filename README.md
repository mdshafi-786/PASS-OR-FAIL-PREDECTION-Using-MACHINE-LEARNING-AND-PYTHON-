# Pass or Fail Prediction Using Machine Learning and Python

## Overview

This project predicts whether a student will **Pass or Fail** based on their study habits and academic performance using Machine Learning techniques in Python. It demonstrates a complete data science workflow — from data cleaning and exploratory analysis to training and evaluating multiple models.

## Dataset Description

The dataset (`student_data.csv`) contains **24 student records** with the following features:

| Column           | Description                                    |
|------------------|------------------------------------------------|
| `StudyHours`     | Number of hours the student studied per day    |
| `PreviousResult` | Score (%) from the previous exam               |
| `Attendance`     | Attendance percentage (0–100)                  |
| `FinalMarks`     | Final exam marks (target variable)             |

> The dataset intentionally includes missing values, outliers, and invalid entries (e.g., `"five"`, negative values, attendance > 100) to demonstrate real-world data cleaning.

**Pass/Fail threshold:** A student is classified as **PASS** if `FinalMarks >= 50`, otherwise **FAIL**.

## Tech Stack

- **Language:** Python 3.x
- **Libraries:** pandas, NumPy, scikit-learn, matplotlib, seaborn, SciPy
- **Environment:** Jupyter Notebook

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/mdshafi-786/PASS-OR-FAIL-PREDECTION-Using-MACHINE-LEARNING-AND-PYTHON-.git
   cd PASS-OR-FAIL-PREDECTION-Using-MACHINE-LEARNING-AND-PYTHON-
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook "PASS OR FAIL  PREDECTION.ipynb"
   ```

4. **Run all cells** from top to bottom (Kernel → Restart & Run All).

## Project Structure

```
├── PASS OR FAIL  PREDECTION.ipynb   # Main notebook
├── student_data.csv                  # Dataset
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
```

## Notebook Sections

1. **Import Libraries** — All required libraries imported upfront  
2. **Load & Explore Data** — Load CSV, inspect shape, types, and statistics  
3. **Data Cleaning & Preprocessing** — Handle missing values, fix invalid data, cap outliers  
4. **Exploratory Data Analysis (EDA)** — Correlation heatmap, distributions, pairplots, boxplots  
5. **Feature Selection & Train/Test Split** — Prepare features `X` and target `y`  
6. **Model Training (Linear Regression)** — Train and predict with Linear Regression  
7. **Regression Evaluation Metrics** — MAE, RMSE, R² Score  
8. **Cross-Validation** — 5-fold CV R² scores  
9. **Pass/Fail Classification** — Compare Logistic Regression, Decision Tree, Random Forest, KNN, SVM  
10. **User Input Prediction** — Interactive cell for custom student predictions  
11. **Conclusion** — Summary of findings and best-performing model  

## Results Summary

| Metric            | Linear Regression |
|-------------------|:-----------------:|
| MAE               | ~2–4              |
| RMSE              | ~3–5              |
| R² Score          | ~0.90+            |
| Cross-Val Mean R² | ~0.85+            |

Among the classification models, **Random Forest** typically achieves the highest accuracy on this dataset.

## Usage — Interactive Prediction

Run the **User Input Prediction** cell in the notebook and enter:
```
Enter Study Hours: 7
Enter Previous Result: 85
Enter Attendance %: 90
```
Output:
```
Predicted Final Marks: 88.xx
Result: ✅ PASS
```

## License

This project is open-source and available for educational purposes.
