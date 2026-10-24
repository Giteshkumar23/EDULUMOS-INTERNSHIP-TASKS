# House Price Prediction Using Linear Regression

## 1. Project Overview

This project uses machine learning to estimate median house values using the California Housing dataset. A Linear Regression model is trained and evaluated using Python and Scikit-learn.

## 2. Objectives

- Load and inspect housing data.
- Check missing values and duplicate records.
- Perform exploratory data analysis (EDA).
- Visualize distributions and feature correlations.
- Train a Linear Regression model.
- Evaluate model performance on unseen test data.
- Save predictions, evaluation metrics, and visualizations.

## 3. Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Visual Studio Code

## 4. Dataset

The project uses the California Housing dataset provided by Scikit-learn.

- Total observations: 20,640
- Input features: 8
- Target variable: `MedHouseVal`
- Missing values detected: 0
- Duplicate rows detected: 0

The target represents median house value in units of $100,000. The data describes housing characteristics at a regional level rather than individual property listings.

## 5. Methodology

1. Load and inspect the dataset.
2. Check data quality and summarize numerical variables.
3. Explore target distributions and feature correlations.
4. Split the data into training (80%) and testing (20%) sets.
5. Train a Linear Regression model.
6. Evaluate predictions using MAE, RMSE, and R².
7. Save model evaluation results, predictions, coefficients, and charts.

The train-test split uses `random_state=42` for reproducibility.

## 6. Model Results

## Model Performance and Comparison

Two regression models were evaluated using the same 80/20 train-test split and random state of 42.

| Model | MAE | RMSE | R² Score |
|---|---:|---:|---:|
| Linear Regression | 0.5332 | 0.7456 | 0.5758 |
| Random Forest Regressor | 0.3275 | 0.5053 | 0.8051 |

### Results Interpretation

The Random Forest Regressor achieved a higher R² score and lower MAE and RMSE than Linear Regression on the test set. It is therefore the preferred model among the two evaluated in this experiment.

The Random Forest model achieved an R² score of approximately 0.8051, indicating that it explains about 80.5% of the variation in the test targets.

The MAE of 0.3275 corresponds to approximately $32,750 in the dataset's target-value scale. This is an average absolute error, not a guaranteed prediction error for an individual property.

These results come from a single test split. Further cross-validation and testing on other data would be needed to establish how well the model generalizes.

### Generated Outputs

- `metrics.json` — Linear Regression evaluation metrics.
- `model_comparison.csv` — comparison of the two models.
- `predictions.csv` — Linear Regression predictions.
- `random_forest_predictions.csv` — Random Forest predictions.
- `target_distribution.png` — target-value distribution.
- `correlation_heatmap.png` — feature correlation heatmap.
- `actual_vs_predicted.png` — Linear Regression prediction chart.
- `residual_plot.png` — Linear Regression residual chart.
- `random_forest_actual_vs_predicted.png` — Random Forest prediction chart.
- `feature_coefficients.csv` — Linear Regression coefficients.



## 7. Generated Outputs

The `outputs/` directory contains:

- `target_distribution.png`
- `correlation_heatmap.png`
- `actual_vs_predicted.png`
- `residual_plot.png`
- `metrics.json`
- `predictions.csv`
- `feature_coefficients.csv`

## 8. How to Run

### Create and activate a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### Run the project

From the project root:

```powershell
python notebooks/house_price_prediction.py
```

The first run requires an internet connection to fetch the dataset.

## 9. Limitations

- The model assumes a linear relationship between features and the target.
- Regional housing data cannot capture every property-specific characteristic.
- The dataset's target values have an upper limit, which can affect predictions for expensive areas.
- Further feature engineering and comparison with other regression models may improve performance.

## 10. Conclusion

The project demonstrates the end-to-end workflow of a basic regression task, including data inspection, exploratory analysis, model training, evaluation, and output generation. The results establish a baseline for future experimentation.

## 11. Future Improvements

- Compare Linear Regression with Ridge, Random Forest, and other regression models.
- Investigate outliers and residual patterns.
- Perform cross-validation and hyperparameter tuning where appropriate.
- Improve visualizations and document comparative results.

---

Developed as a Data Science internship project for EduLumos.