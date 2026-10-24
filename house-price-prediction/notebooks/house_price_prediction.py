
from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# 1. Project paths and settings
# --------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")
RANDOM_STATE = 42


# --------------------------------------------------
# 2. Load the housing dataset
# --------------------------------------------------
print("\n1. Loading California Housing dataset...")

housing = fetch_california_housing(as_frame=True)
df = housing.frame.copy()

print(f"Dataset shape: {df.shape}")
print("\nFirst five rows:")
print(df.head())


# --------------------------------------------------
# 3. Inspect and clean the data
# --------------------------------------------------
print("\n2. Checking data quality...")

print("\nMissing values:")
print(df.isnull().sum())

duplicate_count = int(df.duplicated().sum())
print(f"\nDuplicate rows: {duplicate_count}")

df = df.drop_duplicates().dropna().copy()

print(f"Rows after cleaning: {len(df)}")

print("\nSummary statistics:")
print(df.describe().round(2))


# --------------------------------------------------
# 4. Separate features and target
# --------------------------------------------------
target = "MedHouseVal"

X = df.drop(columns=[target])
y = df[target]

print("\n3. Features:", list(X.columns))
print("Target:", target)


# --------------------------------------------------
# 5. Exploratory data analysis
# --------------------------------------------------
print("\n4. Creating exploratory analysis charts...")

# Target distribution
plt.figure(figsize=(9, 5))
sns.histplot(y, bins=40, kde=True)
plt.title("Distribution of Median House Values")
plt.xlabel("Median house value (units of $100,000)")
plt.ylabel("Number of observations")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "target_distribution.png", dpi=150)
plt.close()

# Feature correlation heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)
plt.title("Housing Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=150)
plt.close()


# --------------------------------------------------
# 6. Split the data
# --------------------------------------------------
print("\n5. Splitting data into training and test sets...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE
)

print(f"Training rows: {len(X_train)}")
print(f"Testing rows: {len(X_test)}")


# --------------------------------------------------
# 7. Train Linear Regression
# --------------------------------------------------
print("\n6. Training Linear Regression model...")

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)


# --------------------------------------------------
# 8. Evaluate the model
# --------------------------------------------------
print("\n7. Evaluating model performance...")

mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

metrics = {
    "model": "Linear Regression",
    "dataset": "California Housing",
    "random_state": RANDOM_STATE,
    "test_size": 0.20,
    "test_samples": int(len(y_test)),
    "MAE": float(mae),
    "RMSE": float(rmse),
    "R2": float(r2)
}

for name, value in metrics.items():
    if isinstance(value, float):
        print(f"{name}: {value:.4f}")
    else:
        print(f"{name}: {value}")

with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=4)


# --------------------------------------------------
# 9. Save predictions
# --------------------------------------------------
results = pd.DataFrame({
    "Actual_Value_100k_USD": y_test.to_numpy(),
    "Predicted_Value_100k_USD": predictions
})

results.to_csv(OUTPUT_DIR / "predictions.csv", index=False)


# --------------------------------------------------
# 10. Actual vs predicted chart
# --------------------------------------------------
print("\n8. Creating evaluation charts...")

plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=predictions, alpha=0.45)

lower = min(y_test.min(), predictions.min())
upper = max(y_test.max(), predictions.max())

plt.plot([lower, upper], [lower, upper], linestyle="--")
plt.xlabel("Actual value (units of $100,000)")
plt.ylabel("Predicted value (units of $100,000)")
plt.title("Actual vs Predicted House Values")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "actual_vs_predicted.png", dpi=150)
plt.close()


# --------------------------------------------------
# 11. Residual analysis
# --------------------------------------------------
residuals = y_test.to_numpy() - predictions

plt.figure(figsize=(8, 6))
sns.scatterplot(x=predictions, y=residuals, alpha=0.45)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted value (units of $100,000)")
plt.ylabel("Residual (actual - predicted)")
plt.title("Residual Analysis")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "residual_plot.png", dpi=150)
plt.close()


# --------------------------------------------------
# 12. Feature coefficients
# --------------------------------------------------
coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

coefficients["Absolute_Coefficient"] = coefficients["Coefficient"].abs()
coefficients = coefficients.sort_values(
    "Absolute_Coefficient", ascending=False
)

coefficients.to_csv(OUTPUT_DIR / "feature_coefficients.csv", index=False)

print("\nFeature coefficients:")
print(coefficients[["Feature", "Coefficient"]].to_string(index=False))


# --------------------------------------------------
# 12. Compare Linear Regression with Random Forest
# --------------------------------------------------
from sklearn.ensemble import RandomForestRegressor

print("\n9. Comparing regression models...")

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
import joblib

# Create the model directory if it doesn't exist
model_dir = PROJECT_DIR / "model"
model_dir.mkdir(parents=True, exist_ok=True)

# Save the trained Random Forest model
model_path = model_dir / "random_forest_model.joblib"
joblib.dump(rf_model, model_path)

print(f"Random Forest model saved to: {model_path}")
rf_predictions = rf_model.predict(X_test)

comparison = pd.DataFrame([
    {
        "Model": "Linear Regression",
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": mean_squared_error(y_test, predictions) ** 0.5,
        "R2": r2_score(y_test, predictions)
    },
    {
        "Model": "Random Forest",
        "MAE": mean_absolute_error(y_test, rf_predictions),
        "RMSE": mean_squared_error(y_test, rf_predictions) ** 0.5,
        "R2": r2_score(y_test, rf_predictions)
    }
])

comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

print("\nModel comparison:")
print(comparison.round(4).to_string(index=False))


# Save Random Forest predictions
rf_results = pd.DataFrame({
    "Actual_Value_100k_USD": y_test.to_numpy(),
    "Predicted_Value_100k_USD": rf_predictions
})

rf_results.to_csv(
    OUTPUT_DIR / "random_forest_predictions.csv",
    index=False
)

# Plot actual versus predicted values
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=rf_predictions, alpha=0.4)

lower = min(y_test.min(), rf_predictions.min())
upper = max(y_test.max(), rf_predictions.max())

plt.plot([lower, upper], [lower, upper], linestyle="--")
plt.xlabel("Actual value (units of $100,000)")
plt.ylabel("Predicted value (units of $100,000)")
plt.title("Random Forest: Actual vs Predicted")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "random_forest_actual_vs_predicted.png",
    dpi=150
)
plt.close()

print("\nRandom Forest predictions and chart saved.")




# --------------------------------------------------
# 13. Finish
# --------------------------------------------------
print("\nProject run completed successfully!")
print(f"Results and charts saved in: {OUTPUT_DIR}")
