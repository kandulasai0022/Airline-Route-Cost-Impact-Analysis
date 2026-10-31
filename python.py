import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, classification_report
)

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
TEST_SIZE = 0.20

FILE_PATH = "/Users/manikantasai/Downloads/route_cost_impact.csv"
OUTPUT_DIR = "route_cost_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(FILE_PATH)

print("\n========== DATASET OVERVIEW ==========")
print("Shape:", df.shape)
print("\nFirst five rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())

df.columns = df.columns.str.strip()
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].astype("string").str.strip()

df["month"] = pd.to_datetime(df["month"], errors="coerce")
df["year"] = df["month"].dt.year
df["month_number"] = df["month"].dt.month

print("\n========== NUMERICAL SUMMARY ==========")
print(df.describe(include="all").T)

print("\nCancellation counts:")
print(df["flight_cancelled"].value_counts(dropna=False))

print("\nRerouting counts:")
print(df["rerouted"].value_counts(dropna=False))

plt.figure(figsize=(8, 5))
sns.histplot(df["total_ticket_price_usd"].dropna(), bins=35, kde=True)
plt.title("Distribution of Total Ticket Prices")
plt.xlabel("Total Ticket Price (USD)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
plt.savefig(os.path.join(OUTPUT_DIR, "01_ticket_price_distribution.png"), dpi=300)
plt.show()

phase_price = (
    df.groupby("conflict_phase", dropna=False)["total_ticket_price_usd"]
      .mean().sort_values()
)
plt.figure(figsize=(9, 5))
phase_price.plot(kind="barh")
plt.title("Average Ticket Price by Conflict Phase")
plt.xlabel("Average Ticket Price (USD)")
plt.ylabel("Conflict Phase")
plt.tight_layout()
plt.show()
plt.savefig(os.path.join(OUTPUT_DIR, "02_ticket_price_by_phase.png"), dpi=300)
plt.show()

# Jet fuel price and fuel surcharge relationship
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df, x="jet_fuel_usd_barrel", y="fuel_surcharge_usd",
    hue="conflict_phase", legend=False, alpha=0.65
)
plt.title("Jet Fuel Price vs Fuel Surcharge")
plt.xlabel("Jet Fuel Price (USD/barrel)")
plt.ylabel("Fuel Surcharge (USD)")
plt.tight_layout()
plt.show()
plt.savefig(os.path.join(OUTPUT_DIR, "03_fuel_price_vs_surcharge.png"), dpi=300)
plt.show()


def make_preprocessor(X):
    """Build preprocessing using training data's column types."""
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    transformer = ColumnTransformer(transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])

    return transformer


print("\n========== MODEL 1: LINEAR REGRESSION ==========")

regression_features = [
    "year", "month_number", "conflict_phase", "airline",
    "origin_city", "destination_city", "aircraft_type",
    "original_distance_km", "brent_crude_usd", "jet_fuel_usd_barrel"
]

regression_target = "total_ticket_price_usd"

X_reg = df[regression_features].copy()
y_reg = pd.to_numeric(df[regression_target], errors="coerce")

valid_rows = y_reg.notna()
X_reg = X_reg.loc[valid_rows]
y_reg = y_reg.loc[valid_rows]

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_reg, y_reg,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

regression_pipeline = Pipeline(steps=[
    ("preprocessing", make_preprocessor(Xr_train)),
    ("model", LinearRegression())
])

regression_pipeline.fit(Xr_train, yr_train)
regression_predictions = regression_pipeline.predict(Xr_test)

mae = mean_absolute_error(yr_test, regression_predictions)
mse = mean_squared_error(yr_test, regression_predictions)
rmse = np.sqrt(mse)
r2 = r2_score(yr_test, regression_predictions)

print(f"MAE  : ${mae:,.2f}")
print(f"MSE  : {mse:,.2f}")
print(f"RMSE : ${rmse:,.2f}")
print(f"R2   : {r2:.4f}")

plt.figure(figsize=(7, 5))
plt.scatter(yr_test, regression_predictions, alpha=0.65)
lower = min(yr_test.min(), regression_predictions.min())
upper = max(yr_test.max(), regression_predictions.max())
plt.plot([lower, upper], [lower, upper], "k--", label="Perfect prediction")
plt.title("Linear Regression: Actual vs Predicted Ticket Price")
plt.xlabel("Actual Ticket Price (USD)")
plt.ylabel("Predicted Ticket Price (USD)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "04_linear_regression_actual_vs_predicted.png"), dpi=300)
plt.show()


print("\n========== CLASSIFICATION DATA ==========")


classification_features = [
    "year", "month_number", "conflict_phase", "airline", "iata_code",
    "origin_city", "destination_city", "aircraft_type",
    "original_distance_km", "brent_crude_usd", "jet_fuel_usd_barrel"
]

X_class = df[classification_features].copy()

# Convert Yes/No labels safely into 1/0
target_text = df["flight_cancelled"].astype("string").str.strip().str.lower()
y_class = target_text.map({"no": 0, "yes": 1})

# Remove rows with unknown target labels
valid_class_rows = y_class.notna()
X_class = X_class.loc[valid_class_rows]
y_class = y_class.loc[valid_class_rows].astype(int)

print("Class distribution:")
print(y_class.value_counts().rename(index={0: "Not Cancelled", 1: "Cancelled"}))
print("Cancellation percentage:", round(y_class.mean() * 100, 2), "%")

X_train, X_test, y_train, y_test = train_test_split(
    X_class, y_class,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_class
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

def evaluate_classifier(model, model_name, X_test, y_test):
    """Print classification metrics and draw a confusion matrix."""
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    balanced_acc = balanced_accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    roc_auc = roc_auc_score(y_test, probabilities)
    pr_auc = average_precision_score(y_test, probabilities)

    print(f"\n========== {model_name.upper()} ==========")
    print(f"Accuracy          : {accuracy:.4f}")
    print(f"Balanced Accuracy : {balanced_acc:.4f}")
    print(f"Precision         : {precision:.4f}")
    print(f"Recall            : {recall:.4f}")
    print(f"F1-score          : {f1:.4f}")
    print(f"ROC-AUC           : {roc_auc:.4f}")
    print(f"PR-AUC            : {pr_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(
        y_test, predictions,
        labels=[0, 1],
        target_names=["Not Cancelled", "Cancelled"],
        zero_division=0
    ))

    cm = confusion_matrix(y_test, predictions, labels=[0, 1])
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Not Cancelled", "Cancelled"],
        yticklabels=["Not Cancelled", "Cancelled"]
    )
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    safe_name = model_name.lower().replace(" ", "_")
    plt.savefig(os.path.join(OUTPUT_DIR, f"confusion_{safe_name}.png"), dpi=300)
    plt.show()

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Balanced Accuracy": balanced_acc,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc
    }

classification_models = {
    "Logistic Regression": LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=6,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=RANDOM_STATE
    ),

    "K-Nearest Neighbors": KNeighborsClassifier(
        n_neighbors=7,
        weights="distance"
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
}

classification_results = []
trained_classifiers = {}

for model_name, estimator in classification_models.items():

    model_pipeline = Pipeline(steps=[
        ("preprocessing", make_preprocessor(X_train)),
        ("model", estimator)
    ])

    model_pipeline.fit(X_train, y_train)

    trained_classifiers[model_name] = model_pipeline

    result = evaluate_classifier(
        model_pipeline, model_name, X_test, y_test
    )
    classification_results.append(result)


results_df = pd.DataFrame(classification_results)

print("\n========== CLASSIFICATION MODEL COMPARISON ==========")
print(results_df.round(4).to_string(index=False))

results_df.to_csv(
    os.path.join(OUTPUT_DIR, "classification_model_results.csv"),
    index=False
)

metrics_to_plot = [
    "Accuracy", "Balanced Accuracy", "Precision",
    "Recall", "F1-score", "ROC-AUC", "PR-AUC"
]

results_df.set_index("Model")[metrics_to_plot].plot(
    kind="bar", figsize=(12, 6), ylim=(0, 1.05)
)
plt.title("Comparison of Classification Models")
plt.ylabel("Score")
plt.xlabel("")
plt.xticks(rotation=15, ha="right")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "model_comparison.png"), dpi=300)
plt.show()

best_model_row = results_df.sort_values(
    by="F1-score", ascending=False
).iloc[0]

print("\nBest classifier by F1-score:")
print(best_model_row.to_string())

print("\nAll output graphs and results are saved in:", OUTPUT_DIR)
