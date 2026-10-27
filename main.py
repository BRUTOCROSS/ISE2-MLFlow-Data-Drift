
import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import ks_2samp
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Load the Iris dataset
iris = load_iris(as_frame=True)
data = iris.data
target = iris.target

print("Dataset loaded successfully.")
print(f"Total records: {len(data)}")
print(f"Features: {list(data.columns)}")

# 2. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    data,
    target,
    test_size=0.2,
    random_state=42,
    stratify=target
)

# 3. Configure MLflow experiment
mlflow.set_experiment("Iris_Data_Drift_Analysis")

# 4. Train the model and track the experiment
with mlflow.start_run(run_name="RandomForest_Baseline"):
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    mlflow.log_param("model", "RandomForestClassifier")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("training_rows", len(X_train))
    mlflow.log_metric("test_rows", len(X_test))

    # Explicitly trust the tree type used by our own trained model
    mlflow.sklearn.log_model(
        model,
        name="iris_model",
        skops_trusted_types=["sklearn.tree._tree.Tree"]
    )

    print(f"\nModel accuracy: {accuracy:.4f}")
    print("Model and experiment logged successfully.")

# 5. Prepare reference and current datasets
reference_data = X_train.copy()
current_data = X_test.copy()

# Simulate data drift by shifting sepal length
current_data["sepal length (cm)"] = (
    current_data["sepal length (cm)"] + 1.5
)

print("\nSimulated drift in sepal length.")

# 6. Detect data drift using the Kolmogorov-Smirnov test
drift_results = []

for feature in reference_data.columns:
    statistic, p_value = ks_2samp(
        reference_data[feature],
        current_data[feature]
    )

    drift_detected = p_value < 0.05

    drift_results.append({
        "feature": feature,
        "ks_statistic": statistic,
        "p_value": p_value,
        "drift_detected": drift_detected
    })

    print(
        f"{feature}: KS={statistic:.4f}, "
        f"p-value={p_value:.6f}, "
        f"drift={drift_detected}"
    )

# 7. Save drift results
results_df = pd.DataFrame(drift_results)
results_df.to_csv("drift_results.csv", index=False)

# 8. Track drift results in MLflow
with mlflow.start_run(run_name="Data_Drift_Detection"):
    mlflow.log_param(
        "drift_test",
        "Kolmogorov-Smirnov two-sample test"
    )
    mlflow.log_param("significance_level", 0.05)
    mlflow.log_param("drift_simulation", "Sepal length shifted by 1.5 cm")

    drift_count = int(results_df["drift_detected"].sum())
    mlflow.log_metric("features_with_drift", drift_count)

    for _, row in results_df.iterrows():
        safe_feature = (
            row["feature"]
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
        )

        mlflow.log_metric(
            f"{safe_feature}_ks_statistic",
            float(row["ks_statistic"])
        )
        mlflow.log_metric(
            f"{safe_feature}_p_value",
            float(row["p_value"])
        )

    mlflow.log_artifact("drift_results.csv")

# 9. Plot reference and current distributions
feature = "sepal length (cm)"

plt.figure(figsize=(8, 5))

plt.hist(
    reference_data[feature],
    bins=12,
    alpha=0.6,
    label="Reference data"
)

plt.hist(
    current_data[feature],
    bins=12,
    alpha=0.6,
    label="Current data (simulated drift)"
)

plt.xlabel(feature)
plt.ylabel("Frequency")
plt.title("Data Drift Detection - Iris Dataset")
plt.legend()
plt.tight_layout()
plt.savefig("drift_plot.png")
plt.close()

# 10. Finish
print("\n---------------- RESULTS ----------------")
print(f"Model accuracy: {accuracy:.4f}")
print(f"Features with detected drift: {drift_count}")
print("Results saved to drift_results.csv")
print("Plot saved to drift_plot.png")
print("MLflow experiment tracking completed.")
print("------------------------------------------")