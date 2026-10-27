
# MLflow Experiment Tracking and Data Drift Analysis

## Overview
This project demonstrates ML model training, experiment tracking, and data drift detection using MLflow and the Iris dataset.

## Objectives
- Train a Random Forest classification model.
- Track model parameters, accuracy, and artifacts using MLflow.
- Simulate data drift in the sepal length feature.
- Detect drift using the Kolmogorov-Smirnov (KS) statistical test.
- Log drift metrics and results in MLflow.

## Dataset
- Dataset: Iris
- Records: 150
- Input features: 4
- Target: Iris species

## Methodology
The dataset is split into training (80%) and test (20%) subsets. A Random Forest classifier is trained, and its accuracy is recorded in MLflow.

Data drift is simulated by adding 1.5 cm to the test subset's sepal length values. The KS two-sample test compares reference and current feature distributions using a significance threshold of 0.05.

## Results
- Model accuracy: 90%
- Features tested: 4
- Features flagged for drift: 1
- Drifted feature: sepal length

The drift is simulated for demonstration purposes and does not represent a real production monitoring result.

## Installation

```bash
python -m pip install -r requirements.txt
```

## Run the project

```bash
python main.py
```

## View MLflow

In another terminal, run:

```bash
python -m mlflow ui --port 5001
```

Open http://127.0.0.1:5001 in your browser.

## Output files
- `drift_results.csv`: KS test results for each feature.
- `drift_plot.png`: Reference and simulated current-data distributions.

## Technologies
Python, MLflow, Pandas, Scikit-learn, SciPy, Matplotlib.