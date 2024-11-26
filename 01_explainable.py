# Databricks notebook source
# MAGIC %md
# MAGIC # Elevator Predictive Maintenance Dataset: Anomaly Detection
# MAGIC
# MAGIC This notebook demonstrates the use of the Elevator Predictive Maintenance Dataset from Huawei German Research Center for anomaly detection. We'll use the ECOD (Empirical Cumulative Distribution Functions for Outlier Detection) algorithm from the PyOD library.
# MAGIC
# MAGIC Dataset details:
# MAGIC - Contains operation data from IoT sensors for predictive maintenance in the elevator industry.
# MAGIC - Timeseries data sampled at 4Hz during high-peak and evening elevator usage (16:30 to 23:30).
# MAGIC - Includes data from electromechanical sensors (Door Ball Bearing Sensor), ambiance (Humidity), and physics (Vibration).
# MAGIC
# MAGIC Source: [Kaggle - Elevator Predictive Maintenance Dataset](https://www.kaggle.com/datasets/shivamb/elevator-predictive-maintenance-dataset)

# COMMAND ----------

# MAGIC %pip install -r requirements.txt --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./99_utilities

# COMMAND ----------

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from pyod.models.ecod import ECOD
from sklearn.metrics import precision_score, recall_score, f1_score
from pyspark.sql.functions import current_user


# Get the current user name and store it in a variable
current_user_name = spark.sql("SELECT current_user()").collect()[0][0]

# Set the experiment name
mlflow.set_experiment(f"/Users/{current_user_name}/elevator_anomaly_detection")

# COMMAND ----------

catalog = "daxs"
db = "default"
volume = "csv"

# Make sure that the catalog exists
_ = spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")

# Make sure that the schema exists
_ = spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{db}")

# Make sure that the volume exists
_ = spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{db}.{volume}")

# COMMAND ----------

import subprocess
import kagglehub

path = kagglehub.dataset_download("shivamb/elevator-predictive-maintenance-dataset", force_download=True)
bash = f"""mv {path}/predictive-maintenance-dataset.csv /Volumes/{catalog}/{db}/{volume}/predictive-maintenance-dataset.csv"""
process = subprocess.Popen(bash, shell=True, executable='/bin/bash')
process.wait()

# COMMAND ----------

# MAGIC %md
# MAGIC # Data Loading and Exploratory Data Analysis (EDA)

# COMMAND ----------

# Load the data
df = spark.read.csv(f"/Volumes/{catalog}/{db}/{volume}/predictive-maintenance-dataset.csv", header=True, inferSchema=True).toPandas()
df = df.drop(columns=["ID"])
print(f"Dataset shape: {df.shape}")
display(df.head())

# COMMAND ----------

# Basic information about the dataset
df.info()

# COMMAND ----------

# Statistical summary of the dataset
display(df.describe())

# COMMAND ----------

# MAGIC %md
# MAGIC # Data Preprocessing and Feature Engineering

# COMMAND ----------

# Handle missing values
X = df.fillna(-99)

print("Preprocessed data shape:", X.shape)
display(X.head())

# COMMAND ----------

# MAGIC %md
# MAGIC # Model Training and Evaluation

# COMMAND ----------

# Split the data
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# Train the ECOD model
with mlflow.start_run(run_name="ECOD_model") as run:

    clf = ECOD(contamination=0.1, n_jobs=-1)
    clf.fit(X_train)
    clf.feature_columns_ = X_train.columns.tolist()

    # Predict on both training and test sets
    y_train_pred = clf.labels_
    y_test_pred = clf.predict(X_test)
    y_test_scores = clf.decision_function(X_test)

    # Log parameters
    mlflow.log_param("contamination", 0.1)
    mlflow.log_param("n_jobs", -1)

    # Log metrics
    train_auc = synthetic_auc(clf, X_train)
    test_auc = synthetic_auc(clf, X_test)
    mlflow.log_metric("train_auc", train_auc)
    mlflow.log_metric("test_auc", test_auc)

    # Log model
    signature = infer_signature(X_test, y_test_pred) 
    mlflow.sklearn.log_model(clf, "ecod_model", signature=signature)

    # Register the model
    model_name = f"{catalog}.{db}.ECOD_Anomaly_Detection"

    model_version = mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/ecod_model", model_name)

    # Set this version as the Champion model, using its model alias
    client = mlflow.tracking.MlflowClient()
    client.set_registered_model_alias(
        name=model_name,
        alias="Champion",
        version=model_version.version
    )

    print(f"Model {model_name} version {model_version.version} is now in production")

# COMMAND ----------

# Load the champion model
loaded_model = mlflow.pyfunc.load_model(f"models:/{model_name}@champion")
print(f"Loaded the champion model: {model_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Results and Evaluation

# COMMAND ----------

# Evaluate results on training and test sets
df_train_results = evaluate_results(X_train, y_train_pred, clf, "Training")
df_test_results = evaluate_results(X_test, y_test_pred, clf, "Test")

# COMMAND ----------

# Identify the most and least anomalous test samples
most_anomalous_index = np.argmax(y_test_scores)
least_anomalous_index = np.argmin(y_test_scores)

# Feature names (assuming X_test is a DataFrame)
feature_names = X_test.columns.tolist()

# Generate explain_outlier plots for the most anomalous test sample
print("Most Anomalous Test Sample:")
explain_test_outlier(clf, X_test, most_anomalous_index, feature_names=feature_names)

# Generate explain_outlier plots for the least anomalous test sample
print("Least Anomalous Test Sample:")
explain_test_outlier(clf, X_test, least_anomalous_index, feature_names=feature_names)

# COMMAND ----------

train_auc = synthetic_auc(clf, X_train)
test_auc = synthetic_auc(clf, X_test)

print(f"Training AUC: {train_auc:.4f}")
print(f"Testing AUC: {test_auc:.4f}")

# COMMAND ----------

explanations = explainer(clf, X_test, training=False, explanation_num=3)
explanations.sort_values('scores', ascending=False)

# COMMAND ----------

# MAGIC %md
# MAGIC # Conclusion and Next Steps
# MAGIC
# MAGIC In this notebook, we've performed anomaly detection on the Elevator Predictive Maintenance Dataset using the ECOD algorithm. Here are the key findings and potential next steps:
# MAGIC
# MAGIC 1. We successfully identified anomalies in both the training and test sets, with approximately 10% of data points flagged as anomalies.
# MAGIC 2. The distribution of anomaly scores shows a clear separation between normal and anomalous data points.
# MAGIC 3. We've added functionality to visualize the dimensional outlier graphs for the most and least anomalous cases in the test set, providing more insights into the nature of the anomalies.
# MAGIC
# MAGIC Next steps to improve the model and gain more insights:
# MAGIC
# MAGIC 1. Experiment with different contamination rates to fine-tune the anomaly detection threshold.
# MAGIC 2. Try other anomaly detection algorithms (e.g., Isolation Forest, Local Outlier Factor) and compare their performance.
# MAGIC 3. Perform time series analysis to identify temporal patterns in anomalies.
# MAGIC 4. Investigate the root causes of detected anomalies by analyzing the feature values of anomalous data points.
# MAGIC 5. Develop a real-time anomaly detection system for continuous monitoring of elevator performance.
# MAGIC 6. Collaborate with domain experts to validate the detected anomalies and refine the model based on their feedback.
# MAGIC 7. Analyze the dimensional outlier graphs to identify which features contribute most to the anomalies and use this information for feature selection or engineering.

# COMMAND ----------

# MAGIC %md
# MAGIC This concludes our analysis of the Elevator Predictive Maintenance Dataset using anomaly detection techniques. The insights gained from this analysis, including the visualization of the most and least anomalous cases, can be used to improve elevator maintenance strategies and reduce unplanned stops.

# COMMAND ----------

# MAGIC %md
# MAGIC # Synthetic AUC Calculation

# COMMAND ----------

from sklearn.metrics import roc_auc_score
import numpy as np

# Generate synthetic labels for demonstration purposes
# Note: In a real scenario, you would use actual labeled data
np.random.seed(42)
y_true = np.random.randint(0, 2, size=len(y_test_scores))

# Calculate AUC
auc = roc_auc_score(y_true, y_test_scores)
print(f"Synthetic AUC: {auc:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC Note: This AUC score is based on synthetic labels and is for demonstration purposes only. In a real-world scenario, you would need actual labeled data to calculate a meaningful AUC score for anomaly detection.

# COMMAND ----------

# MAGIC %md
# MAGIC © 2024 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries and dataset are subject to the licenses set forth below.
# MAGIC
# MAGIC | library / datas                        | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | pyod | A Comprehensive and Scalable Python Library for Outlier Detection (Anomaly Detection) | BSD License | https://pypi.org/project/pyod/
# MAGIC | kagglehub | Access Kaggle resources anywhere | Apache 2.0 | https://pypi.org/project/kagglehub/
# MAGIC | predictive-maintenance-dataset.csv | predictive-maintenance-dataset.csv | CC0 1.0 | https://zenodo.org/records/3653909
