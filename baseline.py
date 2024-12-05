# Databricks notebook source
# MAGIC %md
# MAGIC # Baseline Comparison: Isolation Forest vs ECOD
# MAGIC
# MAGIC This notebook provides a baseline comparison using Isolation Forest to contrast with DAXS's ECOD implementation.
# MAGIC It deliberately avoids the efficient model encoding strategy used by DAXS to demonstrate the performance benefits.

# COMMAND ----------

# MAGIC %pip install -r requirements.txt --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt

def evaluate_results(results_df):
    """
    Evaluate and visualize anomaly detection results.
    
    Args:
        results_df: DataFrame containing 'predict' and 'scores' columns
    """
    n_anomalies = results_df['predict'].sum()
    pct_anomalies = (n_anomalies/len(results_df))*100
    
    print(f"\nResults Summary:")
    print(f"Detected anomalies: {n_anomalies} ({pct_anomalies:.2f}%)")

    # Visualize anomaly scores distribution
    plt.figure(figsize=(12, 6))
    plt.hist(results_df['scores'], bins=50)
    plt.title('Distribution of Anomaly Scores')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Frequency')
    plt.show()
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import time
from datetime import datetime

# Get the current user name
current_user_name = spark.sql("SELECT current_user()").collect()[0][0]

# Set the experiment name
mlflow.set_experiment(f"/Users/{current_user_name}/elevator_anomaly_detection_baseline")

# COMMAND ----------

# Initialize storage 
catalog = "daxs"
db = "default"
volume = "csv"

# Make sure catalog exists
_ = spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
_ = spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{db}")

# COMMAND ----------

# Read training data and filter for first 2 turbines only
spark_df = spark.read.table(f"{catalog}.{db}.turbine_data_train_10000")
spark_df = spark_df.filter("turbine_id IN ('Turbine_1', 'Turbine_2')")
print(f"Total records: {spark_df.count()}")
print("Using turbines: Turbine_1, Turbine_2 for faster testing")

# COMMAND ----------


# MAGIC %md
# MAGIC ## Parallel Training of Individual Models
# MAGIC Train individual models per turbine using multiprocessing for optimal performance

# COMMAND ----------

# Get unique turbine IDs
turbine_ids = pdf['turbine_id'].unique()
print(f"Total turbines: {len(turbine_ids)}")

from multiprocessing import Pool, cpu_count
from functools import partial

def train_turbine_model(turbine_id, data, feature_cols):
    """Train an Isolation Forest model for a single turbine"""
    turbine_data = data[data['turbine_id'] == turbine_id][feature_cols].fillna(0)
    clf = IsolationForest(contamination=0.1, random_state=42, n_jobs=1)
    clf.fit(turbine_data)
    return (turbine_id, clf)

# Train individual models in parallel using multiprocessing
start_time = time.time()

with mlflow.start_run(run_name="isolation_forest_parallel_models"):
    
    mlflow.log_param("contamination", 0.1)
    mlflow.log_param("approach", "multiple_models_parallel")
    mlflow.log_param("n_turbines", len(turbine_ids))
    mlflow.log_param("n_cores", cpu_count())
    
    # Create partial function with fixed arguments
    train_func = partial(train_turbine_model, data=pdf, feature_cols=feature_cols)
    
    # Use multiprocessing pool to train models in parallel
    with Pool(processes=cpu_count()) as pool:
        model_results = pool.map(train_func, turbine_ids)
    
    # Convert results to dictionary
    models = dict(model_results)
    
    training_time = time.time() - start_time
    mlflow.log_metric("training_time", training_time)
    
print(f"Training time: {training_time:.2f} seconds using {cpu_count()} cores")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Performance Comparison
# MAGIC
# MAGIC The baseline approaches demonstrate two key limitations compared to DAXS:
# MAGIC
# MAGIC 1. **Storage Efficiency**: Storing models in memory or as separate files is less efficient than DAXS's encoded Delta table approach
# MAGIC
# MAGIC 2. **Training Time**: Sequential training with for loops is significantly slower than DAXS's parallel processing
# MAGIC
# MAGIC 3. **Scalability**: The baseline approaches don't scale well with increasing numbers of turbines and sensors
# MAGIC
# MAGIC DAXS addresses these limitations through:
# MAGIC - Efficient model encoding in Delta tables
# MAGIC - Parallel processing with Pandas UDFs
# MAGIC - Distributed storage and computation

# COMMAND ----------

# MAGIC %md
# MAGIC © 2024 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License.
