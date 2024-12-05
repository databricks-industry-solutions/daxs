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

# MAGIC %run ./99_utilities

# COMMAND ----------

import numpy as np
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

# Read training data
spark_df = spark.read.table(f"{catalog}.{db}.turbine_data_train_10000")
print(f"Total records: {spark_df.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Baseline Approach 1: Single Model for All Turbines
# MAGIC First let's try training one Isolation Forest model for all turbines

# COMMAND ----------

# Convert to pandas and prepare features
pdf = spark_df.toPandas()
feature_cols = [col for col in pdf.columns if col.startswith('sensor_')]
X = pdf[feature_cols].fillna(0)

# Split data
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# Train single model
start_time = time.time()

with mlflow.start_run(run_name="isolation_forest_single_model"):
    
    # Train model
    clf = IsolationForest(contamination=0.1, random_state=42, n_jobs=-1)
    clf.fit(X_train)
    
    # Get predictions
    y_train_pred = clf.predict(X_train)
    y_train_pred = np.where(y_train_pred == 1, 0, 1)  # Convert to binary labels
    
    # Log parameters and metrics
    mlflow.log_param("contamination", 0.1)
    mlflow.log_param("approach", "single_model")
    mlflow.log_metric("training_time", time.time() - start_time)
    
    # Log model
    signature = infer_signature(X_train, y_train_pred)
    mlflow.sklearn.log_model(clf, "isolation_forest_model", signature=signature)
    
    # Register model
    model_name = f"{catalog}.{db}.IsolationForest_Baseline_{current_user_name[:4]}"
    model_version = mlflow.register_model(
        f"runs:/{mlflow.active_run().info.run_id}/isolation_forest_model",
        model_name
    )
    
    # Set as champion
    client = mlflow.tracking.MlflowClient()
    client.set_registered_model_alias(
        name=model_name,
        alias="Champion",
        version=model_version.version
    )

print(f"Training time: {time.time() - start_time:.2f} seconds")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Baseline Approach 2: Individual Models with For Loop
# MAGIC Now let's try training individual models per turbine using a simple for loop

# COMMAND ----------

# Get unique turbine IDs
turbine_ids = pdf['turbine_id'].unique()
print(f"Total turbines: {len(turbine_ids)}")

# Train individual models using for loop
start_time = time.time()
models = {}

with mlflow.start_run(run_name="isolation_forest_multiple_models"):
    
    mlflow.log_param("contamination", 0.1)
    mlflow.log_param("approach", "multiple_models_loop")
    mlflow.log_param("n_turbines", len(turbine_ids))
    
    for turbine_id in turbine_ids:
        # Get data for this turbine
        turbine_data = pdf[pdf['turbine_id'] == turbine_id][feature_cols].fillna(0)
        
        # Train model
        clf = IsolationForest(contamination=0.1, random_state=42)
        clf.fit(turbine_data)
        
        # Store model in dictionary
        models[turbine_id] = clf
        
    training_time = time.time() - start_time
    mlflow.log_metric("training_time", training_time)
    
print(f"Training time: {training_time:.2f} seconds")

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
