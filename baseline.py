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

# COMMAND ----------

# Define catalog and schema
catalog = "daxs"
db = "default"

# Read training data and filter for first 2 turbines only
spark_df = spark.table(f"{catalog}.{db}.turbine_data_train_10000")
spark_df = spark_df.filter("turbine_id IN ('Turbine_1', 'Turbine_2')")
print(f"Total records: {spark_df.count()}")
print("Using turbines: Turbine_1, Turbine_2 for faster testing")

# COMMAND ----------


# MAGIC %md
# MAGIC ## Parallel Training of Individual Models
# MAGIC Train individual models per turbine using multiprocessing for optimal performance

# COMMAND ----------

# Get unique turbine IDs and define feature columns
pdf = spark_df.toPandas()
turbine_ids = pdf['turbine_id'].unique()
feature_cols = pdf.columns.drop(['turbine_id', 'timestamp'])
print(f"Total turbines: {len(turbine_ids)}")
print(f"Feature columns: {len(feature_cols)}")

# Train individual models sequentially
start_time = time.time()

with mlflow.start_run(run_name="isolation_forest_sequential_models"):
    
    mlflow.log_param("contamination", 0.1)
    mlflow.log_param("approach", "multiple_models_sequential")
    mlflow.log_param("n_turbines", len(turbine_ids))
    
    # Train a model for each turbine
    models = {}
    for turbine_id in turbine_ids:
        # Get data for this turbine
        turbine_data = pdf[pdf['turbine_id'] == turbine_id][feature_cols].fillna(0)
        
        # Train model
        clf = IsolationForest(contamination=0.1, random_state=42)
        clf.fit(turbine_data)
        
        # Store model
        models[turbine_id] = clf
    
    # Log all models together as a dictionary
    signature = infer_signature(
        pd.DataFrame(columns=feature_cols), 
        np.array([1])  # Example prediction
    )
    mlflow.sklearn.log_model(
        models,
        "turbine_models",
        signature=signature,
        registered_model_name="isolation_forest_models"
    )
    
    # Set the 'prod' alias for the newly logged model version
    client = mlflow.tracking.MlflowClient()
    latest_version = client.search_model_versions(f"name = 'isolation_forest_models'")[0]
    client.set_registered_model_alias("isolation_forest_models", "prod", latest_version.version)
    
    training_time = time.time() - start_time
    mlflow.log_metric("training_time", training_time)
    
print(f"Training time: {training_time:.2f} seconds")
print("Models dictionary has been saved to MLflow Model Registry")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prediction Using Trained Models

# COMMAND ----------

# Read inference data and filter for first 2 turbines only 
inference_spark_df = spark.table(f"{catalog}.{db}.turbine_data_train_10000")  # Use same table as training for now
inference_spark_df = inference_spark_df.filter("turbine_id IN ('Turbine_1', 'Turbine_2')")
inference_pdf = inference_spark_df.toPandas()

# Load the models dictionary from MLflow using the prod alias
loaded_models = mlflow.sklearn.load_model("models:/isolation_forest_models@prod")

# Perform predictions for each turbine
prediction_results = []
for turbine_id in turbine_ids:
    # Get model for this turbine
    model = loaded_models[turbine_id]
    
    # Get data for this turbine
    turbine_data = inference_pdf[inference_pdf['turbine_id'] == turbine_id][feature_cols].fillna(0)
    
    # Make predictions
    predictions = model.predict(turbine_data)  # Returns -1 for anomalies, 1 for normal
    scores = model.score_samples(turbine_data)
    
    # Convert predictions from [-1,1] to [1,0] to match DAXS format
    predictions = (predictions == -1).astype(int)
    
    # Store results
    result_df = pd.DataFrame({
        'turbine_id': [turbine_id] * len(predictions),
        'timestamp': inference_pdf[inference_pdf['turbine_id'] == turbine_id]['timestamp'],
        'predict': predictions,
        'scores': scores
    })
    prediction_results.append(result_df)

# Combine all results
all_predictions = pd.concat(prediction_results, ignore_index=True)

# Log predictions with MLflow
with mlflow.start_run(run_name="isolation_forest_predictions"):
    mlflow.log_param("n_turbines", len(turbine_ids))
    mlflow.log_param("n_predictions", len(all_predictions))
    
    # Evaluate and visualize results
    evaluate_results(all_predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Performance Comparison
# MAGIC
# MAGIC The baseline approaches demonstrate key limitations compared to DAXS:
# MAGIC
# MAGIC 1. **Training Time**: Sequential training with for loops is significantly slower than DAXS's parallel processing
# MAGIC
# MAGIC 2. **Scalability**: The baseline approaches don't scale well with increasing numbers of turbines and sensors
# MAGIC
# MAGIC DAXS addresses these limitations through:
# MAGIC - Parallel processing with Pandas UDFs
# MAGIC - Distributed computation

# COMMAND ----------

# MAGIC %md
# MAGIC © 2024 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License.
