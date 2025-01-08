# Databricks notebook source
# MAGIC %md
# MAGIC # Baseline Comparison: Isolation Forest vs ECOD
# MAGIC
# MAGIC This notebook provides a baseline comparison using Isolation Forest to contrast with DAXS's ECOD implementation.
# MAGIC It deliberately avoids the efficient model encoding strategy used by DAXS to demonstrate the performance benefits.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Cluster setup
# MAGIC To run this baseline notebook, we used a single-node CPU cluster with [Databricks Runtime 15.4 LTS for ML](https://docs.databricks.com/en/release-notes/runtime/15.4lts-ml.html) with [r8g.xlarge](https://www.databricks.com/product/pricing/product-pricing/instance-types) (memory optimized) instances on AWS (27 DBU/h) or [Standard_E4d_v4](https://www.databricks.com/product/pricing/product-pricing/instance-types) (memory optimized) instances on Azure (18 DBU/h).

# COMMAND ----------

# MAGIC %pip install -r requirements.txt --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md 
# MAGIC ## 2. Benchmark setup
# MAGIC
# MAGIC DAXS incorporates multiple layers of performance optimization, making it challenging to directly compare it with another popular anomaly detection model, such as Isolation Forest. These layers include: 1) Single-node versus multi-node execution (e.g., for-loops versus Pandas UDF), 2) Replacing MLflow with Delta tables to reduce long logging times for multiple models and mitigate the risk of exceeding MLflow API rate limits, and 3) ECOD's assumption of independent variables. In this notebook, we evaluate Isolation Forest on a single-node CPU cluster, without parallelization, and log all models to MLflow. To streamline the process, we limit the number of models trained and inferred to 100.

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
experiment_name = f"/Users/{current_user_name}/elevator_anomaly_detection_baseline"
mlflow.set_experiment(experiment_name)
experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

# COMMAND ----------

# Define catalog and schema
catalog = "daxs"
db = "default"

# Read training data and filter for first 100 turbines only
turbine_set = [f"Turbine_{i}" for i in range(1, 101)]
spark_df = spark.table(f"{catalog}.{db}.turbine_data_train_10000")
spark_df = spark_df.filter(spark_df.turbine_id.isin(turbine_set))
print(f"Total records: {spark_df.count()}")
print("Using turbines: Turbine_1 to Turbine_100 for faster testing")

# COMMAND ----------

# Make sure that the schema exists
_ = spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.benchmark")

# COMMAND ----------

# MAGIC
# MAGIC %md
# MAGIC ## 3. Parallel Training of Individual Models
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
        with mlflow.start_run(run_name=f"{turbine_id}", nested=True, experiment_id=experiment_id) as run:
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
            model_name = f"{catalog}.benchmark.isolation_forest_models_{turbine_id}"
            mlflow.sklearn.log_model(
                clf,
                "model",
                signature=signature,
                registered_model_name=model_name
            )
            
            # Set the 'prod' alias for the newly logged model version
            client = mlflow.tracking.MlflowClient()
            latest_version = client.search_model_versions(f"name = '{model_name}'")[0]
            client.set_registered_model_alias(model_name, "prod", latest_version.version)
    
    training_time = time.time() - start_time
    mlflow.log_metric("training_time", training_time)
    
print(f"Training time: {training_time:.2f} seconds")
print("Models dictionary has been saved to MLflow Model Registry")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Prediction Using Trained Models

# COMMAND ----------

# Read inference data and filter for first 2 turbines only 
inference_spark_df = spark.table(f"{catalog}.{db}.turbine_data_train_10000")  # Use same table as training for now
inference_spark_df = inference_spark_df.filter(inference_spark_df.turbine_id.isin(turbine_set))
inference_pdf = inference_spark_df.toPandas()

# Perform predictions for each turbine
prediction_results = []

for turbine_id in turbine_ids:

    # Load the model dictionary from MLflow using the prod alias
    index = turbine_id.split('_')[-1]
    model = mlflow.sklearn.load_model(f"models:/{catalog}.benchmark.isolation_forest_models_turbine_{index}@prod")
    
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
# MAGIC ## 5. Performance Comparison
# MAGIC
# MAGIC Using this specific configuration, training 100 models took 6 minutes, while inference required 2 minutes. Scaling this approach to 10,000 turbines would translate to 600 minutes for training and 200 minutes for inference. In contrast, DAXS can handle the same scale of exercise in about 4 minutes for both training and inference combined. The key limitations of the baseline approach are:
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
