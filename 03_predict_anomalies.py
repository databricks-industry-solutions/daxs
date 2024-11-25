# Databricks notebook source
# MAGIC %md
# MAGIC # Elevator Anomaly Detection: Prediction and Explanation

# COMMAND ----------

# Install required libraries
%pip install mlflow pyod

# COMMAND ----------

import mlflow
import pandas as pd
import numpy as np
from pyod.models.ecod import ECOD
import matplotlib.pyplot as plt

# COMMAND ----------

# MAGIC %run ./99_utilities

# COMMAND ----------

# Load the latest production model
model_name = "ECOD_Anomaly_Detection"
loaded_model = mlflow.sklearn.load_model(f"models:/{model_name}/Production")

# COMMAND ----------

# Load test data (replace this with your actual test data loading logic)
X_test = spark.table("10x_ad.default.elevator_predictive_maintenance_dataset").toPandas()

# COMMAND ----------

# Make predictions
y_pred = loaded_model.predict(X_test)
anomaly_scores = loaded_model.decision_function(X_test)

# COMMAND ----------

# Get explanations
explanations = explainer(loaded_model, X_test, training=False, explanation_num=3)

# COMMAND ----------

# Combine predictions, scores, and explanations
results = pd.DataFrame({
    'prediction': y_pred,
    'anomaly_score': anomaly_scores
})
results = pd.concat([results, explanations], axis=1)

# COMMAND ----------

# Display results
display(results)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Analyze Anomalies

# COMMAND ----------

# Filter anomalies
anomalies = results[results['prediction'] == 1].sort_values('anomaly_score', ascending=False)

# Display top 10 anomalies with explanations
display(anomalies.head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Visualize Anomaly Scores

# COMMAND ----------

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.hist(results['anomaly_score'], bins=50)
plt.title('Distribution of Anomaly Scores')
plt.xlabel('Anomaly Score')
plt.ylabel('Frequency')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC This notebook demonstrates how to load the deployed model, make predictions on new data, and provide explanations for the anomalies detected. You can further customize this notebook to include additional analysis or visualizations based on your specific requirements.
