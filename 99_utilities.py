# Databricks notebook source
# DBTITLE 1,Import libraries and define utility functions
# Core Data Science Libraries
import numpy as np
import pandas as pd
import functools

# Visualization
import matplotlib.pyplot as plt

# Machine Learning
from sklearn.metrics import roc_auc_score
from pyod.models.ecod import ECOD

def generate_turbine_data(
    df: pd.DataFrame, 
    num_sensors,
    samples_per_turbine,
    start_date,
    ) -> pd.DataFrame:
    turbine_id = [df["turbine_id"].iloc[0]] * samples_per_turbine
    sensor_id = [f'sensor_{i}' for i in range(1, num_sensors + 1)]
    timestamps = [pd.to_datetime(start_date) + pd.Timedelta(minutes=i) for i in range(1, samples_per_turbine + 1)]
    res_df = pd.DataFrame({'turbine_id': turbine_id, 'timestamp': timestamps})
    for i in range(1, num_sensors + 1):
        res_df[f"sensor_{i}"] = list(np.random.normal(loc=0, scale=1, size=samples_per_turbine))
    
    return res_df
  

def create_turbine_dataset(catalog, db, num_turbines, num_sensors, samples_per_turbine, start_date='2025-01-01', return_df=False):
    """
    Creates a synthetic dataset with specified number of turbines,
    each having a fixed number of sensors between num_sensors,
    and adds a timestamp column.
    """
    from pyspark.sql.functions import col, lit
    from pyspark.sql.types import StructType, StructField, StringType, FloatType, IntegerType, TimestampType, ArrayType
    
    columns = [
        StructField('turbine_id', StringType(), True),
        StructField('timestamp', TimestampType(), True),
    ]
    for i in range(1, num_sensors + 1):
        columns.append(StructField(f'sensor_{i}', FloatType()))
    
    turbine_ids = [f'Turbine_{i}' for i in range(1, num_turbines + 1)]

    df = spark.createDataFrame([(tid,) for tid in turbine_ids], ['turbine_id'])

    generate_turbine_data_fn = functools.partial(
        generate_turbine_data, 
        num_sensors=num_sensors,
        samples_per_turbine=samples_per_turbine,
        start_date=start_date,
        )

    df = df.groupBy('turbine_id').applyInPandas(generate_turbine_data_fn, schema=StructType(columns))

    if return_df:
        return df
    else:
        df.write.mode('overwrite').saveAsTable(f'{catalog}.{db}.turbine_data_train_{num_turbines}')


# COMMAND ----------

def evaluate_results(results_df):
    """
    Evaluate and visualize anomaly detection results from predict_explain output.
    
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





def predict_explain(clf, X, feature_cols, top_n):
    """
    Generates predictions, scores, and feature-based explanations for a given classifier and dataset.

    Args:
        clf: A trained classifier object with `predict` and `decision_function` methods.
        X: A pandas DataFrame of input features for prediction.
        feature_cols: A list of feature names corresponding to the columns of X.
        top_n: The number of top features contributing to the prediction to include in explanations.

    Returns:
        tuple: 
            - predict (ndarray): Predicted labels for the input data.
            - scores (ndarray): Decision scores from the classifier.
            - explanations (list): A list of dictionaries containing feature names, values, and contributions 
              for the top contributing features per observation.
    """

    # Calculate predictions and scores
    predict = clf.predict(X)
    scores = clf.decision_function(X)
    
    # Get raw scores
    if hasattr(clf, 'O'):
        raw_scores = clf.O[-X.shape[0]:]
    else:
        raw_scores = clf.decision_function(X)

    # Rank features for anomalies
    ranked = np.argsort(-raw_scores, axis=1)
    max_n = min(raw_scores.shape[1], top_n)    
    explanations = []
    num_observations = len(predict)
    for idx in range(num_observations):
        explaners = []
        for i in range(max_n):
            feature_idx = ranked[idx, i]
            feature_name = feature_cols[feature_idx]
            feature_value = X.iloc[idx, feature_idx]
            strength = (raw_scores[idx, feature_idx] / scores[idx]) * 100
            explaner = {
                f'{i+1}_feature': feature_name,
                f'{i+1}_value': str(round(float(feature_value), 3)),
                f'{i+1}_contribution': f"{round(strength)}%",
            }
            explaners.append(explaner)
        explanations.append(explaners)

    return predict, scores, explanations


