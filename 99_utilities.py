# Databricks notebook source
import numpy as np
import pandas as pd
import functools
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from pyod.models.ecod import ECOD

# COMMAND ----------

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




def explain_test_outlier(clf, X_test, index, columns=None, cutoffs=None,
                         feature_names=None, file_name=None, file_type=None):
    """
    Plot dimensional outlier graph for a given data point within
    the test dataset.

    Parameters
    ----------
    clf : ECOD object
        The trained ECOD model.

    X_test : pandas DataFrame or numpy array
        The test data.

    index : int
        The index of the data point one wishes to obtain
        a dimensional outlier graph for.

    columns : list
        Specify a list of features/dimensions for plotting. If not
        specified, use all features.

    cutoffs : list of floats in (0., 1), optional (default=[0.95, 0.99])
        The significance cutoff bands of the dimensional outlier graph.

    feature_names : list of strings
        The display names of all columns of the dataset,
        to show on the x-axis of the plot.

    file_name : string
        The name to save the figure.

    file_type : string
        The file type to save the figure.

    Returns
    -------
    None
        Displays a matplotlib plot.
    """

    # Ensure that clf.decision_function(X_test) has been called
    # Get the number of test samples
    n_test_samples = X_test.shape[0]
    
    # Access the O matrix for test data
    O_test = clf.O[-n_test_samples:, :]
    
    # Determine columns to plot
    if columns is None:
        columns = list(range(O_test.shape[1]))
        column_range = range(1, O_test.shape[1] + 1)
    else:
        column_range = range(1, len(columns) + 1)

    # Set default cutoff values if not provided
    cutoffs = [1 - clf.contamination, 0.99] if cutoffs is None else cutoffs

    # Get O values for all test data and the specified index
    O_values = O_test[:, columns]
    O_row = O_values[index, :]

    # Plot outlier scores
    plt.figure(figsize=(10, 6))
    plt.scatter(column_range, O_row, marker='^', c='black', label='Outlier Score')

    for cutoff in cutoffs:
        plt.plot(column_range, np.quantile(O_values, q=cutoff, axis=0), '--',
                 label=f'{cutoff*100}% Cutoff Band')

    plt.xlim([0.95, max(column_range) + 0.05])
    plt.ylim([0, int(O_values.max()) + 1])
    plt.ylabel('Dimensional Outlier Score')
    plt.xlabel('Dimension')

    ticks = list(column_range)
    if feature_names is not None:
        assert len(feature_names) == len(ticks), \
            "Length of feature_names does not match dataset dimensions."
        plt.xticks(ticks, labels=feature_names, rotation=90)
    else:
        plt.xticks(ticks)

    plt.yticks(range(0, int(O_values.max()) + 1))
    label = 'Outlier' if clf.predict(X_test.iloc[[index]])[0] == 1 else 'Inlier'
    plt.title(f'Outlier Score Breakdown for Test Sample #{index + 1} ({label})')
    plt.legend()
    plt.tight_layout()

    # Save the file if specified
    if file_name is not None:
        if file_type is not None:
            plt.savefig(f"{file_name}.{file_type}", dpi=300)
        else:
            plt.savefig(f"{file_name}.png", dpi=300)
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


def explainer(clf, df, top_n=3):
    """
    Generate explanations for anomalies in the dataset.
    Returns a DataFrame with predictions, scores, and explanations in JSON format.
    Only generates explanations for predicted anomalies (predict=1).
    """
    import json
    
    # Use feature columns stored in the classifier if available
    feature_cols = getattr(clf, 'feature_columns_', df.columns.tolist())
    
    # Select only the feature columns from df
    X = df[feature_cols]

    predict, scores, explanations = predict_explain(clf, X, feature_cols, top_n)
    
    explanations = [' '.join(map(str, exp)) for exp in explanations]

    # Create result DataFrame
    result_df = pd.DataFrame({
        'predict': predict,
        'scores': scores,
        'explanations': np.array(explanations),  
    })

    return result_df.reset_index(drop=True)
