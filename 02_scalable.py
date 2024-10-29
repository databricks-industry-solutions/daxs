# Databricks notebook source
# MAGIC %md
# MAGIC # Anomaly Detection per Turbine using ECOD and Pandas UDFs

# COMMAND ----------
# MAGIC %pip install -U -q mlflow pyod

# COMMAND ----------
# Import necessary libraries
import numpy as np
import pandas as pd
from pyod.models.ecod import ECOD
from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, StringType, FloatType, IntegerType, TimestampType
from pyspark.sql import functions as F
import mlflow
from base64 import urlsafe_b64encode, urlsafe_b64decode
import pickle

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Synthetic Dataset Creation with Timestamp Column

# COMMAND ----------
def create_turbine_data(num_turbines=3, min_sensors=3, max_sensors=5, samples_per_turbine=100, start_date='2021-01-01'):
    """
    Creates a synthetic dataset with specified number of turbines,
    each having a random number of sensors between min_sensors and max_sensors,
    and adds a timestamp column.
    """
    data = []
    base_date = pd.to_datetime(start_date)
    for turbine_id in range(1, num_turbines + 1):
        num_sensors = np.random.randint(min_sensors, max_sensors + 1)
        for i in range(samples_per_turbine):
            sensor_readings = np.random.normal(loc=0, scale=1, size=num_sensors)
            # Pad the sensor readings if necessary
            if len(sensor_readings) < max_sensors:
                sensor_readings = np.pad(sensor_readings, (0, max_sensors - len(sensor_readings)),
                                         'constant', constant_values=np.nan)
            timestamp = base_date + pd.Timedelta(minutes=i)
            data.append([f'Turbine_{turbine_id}', timestamp] + sensor_readings.tolist())
    columns = ['turbine_id', 'timestamp'] + [f'sensor_{i+1}' for i in range(max_sensors)]
    df = pd.DataFrame(data, columns=columns)
    return df

# Create the synthetic dataset with timestamps
df = create_turbine_data(num_turbines=3, min_sensors=3, max_sensors=5, samples_per_turbine=100)
print(df.shape)
df.head()

# COMMAND ----------
# Convert the Pandas DataFrame to Spark DataFrame
spark_df = spark.createDataFrame(df)

# Display the first few rows of the DataFrame
display(spark_df)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Training ECOD Models per Turbine using Pandas UDFs

# COMMAND ----------
# Define the return schema for the Pandas UDF
schema = StructType([
    StructField('turbine_id', StringType(), True),
    StructField('n_used', IntegerType(), True),
    StructField('encode_model', StringType(), True)
])

# COMMAND ----------
# Define the Pandas UDF for training ECOD models per turbine
def train_ecod_model(turbine_pdf: pd.DataFrame) -> pd.DataFrame:
    from pyod.models.ecod import ECOD
    import pickle

    turbine_id = turbine_pdf['turbine_id'].iloc[0]
    feature_columns = turbine_pdf.columns.drop(['turbine_id', 'timestamp'])
    X_train = turbine_pdf[feature_columns].fillna(0)

    n_used = turbine_pdf.shape[0]

    # Train ECOD model
    clf = ECOD()
    clf.fit(X_train)

    # Encode the model
    model_encoder = urlsafe_b64encode(pickle.dumps(clf)).decode("utf-8")

    # Create a return DataFrame
    returnDF = pd.DataFrame(
        [[turbine_id, n_used, model_encoder]],
        columns=["turbine_id", "n_used", "encode_model"]
    )

    return returnDF

# COMMAND ----------
# Group the data by turbine_id and train models using applyInPandas
model_df = spark_df.groupBy('turbine_id').applyInPandas(train_ecod_model, schema=schema)
display(model_df)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Inference

# COMMAND ----------
# Define the result schema
result_schema = spark_df.schema \
    .add(StructField('anomaly', IntegerType(), True)) \
    .add(StructField('anomaly_score', FloatType(), True))

# COMMAND ----------
# Define the function for inference
def predict_with_ecod(turbine_pdf: pd.DataFrame) -> pd.DataFrame:
    import pickle
    from base64 import urlsafe_b64decode

    turbine_id = turbine_pdf['turbine_id'].iloc[0]
    feature_columns = turbine_pdf.columns.drop(['turbine_id', 'timestamp', 'n_used', 'encode_model'])

    # Load the model from the encoded string
    payload = turbine_pdf['encode_model'].iloc[0]
    clf = pickle.loads(urlsafe_b64decode(payload.encode("utf-8")))

    X_test = turbine_pdf[feature_columns].fillna(0)
    y_pred = clf.predict(X_test)
    scores = clf.decision_function(X_test)

    turbine_pdf['anomaly'] = y_pred
    turbine_pdf['anomaly_score'] = scores

    # Remove 'n_used' and 'encode_model' before returning
    result_pdf = turbine_pdf.drop(columns=['n_used', 'encode_model'])

    return result_pdf

# COMMAND ----------
# Join the model DataFrame with the original DataFrame
joined_df = spark_df.join(model_df, on='turbine_id', how='inner')

# COMMAND ----------
# Apply the inference function using applyInPandas
result_df = joined_df.groupBy('turbine_id').applyInPandas(predict_with_ecod, schema=result_schema)
display(result_df)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Conclusion

# COMMAND ----------
