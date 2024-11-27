# Databricks notebook source
# MAGIC %md
# MAGIC # Anomaly Detection per Turbine using ECOD and Pandas UDFs

# COMMAND ----------

# MAGIC %md
# MAGIC In this notebook, we will demonstrate how ECOD can be applied to a large dataset by utilizing Pandas UDFs with Spark.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Cluster setup
# MAGIC We recommend using a cluster with [Databricks Runtime 15.4 LTS for ML](https://docs.databricks.com/en/release-notes/runtime/15.4lts-ml.html) or above. The cluster can be either a single-node or multi-node CPU cluster. This notebook will leverage [Pandas UDF](https://docs.databricks.com/en/udf/pandas.html) and will utilize all the available resource (i.e., cores). Make sure to set the following Spark configurations before you start your cluster: [`spark.sql.execution.arrow.enabled true`](https://spark.apache.org/docs/3.0.1/sql-pyspark-pandas-with-arrow.html#enabling-for-conversion-tofrom-pandas) and [`spark.sql.adaptive.enabled false`](https://spark.apache.org/docs/latest/sql-performance-tuning.html#adaptive-query-execution). You can do this by specifying [Spark configuration](https://docs.databricks.com/en/compute/configure.html#spark-configuration) in the advanced options on the cluster creation page.

# COMMAND ----------

# DBTITLE 1,Install necessary libraries
# MAGIC %pip install -r requirements.txt --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./99_utilities

# COMMAND ----------

# DBTITLE 1,Import necessary libraries
import numpy as np
import pandas as pd
from pyod.models.ecod import ECOD
from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, StringType, FloatType, IntegerType, TimestampType, ArrayType
from pyspark.sql import functions as F
import mlflow
from base64 import urlsafe_b64encode, urlsafe_b64decode
import pickle

# COMMAND ----------

# DBTITLE 1,Specify the catalog and schema
catalog = "daxs"
db = "default"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Synthetic data creation
# MAGIC
# MAGIC To demonstrate how DAXS efficiently handles large-scale inference, we will generate a synthetic dataset. The dataset simulates a fictitious scenario where thousands of wind turbines are monitored in the field. Each turbine is equipped with a hundred sensors, generating readings sampled at a one-minute frequency. Our objective is to train thousands of individual ECOD models and use them for inference. To achieve this, we leverage Pandas UDFs for distributed processing.

# COMMAND ----------

num_turbines = 10000        # number of turbines
num_sensors = 100           # number of sensors in each turbine
samples_per_turbine = 60    # corresponds to 1 hour of data

# COMMAND ----------

# MAGIC %md
# MAGIC We will set the number of Spark shuffle partitions equal to the number of turbines. This ensures that the same number of Spark tasks as turbines is created when performing a `groupby` operation before applying `applyInPandas`. This approach allows us to utilize resources efficiently.

# COMMAND ----------

sqlContext.setConf("spark.sql.shuffle.partitions", num_turbines)

# COMMAND ----------

# MAGIC %md
# MAGIC We run a custom function, `create_turbine_dataset`, defined in the `99_utilities` notebook to create the dataset. Under the hood, the function leverages the Pandas UDF to generate data from a hundred sensors across thousands of turbines. Once generated, the dataset is written to a Delta table: `{catalog}.{db}.turbine_data_{num_turbines}`.

# COMMAND ----------

inference_df = create_turbine_dataset(catalog, db, num_turbines, num_sensors, samples_per_turbine, start_date='2025-02-01', return_df=True)

# COMMAND ----------

from pyspark.sql.functions import col, rand, when

for i in range(50, 55):
    sensor_col = f"sensor_{i}"
    inference_df = inference_df.withColumn(
        sensor_col,
        when(col("turbine_id") == "Turbine_100", col(sensor_col) + rand() + 2.5).otherwise(col(sensor_col))
    )

display(inference_df.filter("turbine_id='Turbine_100' or turbine_id='Turbine_101'"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Perform inference on many ECOD models using Pandas UDF
# MAGIC
# MAGIC Databricks recommends MLflow as the best practice for tracking models and experiment runs. However, at the scale of this exercise, logging and tracking thousands of runs in a short time can be challenging due to resource [limitations](https://docs.databricks.com/en/resources/limits.html) on the MLflow Tracking Server. To address this, we are using a Delta table to track runs and models instead. In this section, we will load the models stored in the `models` table and use them for inference. Again, we will use Pandas UDF to efficiently distribute the inference across the cluster.

# COMMAND ----------

# Define the function for inference
def predict_with_ecod(turbine_pdf: pd.DataFrame) -> pd.DataFrame:
    import pickle  # Import pickle for deserialization
    from base64 import urlsafe_b64decode  # Import decoding utility
    
    # Extract the turbine ID
    turbine_id = turbine_pdf['turbine_id'].iloc[0]

    # Identify feature columns by excluding non-feature columns
    feature_columns = turbine_pdf.columns.drop(['turbine_id', 'timestamp', 'n_used', 'encode_model', 'created_at'])

    # Deserialize the ECOD model from the encoded string
    encode_model = turbine_pdf['encode_model'].iloc[0]
    model = pickle.loads(urlsafe_b64decode(encode_model.encode("utf-8")))

    # Prepare test data by selecting feature columns and filling missing values with 0
    X_test = turbine_pdf[feature_columns].fillna(0)
    
    # Explode the dataframe to multiple rows
    X_test = X_test.explode(X_test.columns.tolist())

    # Cast all columns as float
    X_test = X_test.astype('float64')
    
    # Perform inference using the trained model
    y_pred = model.predict(X_test)              # Predict anomalies
    scores = model.decision_function(X_test)    # Compute anomaly scores

    # Append the results to the original dataframe
    turbine_pdf['anomaly'] = [y_pred]
    turbine_pdf['anomaly_score'] = [scores]

    # Remove unnecessary columns before returning the result
    drop_columns = ['n_used', 'encode_model', 'created_at'] + [col for col in turbine_pdf.columns if col.startswith('sensor')]
    result_pdf = turbine_pdf.drop(columns = drop_columns)

    #explanation = explainer(X_test, model, test) # correct receall is explainer(model, X_test, top_n=3) or just simply explainer(model, X_test)
    #turbine_pdf['explanation'] = dict() 

    return result_pdf.reset_index(drop=True)

# COMMAND ----------

# Define the result schema
result_schema = StructType(
    [
        StructField("turbine_id", StringType()),
        StructField("timestamp", ArrayType(TimestampType())),
    ] 
    + [StructField('anomaly', ArrayType(IntegerType()), True)]
    + [StructField('anomaly_score', ArrayType(FloatType()), True)]
    #+ [StructField('explanation', ArrayType(FloatType()), True)]
)

# COMMAND ----------

# MAGIC %md
# MAGIC We load the models from the Delta table, filtering the `created_at` column to retrieve only the latest version.

# COMMAND ----------

from pyspark.sql.window import Window
from pyspark.sql.functions import col, row_number

# Read data from the delta table
model_df = spark.read.table(f"{catalog}.{db}.models")

# Define window specification
window_spec = Window.partitionBy('turbine_id').orderBy(col('created_at').desc())

# Add row number to each row within the window
model_df_with_row_num = model_df.withColumn('row_num', row_number().over(window_spec))

# Filter to get the latest row for each turbine_id
latest_model_df = model_df_with_row_num.filter(col('row_num') == 1).drop('row_num')

# COMMAND ----------

# MAGIC %md
# MAGIC For demonstration purpose, we perform inference on the training sample. We use the sensor data collected over the last hour simulating a scenario where the outlier detection will be run on an hourly basis in batches.

# COMMAND ----------

from pyspark.sql.functions import collect_list, current_timestamp

# Group by turbine_id and collect values into lists for each column
inference_df_collected = inference_df.groupBy('turbine_id').agg(
    *[collect_list(col_name).alias(col_name) for col_name in inference_df.columns if col_name != 'turbine_id']
)

# COMMAND ----------

# Join the model DataFrame with the latest sensor DataFrame
joined_df = inference_df_collected.join(latest_model_df, on='turbine_id', how='inner')

# Apply the inference function using applyInPandas
result_df = joined_df.groupBy('turbine_id').applyInPandas(predict_with_ecod, schema=result_schema)

# Write the output of applyInPandas to a delta table
(
  result_df
  .withColumn("scored_at", current_timestamp())
  .write.mode("overwrite")
  .saveAsTable(f"{catalog}.{db}.results")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Post prediction analysis

# COMMAND ----------

results = spark.sql(f"""SELECT * FROM {catalog}.{db}.results""")

# COMMAND ----------

exploded_results = results.withColumn("zip", F.arrays_zip("timestamp", "anomaly", "anomaly_score"))\
    .withColumn("zip", F.explode("zip"))\
    .select(
        "turbine_id", 
        F.col("zip.timestamp").alias("timestamp"), 
        F.col("zip.anomaly").alias("anomaly"), 
        F.col("zip.anomaly_score").alias("anomaly_score"), 
        "scored_at")

display(exploded_results.filter("turbine_id='Turbine_100' or turbine_id='Turbine_101'").orderBy("turbine_id", "timestamp"))

# COMMAND ----------

anomaly_counts = exploded_results.filter("anomaly = 1")\
    .groupBy("turbine_id")\
    .count()\
    .withColumnRenamed("count", "anomaly_count")

display(anomaly_counts.orderBy(F.desc("anomaly_count")).limit(10))

# COMMAND ----------

import pickle  # Import pickle for deserialization
from base64 import urlsafe_b64decode  # Import decoding utility

model_df = spark.sql(f"""
                     SELECT encode_model FROM {catalog}.{db}.models WHERE turbine_id = 'Turbine_100' ORDER BY created_at DESC LIMIT 1
                     """).toPandas()
encode_model = model_df["encode_model"].iloc[0]
model = pickle.loads(urlsafe_b64decode(encode_model.encode("utf-8")))

inference_pdf_filtered = inference_df.filter("turbine_id = 'Turbine_100'").toPandas().drop(["turbine_id", "timestamp"], axis=1)

anomalies = spark.sql(f"""
                     SELECT anomaly FROM {catalog}.{db}.results WHERE turbine_id = 'Turbine_100' ORDER BY timestamp
                     """).toPandas()
anomalies = anomalies["anomaly"].iloc[0]

# COMMAND ----------

# Evaluate results
eval_results = evaluate_results(inference_pdf_filtered, anomalies, model, "Test")

# COMMAND ----------

# Identify the most and least anomalous test samples
scores = model.decision_function(inference_pdf_filtered)
most_anomalous_index = np.argmax(scores)
least_anomalous_index = np.argmin(scores)

# Feature names (assuming X_test is a DataFrame)
feature_names = inference_pdf_filtered.columns.tolist()

# Generate explain_outlier plots for the most anomalous test sample
print("Most Anomalous Test Sample:")
explain_test_outlier(model, inference_pdf_filtered, most_anomalous_index, feature_names=feature_names)

# Generate explain_outlier plots for the least anomalous test sample
print("Least Anomalous Test Sample:")
explain_test_outlier(model, inference_pdf_filtered, least_anomalous_index, feature_names=feature_names)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Results
# MAGIC
# MAGIC In this notebook, we demonstrated how ECOD can be used to fit and make prediction on a large dataset by utilizing Pandas UDFs with Spark.
# MAGIC
# MAGIC To execute this notebook, we used a multi-node interactive cluster consisting of 8 workers, each equipped with 4 cores and 16 GB of memory. The setup corresponds to [m5d.xlarge](https://www.databricks.com/product/pricing/product-pricing/instance-types) instances on AWS (12.42 DBU/h) or [Standard_D4ds_v5](https://www.databricks.com/product/pricing/product-pricing/instance-types) instances on Azure (18 DBU/h). Training individual models for 1,440 time points across 10,000 turbines with 100 sensors took approximately 4 minutes. Performing inference on the 10,000 trained models, each executed 60 times, required about 7.5 minutes. 
# MAGIC
# MAGIC An efficient implementation of ECOD combined with Pandas UDF allows these [embarrasigly parallelizable](https://en.wikipedia.org/wiki/Embarrassingly_parallel) operations to scale proportionally with the size of the cluster: i.e., number of cores. 

# COMMAND ----------

# MAGIC %md
# MAGIC © 2024 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries and dataset are subject to the licenses set forth below.
# MAGIC
# MAGIC | library / datas                        | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | pyod | A Comprehensive and Scalable Python Library for Outlier Detection (Anomaly Detection) | BSD License | https://pypi.org/project/pyod/
