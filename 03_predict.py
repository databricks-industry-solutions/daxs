# Databricks notebook source
# MAGIC %md
# MAGIC This solution accelerator notebook is available at [Databricks Industry Solutions](https://github.com/databricks-industry-solutions/daxs).

# COMMAND ----------

# MAGIC %md
# MAGIC # Use Anomaly Detection Models for 10,000 Turbines using ECOD and Pandas UDFs

# COMMAND ----------

# MAGIC %md
# MAGIC In this notebook, we will demonstrate how trained ECOD models can be used to perform inference on a large dataset by utilizing Pandas UDFs with Spark.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Cluster setup
# MAGIC We recommend using a multi-node CPU cluster with [Databricks Runtime 15.4 LTS for ML](https://docs.databricks.com/en/release-notes/runtime/15.4lts-ml.html) or above. This notebook will leverage [Pandas UDF](https://docs.databricks.com/en/udf/pandas.html) and will utilize all the available resource (i.e., cores). Make sure to set the following Spark configurations before you start your cluster: [`spark.sql.execution.arrow.enabled true`](https://spark.apache.org/docs/3.0.1/sql-pyspark-pandas-with-arrow.html#enabling-for-conversion-tofrom-pandas) and [`spark.sql.adaptive.enabled false`](https://spark.apache.org/docs/latest/sql-performance-tuning.html#adaptive-query-execution). You can do this by specifying [Spark configuration](https://docs.databricks.com/en/compute/configure.html#spark-configuration) in the advanced options on the cluster creation page.
# MAGIC
# MAGIC *Before starting, ensure this notebook is the only one attached to the cluster, and detach any previously connected notebooks. This helps prevent the cluster from running out of memory during large-scale inference.*

# COMMAND ----------

# DBTITLE 1,Install necessary libraries
# MAGIC %pip install -r requirements.txt --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Run a utility notebook
# MAGIC %run ./99_utilities

# COMMAND ----------

# DBTITLE 1,Import necessary libraries
import numpy as np
import pandas as pd
from pyod.models.ecod import ECOD
from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, StringType, FloatType, IntegerType, TimestampType, MapType, ArrayType
from pyspark.sql import functions as F
import mlflow
from base64 import urlsafe_b64encode, urlsafe_b64decode
import pickle


# Get the current user name and store it in a variable
current_user_name = spark.sql("SELECT current_user()").collect()[0][0]

# Set the experiment name
mlflow.set_experiment(f"/Users/{current_user_name}/elevator_anomaly_detection")

# COMMAND ----------

# MAGIC %md
# MAGIC Make sure you've run the previous notebook and the catalog and the schema exist.

# COMMAND ----------

# DBTITLE 1,Specify the catalog and schema
catalog = "daxs"
db = "default"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Synthetic data creation
# MAGIC
# MAGIC To demonstrate how ECOD can efficiently handle large-scale inference, we will create a synthetic dataset simulating a hypothetical scenario. In this scenario, thousands of wind turbines are monitored in the field, each equipped with a hundred sensors generating data at one-minute intervals. We've collected an hour's worth of sensor readings and are now prepared to apply batch inference to identify any signs of turbine failures. As in the training notebook, we will leverage Pandas UDFs for distributed processing.

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
# MAGIC We run a custom function, `create_turbine_dataset`, defined in the `99_utilities` notebook to create the dataset. The argument `return_df=True` enables you to load the generated dataset into a Spark DataFrame instead of saving it directly to a Delta table.

# COMMAND ----------

synthetic_data = create_turbine_dataset(catalog, db, num_turbines, num_sensors, samples_per_turbine, start_date='2025-02-01', return_df=True)

# COMMAND ----------

# MAGIC %md
# MAGIC Since this is a simulation, we will introduce anomalies into some of the sensor readings for one turbine. Later, we will evaluate whether ECOD can effectively detect these anomalies.

# COMMAND ----------

from pyspark.sql.functions import col, rand, when, current_timestamp

# Iterate over sensor columns from "sensor_50" to "sensor_54"
for i in range(50, 55):
    sensor_col = f"sensor_{i}"  # Get the sensor column names
    synthetic_data = synthetic_data.withColumn(
        sensor_col,
        # If turbine_id is "Turbine_100", add randomness and offset (5) to the sensor value
        when(col("turbine_id") == "Turbine_100", col(sensor_col) + rand() + 5)
        .otherwise(col(sensor_col))  # Otherwise, keep the original value
    )

# Write the output of applyInPandas to a delta table
(
    synthetic_data
    .withColumn("created_at", current_timestamp())
    .write.mode("overwrite")
    .saveAsTable(f"{catalog}.{db}.turbine_data_inference_{num_turbines}")
)


# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Perform inference using many ECOD models using Pandas UDF
# MAGIC
# MAGIC In this section, we will load the models stored in the `models` table and use them for inference. Again, we will use [Pandas UDF](https://docs.databricks.com/en/udf/pandas.html) to efficiently distribute the inference across the cluster.

# COMMAND ----------

# Define the pandas udf function for inference
def predict_with_ecod(turbine_pdf: pd.DataFrame, explanation_num=3) -> pd.DataFrame:
    import numpy as np
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

    # Perform inference using the trained model. predict_explain method is defined in 99_utilities
    y_pred, scores, explanations = predict_explain(model, X_test, feature_columns, explanation_num)
    
    # Append the results to the original dataframe
    turbine_pdf['anomaly'] = [y_pred]
    turbine_pdf['anomaly_score'] = [scores]
    turbine_pdf['explanations'] = [explanations]
    
    # Remove unnecessary columns before returning the result
    drop_columns = ['n_used', 'encode_model', 'created_at'] + [col for col in turbine_pdf.columns if col.startswith('sensor')]
    result_pdf = turbine_pdf.drop(columns = drop_columns)
    
    return result_pdf.reset_index(drop=True)

# COMMAND ----------

# MAGIC %md
# MAGIC We define the output schema for the Pandas UDF.

# COMMAND ----------

# Number of explanations to generate
explanation_num = 5

# Define the result schema
result_schema = StructType(
    [
        StructField("turbine_id", StringType()),
        StructField("timestamp", ArrayType(TimestampType())),
    ] 
    + [StructField('anomaly', ArrayType(IntegerType()))]
    + [StructField('anomaly_score', ArrayType(FloatType()))]
    + [StructField('explanations', ArrayType(ArrayType(MapType(StringType(), StringType()))))]
)

# COMMAND ----------

# MAGIC %md
# MAGIC We load the models from the Delta table, filtering the `created_at` column to retrieve the latest version.

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
# MAGIC We will perform inference on the synthetic dataset generate above.

# COMMAND ----------

from pyspark.sql.functions import collect_list, current_timestamp

inference_df = spark.read.table(f"{catalog}.{db}.turbine_data_inference_{num_turbines}").drop('created_at')

# Group by turbine_id and collect values into lists for each column
inference_df_collected = inference_df.groupBy('turbine_id').agg(
    *[collect_list(col_name).alias(col_name) for col_name in inference_df.columns if col_name != 'turbine_id']
)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's run the Pandas UDF. The result will be written into a Delta table called `results`.

# COMMAND ----------

import functools

# Join the model DataFrame with the latest sensor DataFrame
joined_df = inference_df_collected.join(latest_model_df, on='turbine_id', how='inner')

# Passing explanation_num to pandas udf using functools
predict_with_ecod_fn = functools.partial(
        predict_with_ecod, 
        explanation_num=explanation_num,
        )

with mlflow.start_run(run_name="ECOD_models_batch_inference") as run:

  # Apply the inference function using applyInPandas
  result_df = joined_df.groupBy('turbine_id').applyInPandas(predict_with_ecod_fn, schema=result_schema)

  # Write the output of applyInPandas to a delta table
  (
    result_df
    .withColumn("scored_at", current_timestamp())
    .write.mode("overwrite")
    .saveAsTable(f"{catalog}.{db}.results")
  )

  mlflow.log_param("source", f"{catalog}.{db}.models")
  mlflow.log_param("target", f"{catalog}.{db}.results")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Post inference analysis
# MAGIC
# MAGIC Let's conduct a post-inference analysis to determine if ECOD successfully detected the anomalies we introduced into our dataset. To do this, we will first query the `results` table.

# COMMAND ----------

results = spark.sql(f"""SELECT * FROM {catalog}.{db}.results""")

# COMMAND ----------

exploded_results = results.withColumn("zip", F.arrays_zip("timestamp", "anomaly", "anomaly_score", "explanations"))\
    .withColumn("zip", F.explode("zip"))\
    .select(
        "turbine_id", 
        F.col("zip.timestamp").alias("timestamp"), 
        F.col("zip.anomaly").alias("anomaly"), 
        F.col("zip.anomaly_score").alias("anomaly_score"), 
        F.col("zip.explanations").alias("explanations"), 
        "scored_at")

# COMMAND ----------

# MAGIC %md
# MAGIC The distribution of anomaly data points per turbine resembles a half-normal distribution. However, when examining the turbines with the highest number of anomaly data points, one clearly stands out—it's our Turbine_100!

# COMMAND ----------

import matplotlib.pyplot as plt

anomaly_counts = exploded_results.filter("anomaly = 1")\
    .groupBy("turbine_id")\
    .count()\
    .withColumnRenamed("count", "anomaly_count")

anomaly_counts_pd = anomaly_counts.orderBy(F.desc("anomaly_count")).toPandas()

plt.hist(anomaly_counts_pd["anomaly_count"], bins=20)
plt.xlabel('Anomaly Count')
plt.ylabel('Number of Turbines')
plt.title('Histogram of Anomaly Counts')
plt.show()

display(anomaly_counts.orderBy(F.desc("anomaly_count")).limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC By leveraging the explainability features of DAXS, we can delve deeper to identify which sensor readings contributed to the anomaly assignments. This information is stored in the `explanations` column of the `results` table.

# COMMAND ----------

display(exploded_results.filter("turbine_id='Turbine_100'").orderBy("turbine_id", "timestamp"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Wrap up
# MAGIC
# MAGIC In this notebook, we demonstrated how ECOD can be used to make inference using thousands of models by leveraging Pandas UDFs with Spark.
# MAGIC
# MAGIC To execute this notebook, we used a multi-node interactive cluster consisting of 8 workers, each equipped with 4 cores and 32 GB of memory. The setup corresponds to [r8g.xlarge](https://www.databricks.com/product/pricing/product-pricing/instance-types) (memory optimized) instances on AWS (27 DBU/h) or [Standard_E4d_v4](https://www.databricks.com/product/pricing/product-pricing/instance-types) (memory optimized) instances on Azure (18 DBU/h). Performing inference on the 10,000 trained models, each executed 60 times, required about 2 minutes. 
# MAGIC
# MAGIC An efficient implementation of ECOD combined with Pandas UDF allows these [embarrasigly parallelizable](https://en.wikipedia.org/wiki/Embarrassingly_parallel) operations to scale proportionally with the size of the cluster: i.e., number of cores. 

# COMMAND ----------

# MAGIC %md
# MAGIC © 2024 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries and dataset are subject to the licenses set forth below.
# MAGIC
# MAGIC | library / datas                        | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | pyod | A Comprehensive and Scalable Python Library for Outlier Detection (Anomaly Detection) | BSD License | https://pypi.org/project/pyod/
