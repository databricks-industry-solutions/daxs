# Databricks notebook source
# MAGIC %md
# MAGIC # Anomaly Detection per Turbine using ECOD and Pandas UDFs

# COMMAND ----------

# MAGIC %md
# MAGIC In this notebook, we will demonstrate how trained ECOD models can be used to perform inference on a large dataset by utilizing Pandas UDFs with Spark.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Cluster setup
# MAGIC We recommend using a cluster with [Databricks Runtime 15.4 LTS for ML](https://docs.databricks.com/en/release-notes/runtime/15.4lts-ml.html) or above. The cluster can be either a single-node or multi-node CPU cluster. This notebook will leverage [Pandas UDF](https://docs.databricks.com/en/udf/pandas.html) and will utilize all the available resource (i.e., cores). Make sure to set the following Spark configurations before you start your cluster: [`spark.sql.execution.arrow.enabled true`](https://spark.apache.org/docs/3.0.1/sql-pyspark-pandas-with-arrow.html#enabling-for-conversion-tofrom-pandas) and [`spark.sql.adaptive.enabled false`](https://spark.apache.org/docs/latest/sql-performance-tuning.html#adaptive-query-execution). You can do this by specifying [Spark configuration](https://docs.databricks.com/en/compute/configure.html#spark-configuration) in the advanced options on the cluster creation page.

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
# MAGIC To demonstrate how DAXS can efficiently handle large-scale inference, we will create a synthetic dataset simulating a hypothetical scenario. In this scenario, thousands of wind turbines are monitored in the field, each equipped with a hundred sensors generating data at one-minute intervals. We've collected an hour's worth of sensor readings and are now prepared to apply batch inference to identify any signs of turbine failures. As in the training notebook, we will leverage Pandas UDFs for distributed processing.

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

inference_df = create_turbine_dataset(catalog, db, num_turbines, num_sensors, samples_per_turbine, start_date='2025-02-01', return_df=True)

# COMMAND ----------

# MAGIC %md
# MAGIC Since this is a simulation, we will introduce anomalies into some of the sensor readings for one turbine. Later, we will evaluate whether ECOD can effectively detect these anomalies.

# COMMAND ----------

from pyspark.sql.functions import col, rand, when

# Iterate over sensor columns from "sensor_50" to "sensor_54"
for i in range(50, 55):
    sensor_col = f"sensor_{i}"  # Get the sensor column names
    inference_df = inference_df.withColumn(
        sensor_col,
        # If turbine_id is "Turbine_100", add randomness and offset (2.5) to the sensor value
        when(col("turbine_id") == "Turbine_100", col(sensor_col) + rand() + 2.5)
        .otherwise(col(sensor_col))  # Otherwise, keep the original value
    )

display(inference_df.filter("turbine_id='Turbine_100'"))


# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Perform inference using many ECOD models using Pandas UDF
# MAGIC
# MAGIC In this section, we will load the models stored in the `models` table and use them for inference. Again, we will use Pandas UDF to efficiently distribute the inference across the cluster.

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
    
    # Perform inference using the trained model
    y_pred = model.predict(X_test)              # Predict anomalies
    scores = model.decision_function(X_test)    # Compute anomaly scores

    # Append the results to the original dataframe
    turbine_pdf['anomaly'] = [y_pred]
    turbine_pdf['anomaly_score'] = [scores]
    
    # Get the anomaly scores of each feature, rank them and calculate explanations
    raw_scores = model.O[-X_test.shape[0]:]
    ranked = np.argsort(-raw_scores, axis=1)
    
    # Limit the number of explanations to the number of features
    max_n = min(raw_scores.shape[1], explanation_num)

    # Add explanations
    explanations = [] 
    num_observations = len(y_pred)
    for idx in range(num_observations):
        explaners = [] 
        for i in range(max_n):
            feature_idx = ranked[idx, i]
            feature_name = feature_columns[feature_idx]
            feature_value = X_test.iloc[idx, feature_idx]
            strength = (raw_scores[idx, feature_idx] / scores[idx]) * 100
            explaner = {
                f'{i+1}_feature': feature_name,
                f'{i+1}_value': str(round(float(feature_value), 3)),
                f'{i+1}_contribution': f"{round(strength)}%"
            }
            explaners.append(explaner)
        explanations.append(explaners)    
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
# MAGIC We will perform inference on the synthetic dataset generate above.

# COMMAND ----------

from pyspark.sql.functions import collect_list, current_timestamp

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

# Apply the inference function using applyInPandas
result_df = joined_df.groupBy('turbine_id').applyInPandas(predict_with_ecod_fn, schema=result_schema)

# Write the output of applyInPandas to a delta table
(
  result_df
  .withColumn("scored_at", current_timestamp())
  .write.mode("overwrite")
  .saveAsTable(f"{catalog}.{db}.results")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Post inference analysis
# MAGIC
# MAGIC Let's conduct a post-inference analysis to determine if DAXS successfully detected the anomalies we introduced into our dataset. To do this, we will first query the `results` table.

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
plt.xlabel('Turbine ID')
plt.ylabel('Anomaly Count')
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
# MAGIC To execute this notebook, we used a multi-node interactive cluster consisting of 8 workers, each equipped with 4 cores and 16 GB of memory. The setup corresponds to [m5d.xlarge](https://www.databricks.com/product/pricing/product-pricing/instance-types) instances on AWS (12.42 DBU/h) or [Standard_D4ds_v5](https://www.databricks.com/product/pricing/product-pricing/instance-types) instances on Azure (18 DBU/h). Performing inference on the 10,000 trained models, each executed 60 times, required about 8 minutes. 
# MAGIC
# MAGIC An efficient implementation of ECOD combined with Pandas UDF allows these [embarrasigly parallelizable](https://en.wikipedia.org/wiki/Embarrassingly_parallel) operations to scale proportionally with the size of the cluster: i.e., number of cores. 

# COMMAND ----------

# MAGIC %md
# MAGIC © 2024 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries and dataset are subject to the licenses set forth below.
# MAGIC
# MAGIC | library / datas                        | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | pyod | A Comprehensive and Scalable Python Library for Outlier Detection (Anomaly Detection) | BSD License | https://pypi.org/project/pyod/
