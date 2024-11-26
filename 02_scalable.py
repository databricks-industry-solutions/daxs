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

# DBTITLE 1,Create the catalog and schema if they don't exist
catalog = "daxs"
db = "default"

# Make sure that the catalog exists
_ = spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")

# Make sure that the schema exists
_ = spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{db}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Synthetic data creation
# MAGIC
# MAGIC To demonstrate how DAXS efficiently handles large-scale operations, we will generate a synthetic dataset and use DAXS to perform model fitting and prediction. The dataset simulates a fictitious scenario where thousands of wind turbines are monitored in the field. Each turbine is equipped with a hundred sensors, generating readings sampled at a one-minute frequency. Our objective is to train thousands of individual ECOD models and use them for inference. To achieve this, we leverage Pandas UDFs for distributed processing.

# COMMAND ----------

num_turbines = 10000        # number of turbines
num_sensors = 100           # number of sensors in each turbine
samples_per_turbine = 1440  # corresponds to 1 day of data

# COMMAND ----------

# MAGIC %md
# MAGIC We will set the number of Spark shuffle partitions equal to the number of turbines. This ensures that the same number of Spark tasks as turbines is created when performing a `groupby` operation before applying `applyInPandas`. This approach allows us to utilize resources efficiently.

# COMMAND ----------

sqlContext.setConf("spark.sql.shuffle.partitions", num_turbines)

# COMMAND ----------

# MAGIC %md
# MAGIC We run a custom function, `create_turbine_dataset`, defined in the `99_utilities` notebook to create the dataset. Under the hood, the function leverages the Pandas UDF to generate data from a hundred sensors across thousands of turbines. Once generated, the dataset is written to a Delta table: `{catalog}.{db}.turbine_data_{num_turbines}`.

# COMMAND ----------

create_turbine_dataset(catalog, db, num_turbines, num_sensors, samples_per_turbine, start_date='2025-01-01')

# COMMAND ----------

# Read the data from the Delta table to Spark DataFrame
spark_df = spark.read.table(f"{catalog}.{db}.turbine_data_train_{num_turbines}")

# Display the first few rows of the DataFrame
display(spark_df.filter("turbine_id='Turbine_1'"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Train many ECOD models using Pandas UDF
# MAGIC Pandas UDF is a feature in PySpark that combines the distributed processing power of Spark with the data manipulation capabilities of pandas It uses Apache Arrow to efficiently transfer data between JVM and Python processes, allowing for vectorized operations that can significantly improve performance compared to traditional row-at-a-time UDFs. The first step in utilizing Pandas UDF is to define a function.

# COMMAND ----------

# Define the Pandas UDF for training ECOD models for each turbine
def train_ecod_model(turbine_pdf: pd.DataFrame) -> pd.DataFrame:
    from pyod.models.ecod import ECOD  # Import the ECOD model from PyOD
    import pickle  # Import pickle for model serialization

    # Extract the turbine ID
    turbine_id = turbine_pdf['turbine_id'].iloc[0]

    # Identify feature columns by excluding non-feature columns
    feature_columns = turbine_pdf.columns.drop(['turbine_id', 'timestamp'])

    # Prepare training data by selecting feature columns and filling missing values with 0
    X_train = turbine_pdf[feature_columns].fillna(0)

    # Count the number of records used for training
    n_used = turbine_pdf.shape[0]

    # Initialize and train the ECOD model
    clf = ECOD(n_jobs=1)
    clf.fit(X_train)

    # Serialize the trained model using base64 encoding
    model_encoder = urlsafe_b64encode(pickle.dumps(clf)).decode("utf-8")

    # Create a return DataFrame with turbine ID, number of records used, and encoded model
    returnDF = pd.DataFrame(
        [[turbine_id, n_used, model_encoder]],
        columns=["turbine_id", "n_used", "encode_model"]
    )

    # Return the result as a DataFrame
    return returnDF

# COMMAND ----------

# MAGIC %md
# MAGIC Next, we define the output schema for the Pandas UDF.

# COMMAND ----------

schema = StructType([
    StructField('turbine_id', StringType(), True),
    StructField('n_used', IntegerType(), True),
    StructField('encode_model', StringType(), True)
])

# COMMAND ----------

# MAGIC %md
# MAGIC Finally, we execute the `applyInPandas` method using the previously defined function and schema. The output of this operation is then written to a Delta table. Note that we are adding the current timestamp to the dataframe. This is to distinguish the latest version of the models with their previous ones.

# COMMAND ----------

from pyspark.sql.functions import current_timestamp

# Group the data by turbine_id and train models using applyInPandas
model_df = spark_df.groupBy('turbine_id').applyInPandas(train_ecod_model, schema=schema)

# Write the output of applyInPandas to a delta table
(
  model_df
  .withColumn("created_at", current_timestamp())
  .write.mode("overwrite")
  .saveAsTable(f"{catalog}.{db}.models")
)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's take a peek at the output table. The `n_used` column shows the number of samples used to train each model, while the `encode_model` column contains the binary representation of each model, which can be easily unpickled and used for inference (as will be demonstrated shortly).

# COMMAND ----------

display(spark.sql(f"""
SELECT turbine_id, n_used, LEFT(encode_model, 20) AS encode_model, created_at FROM {catalog}.{db}.models LIMIT 10
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Results
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
