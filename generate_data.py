# Databricks notebook source
dbutils.widgets.text("catalog", "")
dbutils.widgets.text("db", "")
dbutils.widgets.text("num_turbines", "")
dbutils.widgets.text("num_sensors", "")
dbutils.widgets.text("samples_per_turbine", "")

catalog = dbutils.widgets.get("catalog")
db = dbutils.widgets.get("db")
num_turbines = int(dbutils.widgets.get("num_turbines") or "0")
num_sensors = int(dbutils.widgets.get("num_sensors") or "0")
samples_per_turbine = int(dbutils.widgets.get("samples_per_turbine") or "0")

# COMMAND ----------

import functools
import numpy as np
import pandas as pd

def generate_turbine_data(
        df: pd.DataFrame, 
        num_sensors,
        samples_per_turbine,
        start_date
    ) -> pd.DataFrame:
    turbine_id = [df["turbine_id"].iloc[0]] * samples_per_turbine
    sensor_id = [f'sensor_{i}' for i in range(1, num_sensors + 1)]
    timestamps = [pd.to_datetime(start_date) + pd.Timedelta(minutes=i) for i in range(1, samples_per_turbine + 1)]
    res_df = pd.DataFrame({'turbine_id': turbine_id, 'timestamp': timestamps})
    for i in range(1, num_sensors + 1):
        res_df[f"sensor_{i}"] = list(np.random.normal(loc=0, scale=1, size=samples_per_turbine))
    return res_df
  

def create_turbine_dataset(num_turbines, num_sensors, samples_per_turbine, start_date='2025-01-01'):
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
      columns.append(StructField(f'sensor_{i}', FloatType(), True))
  
  turbine_ids = [f'Turbine_{i}' for i in range(1, num_turbines + 1)]

  df = spark.createDataFrame([(tid,) for tid in turbine_ids], ['turbine_id'])

  generate_turbine_data_fn = functools.partial(
      generate_turbine_data, 
      num_sensors=num_sensors,
      samples_per_turbine=samples_per_turbine,
      start_date=start_date,
      )

  df = df.groupBy('turbine_id').applyInPandas(generate_turbine_data_fn, schema=StructType(columns))

  return df
  

df = create_turbine_dataset(
  num_turbines=num_turbines, 
  num_sensors=num_sensors, 
  samples_per_turbine=samples_per_turbine,
  )

df.write.mode('overwrite').saveAsTable(f'{catalog}.{db}.turbine_data_{num_turbines}')
