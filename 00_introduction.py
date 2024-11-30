# Databricks notebook source
# MAGIC %md
# MAGIC This solution accelerator notebook is available at [Databricks Industry Solutions](https://github.com/databricks-industry-solutions/daxs).

# COMMAND ----------

# MAGIC %md
# MAGIC <img src='https://github.com/databricks-industry-solutions/.github/raw/main/profile/solacc_logo_wide.png' width="1000" ></img>
# MAGIC
# MAGIC # Training 10,000 Anomaly Detection Models for Under $1 on One Billion Records
# MAGIC
# MAGIC ![DAXS Architecture](daxs_architecture.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Introduction
# MAGIC
# MAGIC In today's industrial IoT landscape, organizations need to monitor thousands of assets generating billions of sensor readings. However, the computational costs of processing such massive datasets have traditionally been prohibitive. DAXS shatters this barrier by demonstrating how to:
# MAGIC
# MAGIC - Train 10,000 individual anomaly detection models
# MAGIC - Process over 1 billion sensor readings (10,000 turbines × 100 sensors × 1,440 daily samples)
# MAGIC - Generate detailed explanations for every prediction
# MAGIC - Complete the entire pipeline for under $1
# MAGIC
# MAGIC The architecture diagram below shows how DAXS achieves this through efficient data partitioning, parallel processing with Pandas UDFs, and optimized model serialization. This solution makes industrial-scale anomaly detection accessible to organizations of any size.
# MAGIC
# MAGIC ![DAXS Architecture](DAXS.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Key Features
# MAGIC
# MAGIC - **Explainable Anomaly Detection**: DAXS utilizes the ECOD (Empirical Cumulative Distribution Functions for Outlier Detection) algorithm to detect anomalies in sensor data. Unlike traditional black-box models, ECOD offers transparency by identifying which specific sensors or features contribute to an anomaly. This level of explainability empowers maintenance teams to quickly pinpoint the root cause of issues and take targeted corrective actions.
# MAGIC
# MAGIC - **Scalability**: Built with scalability at its core, DAXS can handle datasets with over a billion records and train thousands of models efficiently. It enables organizations to achieve anomaly detection across hundreds of facilities through one unified endpoint and optimized algorithms while leveraging distributed computing platforms to ensure reliable performance.
# MAGIC
# MAGIC - **Cost-Effectiveness**: DAXS is designed to minimize computational costs through efficient resource utilization and optimized processing techniques. This approach makes it a super cheap solution that is accessible to organizations of all sizes, eliminating the financial barriers often associated with large-scale machine learning initiatives.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Why DAXS?
# MAGIC
# MAGIC DAXS addresses three critical challenges in industrial IoT anomaly detection:
# MAGIC
# MAGIC - **Ultra Cost-Efficiency**: Process billions of records and train thousands of models for less than $1, making enterprise-scale anomaly detection accessible to any organization.
# MAGIC
# MAGIC - **Massive Scalability**: Handle 10,000+ assets and 100+ sensors per asset efficiently through optimized parallel processing and smart data partitioning.
# MAGIC
# MAGIC - **Detailed Explainability**: For every anomaly detected, identify the exact sensors and measurements that contributed, enabling immediate root cause analysis.

# COMMAND ----------

# MAGIC %md
# MAGIC ## How DAXS Works
# MAGIC
# MAGIC DAXS employs the ECOD algorithm for anomaly detection, which is highly efficient and interpretable. Here's a high-level overview of how DAXS operates:
# MAGIC
# MAGIC 1. **Data Ingestion**: Collect sensor data from various manufacturing equipment and processes.
# MAGIC
# MAGIC 2. **Preprocessing**: Clean and preprocess the data to handle missing values, outliers, and normalize the features.
# MAGIC
# MAGIC 3. **Model Training**: Train the ECOD model on the preprocessed data. The model learns the empirical distribution of the data to identify outliers.
# MAGIC
# MAGIC 4. **Anomaly Detection**: Use the trained model to detect anomalies in new data. The model provides anomaly scores for each data point.
# MAGIC
# MAGIC 5. **Explainability**: For each detected anomaly, DAXS identifies the specific features contributing to the anomaly, enabling quick root-cause analysis.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Use Case Example
# MAGIC
# MAGIC Consider a manufacturing facility with numerous sensors monitoring equipment performance. Traditional anomaly detection methods may alert operators to an issue but provide no insight into what caused it. With DAXS, not only is the anomaly detected, but the specific sensor readings contributing to the anomaly are identified. For example, "An anomaly was detected due to unusually high temperature readings from Sensor 5."

# COMMAND ----------

# MAGIC %md
# MAGIC ## Getting Started with DAXS
# MAGIC
# MAGIC DAXS is built to be easily integrated into your existing infrastructure. Here's how you can get started:
# MAGIC
# MAGIC 1. **Install Dependencies**: Ensure you have the necessary libraries installed, including `pyod`, and `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, and `mlflow` if not using Databricks Machine Learning Runtime.
# MAGIC
# MAGIC ```bash
# MAGIC pip install pyod scikit-learn pandas numpy matplotlib seaborn mlflow
# MAGIC ```
# MAGIC
# MAGIC 2. **Data Preparation**: Load your sensor data into a DataFrame and perform any necessary preprocessing.
# MAGIC
# MAGIC 3. **Training the Model**: Use the DAXS training module to train the ECOD model on your data.
# MAGIC
# MAGIC ```python
# MAGIC from pyod.models.ecod import ECOD
# MAGIC
# MAGIC clf = ECOD()
# MAGIC clf.fit(X_train)
# MAGIC ```
# MAGIC
# MAGIC 4. **Anomaly Detection**: Use the trained model to detect anomalies in new data.
# MAGIC
# MAGIC ```python
# MAGIC y_pred = clf.predict(X_test)
# MAGIC ```
# MAGIC
# MAGIC 5. **Explainability**: Utilize DAXS's explainability functions to understand the contributing factors of each anomaly.
# MAGIC
# MAGIC ```python
# MAGIC from utilities import explain_test_outlier
# MAGIC
# MAGIC explain_test_outlier(clf, X_test, index=0, feature_names=X_test.columns)
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion
# MAGIC
# MAGIC DAXS offers a comprehensive solution for anomaly detection in manufacturing, addressing the key challenges of explainability, scalability, and cost. By providing transparent insights into anomalies and scaling efficiently with your data, DAXS empowers organizations to enhance their predictive maintenance strategies and improve operational efficiency.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC - **Explore the DAXS Modules**: Dive into the other notebooks in this project to see DAXS in action on real datasets.
# MAGIC - **Customize for Your Needs**: Modify the code to suit your specific data and requirements.
# MAGIC - **Contribute**: Join the community in improving DAXS by contributing code, reporting issues, or suggesting new features.

# COMMAND ----------

# MAGIC %md
# MAGIC ## References
# MAGIC
# MAGIC - **ECOD Algorithm**: [ECOD: Unsupervised Outlier Detection Using Empirical Cumulative Distribution Functions](https://arxiv.org/abs/2009.09463)
# MAGIC - **PyOD Library**: [Python Outlier Detection (PyOD)](https://pyod.readthedocs.io/en/latest/)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Contact
# MAGIC
# MAGIC For any questions or inquiries, please contact the DAXS development team at [daxs-support@example.com](mailto:daxs-support@example.com).

# COMMAND ----------

# MAGIC %md
# MAGIC © 2024 DAXS Contributors. All rights reserved.
# MAGIC
