<img src=https://raw.githubusercontent.com/databricks-industry-solutions/.github/main/profile/solacc_logo.png width="600px">

# DAXS: Detection of Anomalies, eXplainable and Scalable

[![DBR](https://img.shields.io/badge/DBR-ML%20Runtime-red)](https://docs.databricks.com/runtime/mlruntime.html)
[![GitHub](https://img.shields.io/badge/GitHub-View%20on%20GitHub-blue)](https://github.com/databricks-industry-solutions/daxs)

DAXS is a cutting-edge solution designed for advanced anomaly detection in manufacturing environments, providing explainable, scalable, and efficient predictive maintenance capabilities through batch processing.

## Overview

In today's rapidly evolving industrial landscape, detecting anomalies in manufacturing processes is crucial for maintaining efficiency and reducing downtime. DAXS leverages the ECOD (Empirical Cumulative Distribution Functions for Outlier Detection) algorithm to address three critical challenges:

1. **Explainability**: Provides transparent insights into which specific sensors or features contribute to detected anomalies, enabling quick root cause analysis.
2. **Scalability**: Handles datasets with over a billion records and trains thousands of models efficiently through one unified endpoint.
3. **Cost-Effectiveness**: Minimizes computational costs through efficient resource utilization and optimized processing techniques.

## Key Components

- **Introduction (`00_introduction.py`)**: Overview of DAXS architecture and core concepts.

- **Explainable Detection (`01_explainable.py`)**: Implementation of ECOD algorithm for transparent anomaly detection on the Elevator Predictive Maintenance Dataset.

- **Scalable Processing (`02_many_models_ad.py`)**: Framework for handling large-scale datasets and multiple model training.

- **Batch Inference (`03_predict.py`)**: Production-ready inference pipeline for efficient batch anomaly detection.

- **Utility Functions (`00_utilities.py`)**: Helper functions for model evaluation, metrics calculation, and anomaly explanation generation.

## Getting Started

To use DAXS, you'll need:
- A Databricks Runtime ML (DBR-ML) cluster
- Python installed along with the following libraries:
- pyod
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- mlflow

You can install these dependencies using pip:

```
pip install pyod scikit-learn pandas numpy matplotlib seaborn mlflow
```

## Usage

1. Start by running the `01_explainable.py` script to train and evaluate the ECOD model on your dataset.
2. Use the `03_predict_anomalies.py` script to make predictions on new data using the trained model.
3. Extend the functionality in `02_many_models_ad.py` to implement scalable anomaly detection for your specific use case.

## Contributing

DAXS is an open-source project, and we welcome contributions from data scientists, machine learning engineers, and software developers. By contributing to DAXS, you can gain valuable experience working with cutting-edge anomaly detection models and collaborate with experts in the field.

## Future Development

- Complete the implementation of scalable anomaly detection in `02_many_models_ad.py`.
- Enhance the `03_predict_anomalies.py` script to incorporate scalability features for real-time, large-scale anomaly detection.
- Develop additional visualization tools for better interpretation of anomalies.
- Implement more advanced explainability techniques to provide deeper insights into detected anomalies.

DAXS has the potential to revolutionize predictive maintenance in the manufacturing industry by providing an accessible, scalable, and explainable solution for anomaly detection. Join us in developing this powerful tool to help businesses improve their operations and reduce unplanned downtime.

