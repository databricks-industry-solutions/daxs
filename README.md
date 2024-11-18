<img src=https://raw.githubusercontent.com/databricks-industry-solutions/.github/main/profile/solacc_logo.png width="600px">

# DAXS: Detection of Anomalies, Explainable and Scalable

DAXS is an advanced, open-source solution for anomaly detection in manufacturing environments, designed to accelerate predictive maintenance through explainable and scalable models.

## Overview

DAXS leverages state-of-the-art anomaly detection techniques, particularly the ECOD (Empirical Cumulative Distribution Functions for Outlier Detection) algorithm, to identify potential issues in manufacturing processes before they escalate into serious problems. The project is built on three core principles:

1. **Detection**: Utilizes robust anomaly detection methods to identify unusual patterns in sensor data.
2. **Explainability**: Provides clear, interpretable insights into detected anomalies, enabling informed decision-making.
3. **Scalability**: Designed to handle large-scale datasets and adapt to various manufacturing environments.

## Key Features

- **Explainable Anomaly Detection**: The `01_explainable.py` module demonstrates the use of ECOD for anomaly detection on the Elevator Predictive Maintenance Dataset. It includes functionality for visualizing anomaly scores and explaining the most significant factors contributing to each anomaly.

- **Scalable Implementation**: While not fully implemented yet, the `02_many_models_ad.py` file is intended to showcase how the anomaly detection process can be scaled to handle multiple models or larger datasets efficiently.

- **Inference and Deployment**: The `03_predict_anomalies.py` script provides a framework for loading a trained model and making predictions on new data. This module is designed to be extended with scalability features to handle real-time, large-scale anomaly detection.

- **Utility Functions**: The `00_utilities.py` file contains a collection of helper functions for evaluating results, calculating synthetic AUC scores, and generating explanations for detected anomalies.

## Getting Started

To use DAXS, you'll need to have Python installed along with the following libraries:
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

