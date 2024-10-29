import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd
from pyod.models.ecod import ECOD

def evaluate_results(X_data, y_pred, clf, set_name):
    df_results = X_data.copy()
    df_results['anomaly'] = y_pred
    df_results['anomaly_score'] = clf.decision_function(X_data)

    print(f"\n--- {set_name} Set Results ---")
    print(f"Detected anomalies: {sum(y_pred)} ({sum(y_pred)/len(y_pred)*100:.2f}%)")

    # Display top 10 anomalies
    display(df_results.sort_values('anomaly_score', ascending=False).head(10))

    # Visualize anomaly scores
    plt.figure(figsize=(12, 6))
    plt.hist(df_results['anomaly_score'], bins=50)
    plt.title(f'Distribution of Anomaly Scores - {set_name} Set')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Frequency')
    plt.show()

    return df_results

def synthetic_auc(model, X, n_synthetic=1000):
    # Generate synthetic normal data
    normal_synthetic = np.random.normal(loc=X.mean(axis=0), scale=X.std(axis=0), size=(n_synthetic, X.shape[1]))
    
    # Generate synthetic anomalies
    anomaly_synthetic = np.random.uniform(low=X.min(axis=0), high=X.max(axis=0), size=(n_synthetic, X.shape[1]))
    
    # Combine synthetic data
    X_synthetic = np.vstack([normal_synthetic, anomaly_synthetic])
    y_synthetic = np.hstack([np.zeros(n_synthetic), np.ones(n_synthetic)])
    
    # Get anomaly scores
    scores = -model.decision_function(X_synthetic)
    
    # Calculate AUC
    auc = roc_auc_score(y_synthetic, scores)
    return auc

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

def explainer(clf, df, training=False, explanation_num=3):
    # Use feature columns stored in the classifier if available
    feature_cols = getattr(clf, 'feature_columns_', df.columns.tolist())
    
    # Select only the feature columns from df
    X = df[feature_cols]
    
    # Calculate predictions and scores
    predict = clf.predict(X)
    scores = clf.decision_function(X)
    
    # Get raw scores
    if hasattr(clf, 'O'):
        raw_scores = clf.O[-X.shape[0]:] if not training else clf.O
    else:
        # If clf.O is not available (e.g., in the deployed model), use decision_function
        raw_scores = clf.decision_function(X)
    
    # Create result DataFrame
    result_df = pd.DataFrame(X, columns=feature_cols)
    result_df['predict'] = predict
    result_df['scores'] = scores
    
    # Rank features and calculate explanations
    ranked = np.argsort(-raw_scores, axis=1)
    max_explanation_num = min(raw_scores.shape[1], explanation_num)
    
    for i in range(max_explanation_num):
        ranked_ids = ranked[:, i]
        result_df[f'Explanation_{i+1}_Feature'] = [feature_cols[j] for j in ranked_ids]
        result_df[f'Explanation_{i+1}_Value'] = X.to_numpy()[np.arange(len(result_df)), ranked_ids]
        raw_scores_per_sample = raw_scores[np.arange(len(result_df)), ranked_ids]
        result_df[f'Explanation_{i+1}_Strength'] = raw_scores_per_sample / scores
    
    return result_df.reset_index(drop=True)
