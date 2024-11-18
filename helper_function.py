import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Load Dataset
def load_data(filepath):
    """
    Load dataset from a CSV file.
    Args:
        filepath (str): Path to the dataset file.
    Returns:
        DataFrame: Loaded dataset.
    """
    return pd.read_csv(filepath)

# Split Data
def split_data(data, target_column, test_size=0.2, random_state=42):
    """
    Split data into train and test sets.
    Args:
        data (DataFrame): Dataset.
        target_column (str): Name of the target column.
        test_size (float): Proportion of test data.
        random_state (int): Random seed.
    Returns:
        tuple: X_train, X_test, y_train, y_test.
    """
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_and_evaluate_model_df(algorithm, X_train, y_train, X_test, y_test, **kwargs):
    """
    Train and evaluate a regression model and return metrics as a DataFrame.
    Args:
        algorithm (str): Name of the algorithm ('linear', 'ridge', 'lasso', 'random_forest').
        X_train (array-like): Training features.
        y_train (array-like): Training target.
        X_test (array-like): Test features.
        y_test (array-like): Test target.
        **kwargs: Additional parameters for the specific algorithm.
    Returns:
        DataFrame: DataFrame containing evaluation metrics and algorithm name.
    """
    # Handle algorithm-specific parameters
    if algorithm == 'linear':
        # LinearRegression does not support 'alpha'
        model = LinearRegression()
    elif algorithm == 'ridge':
        model = Ridge(**{k: v for k, v in kwargs.items() if k in ['alpha', 'random_state']})
    elif algorithm == 'lasso':
        model = Lasso(**{k: v for k, v in kwargs.items() if k in ['alpha', 'random_state']})
    elif algorithm == 'random_forest':
        model = RandomForestRegressor(**{k: v for k, v in kwargs.items() if k in ['n_estimators', 'random_state']})
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    metrics = {
        "Model": algorithm,
        "Mean Absolute Error (MAE)": mean_absolute_error(y_test, y_pred),
        "Mean Squared Error (MSE)": mean_squared_error(y_test, y_pred),
        "Root Mean Squared Error (RMSE)": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R-squared (R²)": r2_score(y_test, y_pred)
    }
    
    # Convert metrics to DataFrame
    metrics_df = pd.DataFrame([metrics])
    return model, metrics_df

# Feature Importance
def feature_importance(model, feature_names):
    """
    Get feature importances from the trained model.
    Args:
        model: Trained Random Forest model.
        feature_names (list): Names of the features.
    Returns:
        DataFrame: Feature importance values.
    """
    importance = model.feature_importances_
    return pd.DataFrame({"Feature": feature_names, "Importance": importance}).sort_values(by="Importance", ascending=False)
