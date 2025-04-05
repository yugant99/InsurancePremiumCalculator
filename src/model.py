import pandas as pd
import numpy as np
import xgboost as xgb
import os
import pickle
import json
from datetime import datetime
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectFromModel
import joblib

# Define model directory
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

def load_data(filepath):
    """
    Load the insurance data for model training
    
    Args:
        filepath: Path to the CSV file
    
    Returns:
        DataFrame with loaded data
    """
    return pd.read_csv(filepath)

def preprocess_data(df, is_training=True):
    """
    Preprocess the data for model training or prediction
    
    Args:
        df: Input DataFrame
        is_training: Whether preprocessing is for training or prediction
    
    Returns:
        Preprocessed DataFrame, feature columns, label encoders
    """
    # Create a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Select relevant features for model
    # These are the features that are most likely to influence premium
    feature_cols = [
        'SEX', 'INSURED_VALUE', 'PROD_YEAR', 'SEATS_NUM', 
        'CARRYING_CAPACITY', 'TYPE_VEHICLE', 'CCM_TON', 
        'MAKE', 'USAGE', 'HAS_CLAIM'
    ]
    
    # For training, ensure the target variable is available
    if is_training:
        feature_cols.append('PREMIUM')
    
    # Select only the columns we need
    if all(col in data.columns for col in feature_cols):
        data = data[feature_cols]
    else:
        missing_cols = [col for col in feature_cols if col not in data.columns]
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Handle missing values
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = data[col].fillna(data[col].mode()[0])
        else:
            data[col] = data[col].fillna(data[col].median())
    
    # Check for null values after filling
    null_counts = data.isnull().sum()
    if null_counts.sum() > 0:
        print(f"Warning: Null values remain after filling: {null_counts[null_counts > 0]}")
    
    # Initialize or load label encoders
    label_encoders = {}
    categorical_cols = ['TYPE_VEHICLE', 'MAKE', 'USAGE']
    
    if is_training:
        # Create new encoders for training
        for col in categorical_cols:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            label_encoders[col] = le
    else:
        # Load existing encoders for prediction
        encoder_path = os.path.join(MODEL_DIR, 'label_encoders.pkl')
        if os.path.exists(encoder_path):
            label_encoders = joblib.load(encoder_path)
            for col in categorical_cols:
                if col in data.columns:
                    # Handle unseen categories
                    unseen_categories = set(data[col].unique()) - set(label_encoders[col].classes_)
                    if unseen_categories:
                        print(f"Warning: Unseen categories in {col}: {unseen_categories}")
                        # Map unseen categories to -1 (or another strategy)
                        for cat in unseen_categories:
                            data.loc[data[col] == cat, col] = label_encoders[col].classes_[0]
                    # Transform with loaded encoder
                    data[col] = label_encoders[col].transform(data[col])
        else:
            raise FileNotFoundError(f"Label encoders not found at {encoder_path}")
    
    return data, feature_cols, label_encoders

def perform_feature_selection(X, y, max_features=10):
    """
    Perform feature selection to identify most important features
    
    Args:
        X: Feature DataFrame
        y: Target variable
        max_features: Maximum number of features to select
    
    Returns:
        Selected feature names, feature importance scores
    """
    print(f"Performing feature selection on {X.shape[1]} features...")
    
    # Use Random Forest for feature importance
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Select top features
    top_features = feature_importance.head(max_features)['feature'].tolist()
    
    print(f"Selected {len(top_features)} top features: {top_features}")
    
    return top_features, feature_importance

def train_model(data_path, model_type='random_forest', feature_selection=True):
    """
    Train and save a predictive model for insurance premiums
    
    Args:
        data_path: Path to the training data
        model_type: Type of model to train ('linear', 'random_forest', or 'gradient_boosting')
        feature_selection: Whether to perform feature selection
    
    Returns:
        Dictionary with trained model, metrics, and feature information
    """
    print(f"Loading data from {data_path}...")
    data = load_data(data_path)
    
    print(f"Preprocessing data with shape {data.shape}...")
    processed_data, feature_cols, label_encoders = preprocess_data(data, is_training=True)
    
    # Split features and target
    X = processed_data.drop('PREMIUM', axis=1)
    y = processed_data['PREMIUM']
    
    print(f"Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Feature selection (optional)
    if feature_selection:
        selected_features, feature_importance = perform_feature_selection(X_train, y_train)
        X_train = X_train[selected_features]
        X_test = X_test[selected_features]
    else:
        selected_features = X.columns.tolist()
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': np.ones(X.shape[1])
        })
    
    # Train model based on specified type
    print(f"Training {model_type} model...")
    match model_type:

        case 'linear':
          model = LinearRegression()

        case 'x_gradient_boosting':
          model = XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        case 'gradient_boosting':
          model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        case 'random_forest':
           model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
    
    # Fit the model
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Evaluation Results:")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE: {mae:.2f}")
    print(f"  R2 Score: {r2:.4f}")
    
    # Create timestamp for model version
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save the model and related information
    model_name = f"{model_type}_{timestamp}"
    model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
    
    print(f"Saving model to {model_path}...")
    joblib.dump(model, model_path)
    
    # Save label encoders
    encoder_path = os.path.join(MODEL_DIR, 'label_encoders.pkl')
    joblib.dump(label_encoders, encoder_path)
    
    # Save feature information (for user interface and prediction)
    feature_info = {
        'selected_features': selected_features,
        'feature_importance': feature_importance.to_dict(orient='records'),
        'categorical_cols': ['TYPE_VEHICLE', 'MAKE', 'USAGE'],
        'numeric_cols': [col for col in selected_features if col not in ['TYPE_VEHICLE', 'MAKE', 'USAGE']],
        'target_col': 'PREMIUM'
    }
    
    feature_path = os.path.join(MODEL_DIR, 'feature_info.json')
    with open(feature_path, 'w') as f:
        json.dump(feature_info, f, indent=2)
    
    # Save model metadata for future reference
    model_metadata = {
        'model_name': model_name,
        'model_type': model_type,
        'timestamp': timestamp,
        'metrics': {
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        },
        'feature_selection': feature_selection,
        'n_features': len(selected_features),
        'model_path': model_path,
        'feature_info_path': feature_path,
        'encoder_path': encoder_path
    }
    
    metadata_path = os.path.join(MODEL_DIR, f"{model_name}_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(model_metadata, f, indent=2)
    
    return {
        'model': model,
        'metrics': {
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        },
        'feature_importance': feature_importance,
        'selected_features': selected_features,
        'model_path': model_path
    }

def compare_models(data_path):
    """
    Train and compare multiple models
    
    Args:
        data_path: Path to the training data
    
    Returns:
        Dictionary with best model and its results
    """
    print("Comparing different model types...")
    
    model_types = ['linear', 'x_gradient_boosting','gradient_boosting','random_forest']
    results = {}
    
    for model_type in model_types:
        print(f"\nTraining {model_type} model...")
        result = train_model(data_path, model_type=model_type)
        results[model_type] = result
    
    # Find best model based on RMSE
    best_model_type = min(results, key=lambda k: results[k]['metrics']['rmse'])
    
    print(f"\nBest model: {best_model_type}")
    print(f"RMSE: {results[best_model_type]['metrics']['rmse']:.2f}")
    print(f"R2 Score: {results[best_model_type]['metrics']['r2']:.4f}")
    
    return {
        'best_model_type': best_model_type,
        'best_model_result': results[best_model_type],
        'all_results': results
    }

def load_model_for_prediction():
    """
    Load the latest trained model and related information for prediction
    
    Returns:
        Dictionary with model, feature info, and encoders
    """
    # Find the latest model file
    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.pkl') and not f.startswith('label_encoders')]
    
    if not model_files:
        raise FileNotFoundError(f"No model files found in {MODEL_DIR}")
    
    # Find the latest model by timestamp in filename
    latest_model = sorted(model_files)[-1]
    model_path = os.path.join(MODEL_DIR, latest_model)
    
    # Load model
    model = joblib.load(model_path)
    
    # Load feature information
    feature_path = os.path.join(MODEL_DIR, 'feature_info.json')
    with open(feature_path, 'r') as f:
        feature_info = json.load(f)
    
    # Load label encoders
    encoder_path = os.path.join(MODEL_DIR, 'label_encoders.pkl')
    label_encoders = joblib.load(encoder_path)
    
    return {
        'model': model,
        'feature_info': feature_info,
        'label_encoders': label_encoders,
        'model_path': model_path
    }

def predict_premium(input_data):
    """
    Predict premium based on input data
    
    Args:
        input_data: Dictionary or DataFrame with input features
    
    Returns:
        Predicted premium
    """
    # Load model and related information
    model_data = load_model_for_prediction()
    model = model_data['model']
    feature_info = model_data['feature_info']
    label_encoders = model_data['label_encoders']
    
    # Convert input to DataFrame if it's a dictionary
    if isinstance(input_data, dict):
        input_df = pd.DataFrame([input_data])
    else:
        input_df = input_data.copy()
    
    # Check if all required features are present
    missing_features = [f for f in feature_info['selected_features'] if f not in input_df.columns]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    # Encode categorical features
    for col in feature_info['categorical_cols']:
        if col in input_df.columns and col in label_encoders:
            # Handle unseen categories
            for val in input_df[col].unique():
                if val not in label_encoders[col].classes_:
                    input_df.loc[input_df[col] == val, col] = label_encoders[col].classes_[0]
            
            # Encode the column
            input_df[col] = label_encoders[col].transform(input_df[col])
    
    # Select only the features used by the model
    X = input_df[feature_info['selected_features']]
    
    # Make prediction
    prediction = model.predict(X)
    
    if len(prediction) == 1:
        return prediction[0]
    else:
        return prediction

if __name__ == "__main__":
    # Example usage
    data_path = '/Users/crystalwang/Downloads/github/InsurancePremiumCalculator/data/insurance_cleaned_colab.csv'
    
    # Train a single model
    # result = train_model(data_path, model_type='random_forest')
    
    # Compare multiple models and find the best one
    comparison = compare_models(data_path)
    
    # Example of making a prediction with dummy data
    sample_input = {
        'SEX': 0,
        'INSURED_VALUE': 300000,
        'PROD_YEAR': 2010,
        'SEATS_NUM': 4,
        'CARRYING_CAPACITY': 5,
        'TYPE_VEHICLE': 'Pick-up',
        'CCM_TON': 2500,
        'MAKE': 'TOYOTA',
        'USAGE': 'Private',
        'HAS_CLAIM': 0
    }
    
    try:
        pred = predict_premium(sample_input)
        print(f"\nSample prediction for {sample_input['MAKE']} {sample_input['TYPE_VEHICLE']}:")
        print(f"Predicted premium: ${pred:.2f}")
    except Exception as e:
        print(f"Prediction failed: {e}")