import polars as pl
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

# --- Configuration ---
MODEL_FILENAME = 'performance_pipeline.joblib'
CSV_FILENAME = 'StudentPerformanceFactors.csv' # Assumes this file is present

# Define the columns used in the model
NUMERICAL_FEATURES = [
    "Hours_Studied", 
    "Attendance", 
    "Sleep_Hours", 
    "Previous_Scores",
    "Tutoring_Sessions",
    "Physical_Activity",
]
CATEGORICAL_FEATURES = [
    "Parental_Involvement", 
    "Motivation_Level", 
    "Internet_Access",
    "Family_Income",
    "Teacher_Quality",
    "School_Type",
    "Gender",
    "Distance_from_Home",
    "Access_to_Resources",
    "Extracurricular_Activities",
    "Peer_Influence",
]
TARGET_COLUMN = "Exam_Score"

def create_model_pipeline(random_state=42):
    """
    Creates a scikit-learn pipeline for preprocessing and model training.
    """
    # 1. Define Preprocessing Steps
    # Standard Scaler for numerical data
    numerical_transformer = StandardScaler()
    
    # One-Hot Encoder for categorical data
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    # Combine transformers using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, NUMERICAL_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ],
        remainder='drop' # Drop all other columns not listed above
    )
    
    # 2. Define the Model Pipeline
    # The pipeline first runs the preprocessor, then trains the Random Forest Regressor
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=random_state))
    ])
    
    return pipeline

def load_data_and_train(pipeline):
    """
    Loads data using Polars, prepares it, trains the model, and evaluates performance.
    """
    try:
        # Load data using Polars
        df = pl.read_csv(CSV_FILENAME)
        
        # Convert Polars DataFrame to Pandas for scikit-learn compatibility
        df_pandas = df.select(
            NUMERICAL_FEATURES + CATEGORICAL_FEATURES + [TARGET_COLUMN]
        ).to_pandas()
        
        # Split data into features (X) and target (y)
        X = df_pandas.drop(TARGET_COLUMN, axis=1)
        y = df_pandas[TARGET_COLUMN]
        
        # Split into training and testing sets (for evaluation)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print("Starting model training...")
        pipeline.fit(X_train, y_train)
        print("Training complete.")
        
        # Evaluate model performance
        y_pred = pipeline.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\nModel Performance on Test Set (20% of data):")
        print(f"Mean Absolute Error (MAE): {mae:.2f}")
        print(f"R-squared (R2): {r2:.2f}")

        return pipeline
        
    except FileNotFoundError:
        print(f"Error: CSV file '{CSV_FILENAME}' not found. Cannot train model.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during training: {e}")
        return None

if __name__ == "__main__":
    # 1. Create the pipeline
    model_pipeline = create_model_pipeline()
    
    # 2. Load data and train
    trained_pipeline = load_data_and_train(model_pipeline)
    
    if trained_pipeline:
        # 3. Save the trained pipeline using joblib
        joblib.dump(trained_pipeline, MODEL_FILENAME)
        print(f"\nSUCCESS: Trained ML Pipeline saved as '{MODEL_FILENAME}'.")
        print("This file can now be loaded into your Shiny application.")
