import numpy as np
import pandas as pd
import os
import json
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


class AdultDataProcessor:
    """Handles loading and preprocessing of the Adult Income dataset"""
    
    def __init__(self):
        self.df = None
        self.X = None
        self.y = None
        self.metadata = None
        self.target_column = 'income'
        self.preprocessing_pipeline = None
        self.categorical_columns = None
        self.numerical_columns = None
        
    def load_from_uci(self):
        """Load data directly from UCI repository"""
        # Fetch dataset
        adult = fetch_ucirepo(id=2)
        
        # Extract features, targets, and metadata
        self.X = adult.data.features
        self.y = adult.data.targets
        self.metadata = adult.metadata
        
        # Combine features and target into one DataFrame
        self.df = pd.concat([self.X, self.y], axis=1)
        
        # Keep track of the target column name
        self.target_column = self.y.columns[0]
        
        return self.df
    
    def load_from_csv(self, file_path):
        """Load data from a CSV file"""
        self.df = pd.read_csv(file_path)
        return self.df
    
    def create_preprocessing_pipeline(self):
        """Create preprocessing pipeline using sklearn components"""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_from_uci() or load_from_csv() first.")
        
        # Clean target column
        self.df[self.target_column] = self.df[self.target_column].str.strip().str.replace('.', '', regex=False)
        
        # Replace '?' with np.nan for consistent missing value notation
        self.df.replace('?', np.nan, inplace=True)
        
        # Identify categorical and numerical columns
        self.categorical_columns = self.df.select_dtypes(include=['object']).columns.tolist()
        self.numerical_columns = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Remove target from feature columns
        if self.target_column in self.categorical_columns:
            self.categorical_columns.remove(self.target_column)
        if self.target_column in self.numerical_columns:
            self.numerical_columns.remove(self.target_column)
        
        # Create preprocessing for categorical features
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Create preprocessing for numerical features
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Combine preprocessing for categorical and numerical features
        self.preprocessing_pipeline = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_columns),
                ('cat', categorical_transformer, self.categorical_columns)
            ],
            remainder='drop'  # Drop any other columns
        )
        
        # Return column information for later reference
        return {
            'numerical_columns': self.numerical_columns,
            'categorical_columns': self.categorical_columns,
            'target_column': self.target_column
        }
    
    def preprocess_data(self):
        """Apply preprocessing steps to the data using pipeline"""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_from_uci() or load_from_csv() first.")
        
        if self.preprocessing_pipeline is None:
            self.create_preprocessing_pipeline()
            
        # Convert target to binary format (for consistent encoding)
        y = (self.df[self.target_column].str.contains('>50K')).astype(int)
        
        # Extract features (all columns except target)
        X = self.df.drop(columns=[self.target_column])
        
        return X, y
    
    def train_test_split(self, test_size=0.2, random_state=42):
        """Split data into training and testing sets"""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_from_uci() or load_from_csv() first.")
        
        # Process data
        X, y = self.preprocess_data()
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        return X_train, X_test, y_train, y_test
    
    def fit_transform_train_data(self, X_train, y_train):
        """Fit the preprocessing pipeline on training data and transform it"""
        if self.preprocessing_pipeline is None:
            raise ValueError("Preprocessing pipeline not created. Call create_preprocessing_pipeline() first.")
        
        # Fit and transform the training data
        X_train_transformed = self.preprocessing_pipeline.fit_transform(X_train)
        
        return X_train_transformed, y_train
    
    def transform_test_data(self, X_test):
        """Transform test data using the fitted preprocessing pipeline"""
        if self.preprocessing_pipeline is None:
            raise ValueError("Preprocessing pipeline not fitted. Call fit_transform_train_data() first.")
        
        # Transform the test data
        X_test_transformed = self.preprocessing_pipeline.transform(X_test)
        
        return X_test_transformed
    
    def get_feature_names(self):
        """Get feature names after transformation"""
        if self.preprocessing_pipeline is None or not hasattr(self.preprocessing_pipeline, 'named_transformers_'):
            raise ValueError("Pipeline not fitted yet. Call fit_transform_train_data() first.")
        
        # Get feature names from numerical transformer
        numerical_features = self.numerical_columns
        
        # Get feature names from OneHotEncoder
        categorical_transformer = self.preprocessing_pipeline.named_transformers_['cat']
        onehot_encoder = categorical_transformer.named_steps['onehot']
        categorical_features = []
        
        if hasattr(onehot_encoder, 'get_feature_names_out'):
            # For newer sklearn versions
            categorical_features = onehot_encoder.get_feature_names_out(self.categorical_columns).tolist()
        else:
            # For older sklearn versions
            categorical_features = onehot_encoder.get_feature_names(self.categorical_columns).tolist()
        
        return numerical_features + categorical_features
    
    def save_metadata(self, output_dir):
        """Save metadata information"""
        if self.metadata is None:
            raise ValueError("Metadata not available. Load data first.")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metadata
        metadata_file = os.path.join(output_dir, "adult_metadata.json")
        with open(metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=2)
        
        # Save column information
        if self.categorical_columns is not None and self.numerical_columns is not None:
            column_info = {
                "categorical_columns": self.categorical_columns,
                "numerical_columns": self.numerical_columns,
                "target_column": self.target_column
            }
            
            columns_file = os.path.join(output_dir, "column_info.json")
            with open(columns_file, "w") as f:
                json.dump(column_info, f, indent=2)
        
        # Save unique values for categorical columns
        if self.df is not None:
            unique_values = {}
            for col in self.categorical_columns:
                if col in self.df.columns:
                    unique_values[col] = self.df[col].dropna().unique().tolist()
                
            unique_values_file = os.path.join(output_dir, "unique_values.json")
            with open(unique_values_file, "w") as f:
                json.dump(unique_values, f, indent=2)
