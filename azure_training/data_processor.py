import numpy as np
import pandas as pd
import os
import json
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class AdultDataProcessor:
    """Handles loading and preprocessing of the Adult Income dataset"""
    
    def __init__(self):
        self.df = None
        self.X = None
        self.y = None
        self.metadata = None
        self.target_column = 'income'
        
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
    
    def preprocess_data(self):
        """Apply preprocessing steps to the data"""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_from_uci() or load_from_csv() first.")
        
        # Strip whitespace and trailing periods from income
        self.df[self.target_column] = self.df[self.target_column].str.strip().str.replace('.', '', regex=False)
        
        # Replace '?' with np.nan for consistent missing value notation
        self.df.replace('?', np.nan, inplace=True)
        
        # Handle missing values
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            self.df[col].fillna('Unknown', inplace=True)
        
        # One-Hot Encode categorical features
        self.df_encoded = pd.get_dummies(self.df, drop_first=True)

        # Convert all integer columns to float64 to safely handle missing values
        int_cols = self.df_encoded.select_dtypes(include=['int64']).columns
        if len(int_cols) > 0:
            self.df_encoded[int_cols] = self.df_encoded[int_cols].astype('float64')
        
        # The target column gets converted to "income_>50K" in one-hot encoding
        self.target_column = next(col for col in self.df_encoded.columns if col.startswith('income_'))
        
        return self.df_encoded
    
    def train_test_split(self, test_size=0.2, random_state=42):
        """Split data into training and testing sets"""
        if not hasattr(self, 'df_encoded') or self.df_encoded is None:
            raise ValueError("Data not processed. Call preprocess_data() first.")
        
        X = self.df_encoded.drop(columns=[self.target_column])
        y = self.df_encoded[self.target_column]

        # Double-check integer columns are converted to float64 before splitting
        int_cols = X.select_dtypes(include=['int64']).columns
        if len(int_cols) > 0:
            X[int_cols] = X[int_cols].astype('float64')
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(self, X_train, X_test, columns=None):
        """Scale numerical features"""
        if columns is None:
            # Default to all numerical columns
            columns = X_train.select_dtypes(include=['int64', 'float64']).columns
        
        scaler = StandardScaler()
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        X_train_scaled[columns] = scaler.fit_transform(X_train[columns])
        X_test_scaled[columns] = scaler.transform(X_test[columns])
        
        return X_train_scaled, X_test_scaled, scaler
    
    def save_metadata(self, output_dir):
        """Save metadata information"""
        if self.metadata is None:
            raise ValueError("Metadata not available. Load data first.")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metadata
        metadata_file = os.path.join(output_dir, "adult_metadata.json")
        with open(metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=2)
        
        # Save unique values for categorical columns
        if self.df is not None:
            unique_values = {}
            categorical_cols = self.df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                unique_values[col] = self.df[col].dropna().unique().tolist()
                
            unique_values_file = os.path.join(output_dir, "unique_values.json")
            with open(unique_values_file, "w") as f:
                json.dump(unique_values, f, indent=2)
