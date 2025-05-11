import numpy as np
import pandas as pd
import os
import json
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin


# Define operation functions for feature combinations
# These need to be at the module level to be picklable
def ratio_operation(x, y):
    """Compute ratio with protection against division by zero"""
    return x / (y + 0.1)

def subtract_operation(x, y):
    """Compute difference between two features"""
    return x - y

def multiply_operation(x, y):
    """Compute product of two features"""
    return x * y


class IntToFloatTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer to convert integer columns to float64 to handle missing values properly.
    This prevents MLflow schema enforcement errors during inference.
    
    MLflow's schema inference detects integer columns, but integers cannot represent
    missing values in Python/NumPy. Converting integers to float64 ensures that:
    1. Missing values can be properly represented as NaN
    2. The schema used during inference will match the actual data types
    3. No schema enforcement errors will occur if missing values appear
    """
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X_copy = X.copy()
        for col in X_copy.select_dtypes(include=['int64']).columns:
            X_copy[col] = X_copy[col].astype('float64')
        return X_copy


class ZeroAwareLogTransformer(BaseEstimator, TransformerMixin):
    """
    Apply logarithmic transformation to columns with special handling for zeros.
    
    This transformer applies log(x + 1) to non-zero values and keeps zeros as zeros.
    This is useful for financial features like capital-gain and capital-loss where:
    1. Most values are zeros (no gain/loss)
    2. Non-zero values have high skew and wide range
    3. The distinction between zero and non-zero is meaningful
    
    Parameters:
        columns (list): List of column names to transform (only used with DataFrame input)
    """
    def __init__(self, columns=None):
        self.columns = columns
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        # Check if X is a DataFrame or a numpy array
        if hasattr(X, 'columns'):
            # DataFrame case
            X_copy = X.copy()
            columns_to_transform = self.columns or X_copy.columns
            
            for col in columns_to_transform:
                if col in X_copy.columns:
                    # Create mask for non-zero values
                    non_zero_mask = X_copy[col] > 0
                    
                    # Apply log transformation only to non-zero values
                    if non_zero_mask.any():
                        X_copy.loc[non_zero_mask, col] = np.log1p(X_copy.loc[non_zero_mask, col])
            
            return X_copy
        else:
            # NumPy array case
            X_copy = X.copy()
            
            # Create mask for non-zero values
            non_zero_mask = X_copy > 0
            
            # Apply log transformation only to non-zero values
            if np.any(non_zero_mask):
                X_copy[non_zero_mask] = np.log1p(X_copy[non_zero_mask])
                
            return X_copy
    
    def get_feature_names_out(self, input_features=None):
        return self.columns or input_features


class ColumnSelector(BaseEstimator, TransformerMixin):
    """
    Select specific columns from a DataFrame.
    
    This transformer selects specific columns from a DataFrame, which is useful for
    applying different transformations to different subsets of numerical features.
    
    Parameters:
        columns (list): List of column names to select
    """
    def __init__(self, columns):
        self.columns = columns
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X_subset = X[self.columns].copy()
        return X_subset
    
    def get_feature_names_out(self, input_features=None):
        return self.columns


class FeatureCombiner(BaseEstimator, TransformerMixin):
    """
    Create new features by combining existing ones with specified operations.
    
    This transformer creates new interaction features between variables which might
    reveal additional patterns in the data. It supports various operations such as
    ratios, products, and differences between numeric features.
    
    Parameters:
        combinations (list of dicts): Each dict contains:
            - name: Name of the new feature
            - columns: List of columns to combine
            - operation: Function to apply (must be a picklable function)
    """
    def __init__(self, combinations):
        self.combinations = combinations
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X_copy = X.copy()
        result_df = pd.DataFrame(index=X_copy.index)
        
        for combo in self.combinations:
            cols = combo['columns']
            name = combo['name']
            operation = combo['operation']
            
            # Skip if any column is missing
            if not all(col in X_copy.columns for col in cols):
                continue
                
            # Apply the operation
            if len(cols) == 2:
                # Binary operation (most common case)
                result_df[name] = operation(X_copy[cols[0]], X_copy[cols[1]])
            else:
                # For operations with more than 2 columns
                result = X_copy[cols[0]].copy()
                for col in cols[1:]:
                    result = operation(result, X_copy[col])
                result_df[name] = result
                
        return result_df
    
    def get_feature_names_out(self, input_features=None):
        return [combo['name'] for combo in self.combinations]


class AdultDataProcessor:
    """
    Handles loading and preprocessing of the Adult Income dataset.
    
    This class provides a complete pipeline for working with the Adult Income dataset, 
    including data loading, preprocessing, feature engineering, and splitting the data
    into training and testing sets. It handles both categorical and numerical features
    appropriately.
    
    Attributes:
        df (pandas.DataFrame): The loaded dataset
        X (pandas.DataFrame): Feature data
        y (pandas.DataFrame): Target data
        metadata (dict): Dataset metadata
        target_column (str): Name of the target column (default: 'income')
        preprocessing_pipeline (ColumnTransformer): Sklearn preprocessing pipeline
        categorical_columns (list): Names of categorical columns
        numerical_columns (list): Names of numerical columns
    """
    
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
        """
        Load data directly from UCI repository.
        
        Fetches the Adult Income dataset from the UCI Machine Learning Repository
        and initializes the internal dataframes and metadata.
        
        Returns:
            pandas.DataFrame: The loaded dataset
        """
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

        # To prvent MLflow schema enforcement errors, convert integer columns to float64
        # int_columns = self.df.select_dtypes(include='int64').columns
        # self.df[int_columns] = self.df[int_columns].astype('float64')
        
        return self.df
    
    def load_from_csv(self, file_path):
        """
        Load data from a CSV file.
        
        Parameters:
            file_path (str): Path to the CSV file containing the dataset
            
        Returns:
            pandas.DataFrame: The loaded dataset
        """
        self.df = pd.read_csv(file_path)
        return self.df
    
    def create_preprocessing_pipeline(self):
        """
        Create preprocessing pipeline using sklearn components.
        
        This method builds a comprehensive data preprocessing pipeline that:
        1. Handles missing values in both numerical and categorical features
        2. Converts integer columns to float64 to handle missing values properly
        3. Applies appropriate transformations to correct data distributions:
           - Log transform for highly skewed features (fnlwgt)
           - Zero-aware log transform for financial features (capital-gain, capital-loss)
           - Power transform for moderately skewed features (age)
        4. Standardizes numerical features
        5. One-hot encodes categorical features
        6. Creates interaction features to capture relationships between variables
        
        The pipeline uses scikit-learn's ColumnTransformer to apply different
        transformations to different column groups.
        
        Returns:
            dict: Dictionary containing column information:
                - numerical_columns: List of numerical feature names
                - categorical_columns: List of categorical feature names
                - target_column: Name of the target column
                
        Raises:
            ValueError: If data hasn't been loaded yet
        """
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
        
        # Group numerical features by appropriate transformation type
        log_columns = ['fnlwgt']
        zero_aware_log_columns = ['capital-gain', 'capital-loss']
        power_columns = ['age']
        regular_columns = [col for col in self.numerical_columns 
                          if col not in log_columns + zero_aware_log_columns + power_columns]
        
        # Create specific transformers for different numerical column groups
        log_transformer = Pipeline(steps=[
            ('selector', ColumnSelector(log_columns)),
            ('imputer', SimpleImputer(strategy='median')),
            ('log', ZeroAwareLogTransformer()),  # Apply log transform to all selected columns
            ('scaler', StandardScaler())
        ])
        
        zero_aware_transformer = Pipeline(steps=[
            ('selector', ColumnSelector(zero_aware_log_columns)),
            ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
            ('zero_log', ZeroAwareLogTransformer()),  # Special handling for zeros
            ('scaler', StandardScaler())
        ])
        
        power_transformer = Pipeline(steps=[
            ('selector', ColumnSelector(power_columns)),
            ('imputer', SimpleImputer(strategy='median')),
            ('power', PowerTransformer(method='yeo-johnson')),  # Yeo-Johnson works with negative values
            ('scaler', StandardScaler())
        ])
        
        regular_transformer = Pipeline(steps=[
            ('selector', ColumnSelector(regular_columns)),
            ('int_to_float', IntToFloatTransformer()),  # Convert integers to float64
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Define feature combinations for interaction features
        # These are engineered features that combine multiple variables
        feature_combinations = [
            # Age-Education ratio - income tends to increase with both age and education
            {'name': 'age_education_ratio', 
             'columns': ['age', 'education-num'], 
             'operation': ratio_operation},  # Use named function instead of lambda
            
            # Age-Hours ratio - captures work-life balance which may correlate with income
            {'name': 'age_hours_ratio', 
             'columns': ['age', 'hours-per-week'], 
             'operation': ratio_operation},  # Use named function instead of lambda
            
            # Capital difference - net capital movement may be more informative than separate values
            {'name': 'capital_diff', 
             'columns': ['capital-gain', 'capital-loss'], 
             'operation': subtract_operation},  # Use named function instead of lambda
             
            # Hours-Education product - highly educated people working many hours likely earn more
            {'name': 'hours_education_product', 
             'columns': ['hours-per-week', 'education-num'], 
             'operation': multiply_operation}  # Use named function instead of lambda
        ]
        
        # Feature combiner transformer
        feature_combiner = FeatureCombiner(combinations=feature_combinations)
        
        # Preprocessing for combined features
        interaction_transformer = Pipeline(steps=[
            ('combiner', feature_combiner),
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Main preprocessing columns
        main_columns = log_columns + zero_aware_log_columns + power_columns + regular_columns
        
        # Combine all transformers
        self.preprocessing_pipeline = ColumnTransformer(
            transformers=[
                ('log', log_transformer, log_columns),
                ('zero_aware', zero_aware_transformer, zero_aware_log_columns),
                ('power', power_transformer, power_columns),
                ('regular', regular_transformer, regular_columns),
                ('cat', categorical_transformer, self.categorical_columns),
                ('interactions', interaction_transformer, main_columns)
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
        """
        Apply preprocessing steps to the data using pipeline.
        
        This method applies the preprocessing pipeline to transform the raw data
        into a format suitable for machine learning. The target variable is also
        converted to a binary format.
        
        Returns:
            tuple: (X, y) where X is the feature DataFrame and y is the target series
            
        Raises:
            ValueError: If data hasn't been loaded yet
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_from_uci() or load_from_csv() first.")
        
        if self.preprocessing_pipeline is None:
            self.create_preprocessing_pipeline()
            
        # Convert target to binary format (for consistent encoding)
        y = (self.df[self.target_column].str.contains('>50K')).astype(int)
        
        # Extract features (all columns except target)
        X = self.df.drop(columns=[self.target_column])
        
        # No need for manual conversion - the pipeline will handle it
        return X, y
    
    def train_test_split(self, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets.
        
        Parameters:
            test_size (float): Proportion of the dataset to include in the test split (default: 0.2)
            random_state (int): Random seed for reproducibility (default: 42)
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test) where
                - X_train: Training features
                - X_test: Testing features
                - y_train: Training target
                - y_test: Testing target
                
        Raises:
            ValueError: If data hasn't been loaded yet
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_from_uci() or load_from_csv() first.")
        
        # Process data
        X, y = self.preprocess_data()
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        return X_train, X_test, y_train, y_test
    
    def fit_transform_train_data(self, X_train, y_train):
        """
        Fit the preprocessing pipeline on training data and transform it.
        
        Parameters:
            X_train (pandas.DataFrame): Training feature data
            y_train (pandas.Series): Training target data
            
        Returns:
            tuple: (X_train_transformed, y_train) where X_train_transformed is the
                  transformed training feature data
                  
        Raises:
            ValueError: If preprocessing pipeline hasn't been created
        """
        if self.preprocessing_pipeline is None:
            raise ValueError("Preprocessing pipeline not created. Call create_preprocessing_pipeline() first.")
        
        # Fit and transform the training data
        X_train_transformed = self.preprocessing_pipeline.fit_transform(X_train)
        
        return X_train_transformed, y_train
    
    def transform_test_data(self, X_test):
        """
        Transform test data using the fitted preprocessing pipeline.
        
        Parameters:
            X_test (pandas.DataFrame): Testing feature data
            
        Returns:
            numpy.ndarray: Transformed testing feature data
            
        Raises:
            ValueError: If preprocessing pipeline hasn't been fitted yet
        """
        if self.preprocessing_pipeline is None:
            raise ValueError("Preprocessing pipeline not fitted. Call fit_transform_train_data() first.")
        
        # Transform the test data
        X_test_transformed = self.preprocessing_pipeline.transform(X_test)
        
        return X_test_transformed
    
    def get_feature_names(self):
        """
        Get feature names after transformation.
        
        This method extracts the feature names from the preprocessing pipeline after
        transformation. This is particularly important after one-hot encoding of
        categorical features, as it provides the names of the generated binary features.
        
        The returned feature names can be used to create a DataFrame from the
        transformed feature matrix, making the model results more interpretable.
        
        Returns:
            list: List of feature names after preprocessing transformation
            
        Raises:
            ValueError: If the preprocessing pipeline hasn't been fitted yet
        """
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
        """
        Save metadata information to JSON files.
        
        This method saves various metadata about the dataset and preprocessing to the
        specified directory, including:
        - Dataset metadata
        - Column information (categorical, numerical, target)
        - Unique values for categorical columns
        
        Parameters:
            output_dir (str): Directory path where metadata files will be saved
            
        Raises:
            ValueError: If metadata is not available (data not loaded)
        """
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
