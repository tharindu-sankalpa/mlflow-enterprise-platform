"""
Script to visualize the effects of custom transformers on feature distributions.
This demonstrates how the transformations improve the distributions for modeling.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Import custom transformers
from data_processor import ZeroAwareLogTransformer

def load_data():
    """Load Adult Income dataset from UCI repository"""
    adult = fetch_ucirepo(id=2)
    df = pd.concat([adult.data.features, adult.data.targets], axis=1)
    
    # Replace '?' with np.nan for consistent missing value notation
    df.replace('?', np.nan, inplace=True)
    
    return df

def apply_transformations(df):
    """Apply different transformations to different columns and return the transformed data"""
    transformed_df = df.copy()
    
    # 1. Log transform for fnlwgt
    log_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('log', ZeroAwareLogTransformer())
    ])
    transformed_df['fnlwgt_transformed'] = log_transformer.fit_transform(df[['fnlwgt']])
    
    # 2. Zero-aware log transform for capital-gain and capital-loss
    for column in ['capital-gain', 'capital-loss']:
        zero_aware_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
            ('zero_log', ZeroAwareLogTransformer())
        ])
        transformed_col = zero_aware_transformer.fit_transform(df[[column]])
        transformed_df[f'{column}_transformed'] = transformed_col
    
    # 3. Power transform for age
    power_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('power', PowerTransformer(method='yeo-johnson'))
    ])
    transformed_df['age_transformed'] = power_transformer.fit_transform(df[['age']])
    
    return transformed_df

def plot_transformations(df):
    """Plot original vs transformed distributions"""
    columns_to_transform = {
        'fnlwgt': 'Log Transform',
        'capital-gain': 'Zero-Aware Log Transform',
        'capital-loss': 'Zero-Aware Log Transform',
        'age': 'Power Transform (Yeo-Johnson)'
    }
    
    # Create a figure with subplots
    fig, axes = plt.subplots(len(columns_to_transform), 2, figsize=(12, 4*len(columns_to_transform)))
    
    for i, (column, transform_name) in enumerate(columns_to_transform.items()):
        # Original distribution
        non_zero_count = (df[column] > 0).sum()
        zero_count = (df[column] == 0).sum()
        zero_percentage = 100 * zero_count / len(df) if len(df) > 0 else 0
        
        axes[i, 0].hist(df[column], bins=50)
        axes[i, 0].set_title(f'Original {column}\n(Zeros: {zero_percentage:.1f}%)')
        
        # Transformed distribution
        transformed_col = f'{column}_transformed'
        axes[i, 1].hist(df[transformed_col], bins=50)
        axes[i, 1].set_title(f'After {transform_name}')
    
    plt.tight_layout()
    plt.savefig('transformation_effects.png')
    plt.show()
    
    print(f"Visualization saved to 'transformation_effects.png'")

def main():
    """Main function to load data, apply transformations and visualize results"""
    print("Loading Adult Income dataset...")
    df = load_data()
    
    print("Applying transformations...")
    transformed_df = apply_transformations(df)
    
    print("Plotting transformations...")
    plot_transformations(transformed_df)
    
    # Calculate and print skewness before and after transformation
    print("\nSkewness comparison (before → after):")
    for column in ['fnlwgt', 'capital-gain', 'capital-loss', 'age']:
        original_skew = df[column].skew()
        transformed_skew = transformed_df[f'{column}_transformed'].skew()
        improvement = 100 * (abs(original_skew) - abs(transformed_skew)) / abs(original_skew) if abs(original_skew) > 0 else 0
        
        print(f"{column}: {original_skew:.2f} → {transformed_skew:.2f} ({improvement:.1f}% improvement)")

if __name__ == "__main__":
    main()