"""
Utility Data Processing Module

Handles data cleaning, feature extraction, encoding, and preprocessing for utility item analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from sklearn.preprocessing import LabelEncoder


class UtilityDataProcessor:
    """
    Handles data preprocessing for utility item analysis.
    
    Responsibilities:
    - Data validation and cleaning
    - Feature extraction and encoding
    - Missing value handling
    - Data preprocessing pipeline
    """
    
    def __init__(self, 
                 data: pd.DataFrame, 
                 item_type: str,
                 target_column: str = 'unit_price',
                 id_column: str = 'ITEM',
                 geographic_column: str = 'county'):
        """
        Initialize the data processor.
        
        Args:
            data: DataFrame containing the utility item data
            item_type: Type of utility item (e.g., 'POLE', 'TRANSFORMER')
            target_column: Column name for the target variable
            id_column: Column name for unique item identifier
            geographic_column: Column name for geographic grouping
        """
        self.data = data.copy()
        self.item_type = item_type.upper()
        self.target_column = target_column
        self.id_column = id_column
        self.geographic_column = geographic_column
        
        # Processing results
        self.X = None
        self.y = None
        self.encoders = {}
        self.processed_data = None
        
        print(f"📦 Initialized data processor for {self.item_type}")
        print(f"   Data shape: {self.data.shape}")
        print(f"   Target column: {self.target_column}")
    
    def validate_data(self) -> Dict[str, Any]:
        """
        Validate the input data and return summary.
        
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'total_rows': len(self.data),
            'total_columns': len(self.data.columns),
            'missing_target': self.data[self.target_column].isna().sum() if self.target_column in self.data.columns else 'Column not found',
            'duplicate_rows': self.data.duplicated().sum(),
            'numeric_columns': list(self.data.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(self.data.select_dtypes(include=['object']).columns),
            'missing_values_summary': self.data.isna().sum().to_dict()
        }
        
        print(f"📋 Data Validation Results:")
        print(f"   Rows: {validation_results['total_rows']:,}")
        print(f"   Columns: {validation_results['total_columns']}")
        print(f"   Missing target values: {validation_results['missing_target']}")
        print(f"   Duplicate rows: {validation_results['duplicate_rows']}")
        print(f"   Numeric columns: {len(validation_results['numeric_columns'])}")
        print(f"   Categorical columns: {len(validation_results['categorical_columns'])}")
        
        return validation_results
    
    def identify_feature_columns(self) -> Dict[str, List[str]]:
        """
        Identify and categorize potential feature columns.
        
        Returns:
            Dictionary with categorized column lists
        """
        # Exclude target and ID columns
        excluded_cols = {self.target_column, self.id_column}
        
        numeric_features = []
        categorical_features = []
        
        for col in self.data.columns:
            if col in excluded_cols:
                continue
                
            if self.data[col].dtype in ['int64', 'float64']:
                # Check if it has reasonable numeric characteristics
                if self.data[col].nunique() > 1:  # Has variation
                    numeric_features.append(col)
            elif self.data[col].dtype == 'object':
                # Include categorical with reasonable cardinality
                unique_values = self.data[col].nunique()
                if 1 < unique_values < 50:  # Reasonable cardinality
                    categorical_features.append(col)
        
        feature_info = {
            'numeric': numeric_features,
            'categorical': categorical_features,
            'excluded': list(excluded_cols)
        }
        
        print(f"🔍 Feature Analysis:")
        print(f"   Numeric features ({len(numeric_features)}): {numeric_features}")
        print(f"   Categorical features ({len(categorical_features)}): {categorical_features}")
        
        return feature_info
    
    def handle_missing_values(self, 
                            method: str = 'drop',
                            numeric_fill: str = 'median',
                            categorical_fill: str = 'mode') -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            method: 'drop' or 'impute'
            numeric_fill: 'mean', 'median', or specific value
            categorical_fill: 'mode' or specific value
            
        Returns:
            DataFrame with missing values handled
        """
        df = self.data.copy()
        
        print(f"🔧 Handling missing values using '{method}' method...")
        
        if method == 'drop':
            # Drop rows where target is missing
            initial_rows = len(df)
            df = df.dropna(subset=[self.target_column])
            
            # Drop columns with too many missing values (>50%)
            missing_threshold = 0.5
            cols_to_keep = []
            
            for col in df.columns:
                if col in [self.target_column, self.id_column]:
                    cols_to_keep.append(col)
                    continue
                    
                missing_pct = df[col].isna().sum() / len(df)
                if missing_pct <= missing_threshold:
                    cols_to_keep.append(col)
                else:
                    print(f"   Dropping column '{col}' ({missing_pct:.1%} missing)")
            
            df = df[cols_to_keep]
            print(f"   Rows after cleaning: {len(df)} (removed {initial_rows - len(df)})")
            
        elif method == 'impute':
            # Impute missing values
            for col in df.columns:
                if col in [self.target_column, self.id_column]:
                    continue
                    
                if df[col].dtype in ['int64', 'float64']:
                    # Numeric imputation
                    if numeric_fill == 'mean':
                        fill_value = df[col].mean()
                    elif numeric_fill == 'median':
                        fill_value = df[col].median()
                    else:
                        fill_value = numeric_fill
                    df[col].fillna(fill_value, inplace=True)
                    
                else:
                    # Categorical imputation
                    if categorical_fill == 'mode':
                        mode_val = df[col].mode()
                        fill_value = mode_val[0] if len(mode_val) > 0 else 'Unknown'
                    else:
                        fill_value = categorical_fill
                    df[col].fillna(fill_value, inplace=True)
        
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features for machine learning.
        
        Args:
            df: DataFrame with categorical features
            
        Returns:
            DataFrame with encoded categorical features
        """
        print(f"🏷️ Encoding categorical features...")
        
        df_encoded = df.copy()
        
        for col in df.columns:
            if df[col].dtype == 'object' or str(df[col].dtype) == 'category':
                if col in [self.target_column, self.id_column]:
                    continue
                    
                print(f"   Encoding {col} ({df[col].nunique()} unique values)")
                
                # Use label encoding
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    try:
                        df_encoded[f'{col}_encoded'] = self.encoders[col].fit_transform(df[col].astype(str))
                    except ValueError as e:
                        print(f"   Warning: Could not encode {col}: {e}")
                        continue
                else:
                    # Transform using existing encoder
                    try:
                        df_encoded[f'{col}_encoded'] = self.encoders[col].transform(df[col].astype(str))
                    except ValueError:
                        # Handle unseen categories
                        print(f"   Warning: Unseen categories in {col}, using default encoding")
                        df_encoded[f'{col}_encoded'] = 0
        
        return df_encoded
    
    def prepare_features(self, 
                        handle_missing: str = 'drop',
                        include_original: bool = True) -> None:
        """
        Complete feature preparation pipeline.
        
        Args:
            handle_missing: How to handle missing values ('drop', 'impute')
            include_original: Whether to include original columns in output
        """
        print(f"🔧 Preparing features for {self.item_type} analysis...")
        
        # Step 1: Validate data
        validation_results = self.validate_data()
        
        # Step 2: Handle missing values
        df_clean = self.handle_missing_values(method=handle_missing)
        
        # Step 3: Encode categorical features
        df_encoded = self.encode_categorical_features(df_clean)
        
        # Step 4: Select final feature set
        feature_columns = []
        for col in df_encoded.columns:
            if col in [self.target_column, self.id_column]:
                continue
                
            # Include numeric columns
            if df_encoded[col].dtype in ['int64', 'float64']:
                # Check for variation
                if df_encoded[col].nunique() > 1 and df_encoded[col].std() > 0:
                    feature_columns.append(col)
        
        # Step 5: Create final feature matrix
        if feature_columns:
            self.X = df_encoded[feature_columns].fillna(0)  # Final safety fillna
            self.y = df_encoded[self.target_column]
            
            # Align target with features
            valid_indices = self.X.index.intersection(self.y.index)
            self.X = self.X.loc[valid_indices]
            self.y = self.y.loc[valid_indices]
            
        else:
            print("⚠️ No valid features found!")
            self.X = pd.DataFrame(index=df_encoded.index)
            self.y = df_encoded[self.target_column]
        
        # Store processed data
        self.processed_data = df_encoded
        
        print(f"✅ Feature preparation complete:")
        print(f"   Final feature matrix shape: {self.X.shape}")
        print(f"   Features: {list(self.X.columns)}")
        print(f"   Target variable range: ${self.y.min():.2f} - ${self.y.max():.2f}")
    
    def get_feature_summary(self) -> pd.DataFrame:
        """
        Get a summary of the prepared features.
        
        Returns:
            DataFrame with feature statistics
        """
        if self.X is None:
            raise ValueError("Must call prepare_features() first")
        
        summary_data = []
        
        for col in self.X.columns:
            summary_data.append({
                'feature': col,
                'dtype': str(self.X[col].dtype),
                'unique_values': self.X[col].nunique(),
                'missing_count': self.X[col].isna().sum(),
                'min': self.X[col].min(),
                'max': self.X[col].max(),
                'mean': self.X[col].mean(),
                'std': self.X[col].std()
            })
        
        return pd.DataFrame(summary_data)