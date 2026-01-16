# TDOT Pavement Marking Price Prediction

A machine learning project for predicting pavement marking costs using Tennessee Department of Transportation (TDOT) historical data.

## Overview

This project builds a CatBoost regression model to predict pavement marking prices based on various features including item descriptions, project quantities, location, and temporal factors. The model achieves **66.92% R² accuracy** on 2025 test data.

## Project Structure

- `train_pavement_marking_model.py` - Main ML pipeline for training and evaluation
- `categorize_projects.py` - Project work type classification system  
- `test_xgboost_compatibility.py` - XGBoost algorithm comparison test
- `parse_all_bid_tabs.py` - Data parsing and preprocessing utility
- `Data/` - Contains TDOT data files
- `categorized_items.csv` - Item type classifications
- `project_categories.csv` - Project work type categories

## Model Performance

### CatBoost Model Results
- **R² Score**: 66.92%
- **MAE**: $675.27
- **RMSE**: Optimized for pavement marking predictions
- **Training Data**: 80,505 records (2014-2024)
- **Test Data**: 7,247 records (2025)

### Feature Importance
1. **item_description_expanded** (69.7%) - Specific pavement marking types
2. **log_project_qty** (20.4%) - Project scale (log-transformed)
3. **work_type** (4.7%) - Project category classification
4. **item_type** (2.6%) - Pavement marking category
5. **primary_county** (1.3%) - Geographic location
6. **year** (0.8%) - Temporal trends
7. **quarter** (0.3%) - Seasonal effects
8. **month** (0.2%) - Monthly patterns

## Features

### Data Processing
- **Item Classification**: 4 main pavement marking categories
- **Work Type Classification**: 14 project work types with pattern matching
- **Log Transformations**: Quantity normalization for better model performance
- **Time-based Splits**: Historical data for training, recent data for validation

### Model Features
- **9 engineered features** including categorical and numerical variables
- **Categorical encoding** handled natively by CatBoost
- **Early stopping** and **cross-validation** for optimal performance
- **Comprehensive evaluation** with multiple metrics

### Visualizations
The model generates 8 comprehensive plots:
1. **Actual vs Predicted** scatter plot
2. **Residuals Analysis** for error patterns
3. **Feature Importance** ranking
4. **Prediction Distribution** histograms
5. **Temporal Analysis** by year and quarter
6. **Geographic Analysis** by county
7. **Item Type Performance** breakdown
8. **Work Type Analysis** by project category

## Algorithm Comparison

| Algorithm | R² Score | MAE | Performance |
|-----------|----------|-----|-------------|
| **CatBoost** | **66.92%** | **$675.27** | ✅ **Recommended** |
| XGBoost | 6.81% | $1,036.68 | ❌ Poor performance |

**CatBoost outperforms XGBoost** due to superior categorical variable handling for this dataset.

## Requirements

```
pandas
numpy  
catboost
scikit-learn
matplotlib
xgboost (for compatibility testing)
```

## Usage

1. **Train the main model:**
   ```bash
   python train_pavement_marking_model.py
   ```

2. **Categorize projects:**
   ```bash
   python categorize_projects.py
   ```

3. **Test XGBoost compatibility:**
   ```bash
   python test_xgboost_compatibility.py
   ```

## Data Requirements

- **TDOT_data.csv**: Main dataset with pavement marking records
- **Categorical columns**: item_description, primary_county
- **Numerical columns**: project_qty, award_year, award_month
- **Target variable**: unit_cost

## Model Architecture

```
Input Features (9) → CatBoost Regressor → Price Prediction
                   ↓
Feature Engineering:
- Item type categorization
- Work type classification  
- Log quantity transformation
- Temporal feature extraction
```

## Results Summary

The CatBoost model successfully predicts pavement marking costs with **high accuracy (66.92% R²)** and provides comprehensive insights through feature importance analysis and visualization. The model is production-ready for TDOT cost estimation workflows.

## Future Enhancements

- Hyperparameter tuning for further optimization
- Additional feature engineering opportunities
- Integration with real-time bidding systems
- Expanded geographic analysis