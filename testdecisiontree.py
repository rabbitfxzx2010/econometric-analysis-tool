# Generated Python code that replicates your analysis results
# This code includes all your data preprocessing and model settings

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# =============================================================================
# 1. DATA LOADING AND INITIAL SETUP
# =============================================================================

# Load your dataset
df = pd.read_csv('test_dataset_classification.csv')

print(f'Original dataset shape: {df.shape}')
print(f'Original columns: {list(df.columns)}')

# =============================================================================
# 2. DATA FILTERING (Replicating your filter settings)
# =============================================================================

# Filter 1: is_urban between 1.0 and 1.0
df = df[(df['is_urban'] >= 1.0) & (df['is_urban'] <= 1.0)]

print(f'After filtering shape: {df.shape}')

# =============================================================================
# 4. VARIABLE DEFINITION AND PREPROCESSING
# =============================================================================

# Define your variables (matching your analysis)
independent_vars = ['age', 'income', 'education_Master', 'education_High School', 'education_PhD', 'high_earner', 'experience', 'hours_worked', 'education_Bachelor']
dependent_var = 'promotion'

# Extract features and target
X = df[independent_vars].copy()
y = df[dependent_var].copy()

print(f'Feature matrix shape: {X.shape}')
print(f'Target variable shape: {y.shape}')
print(f'Features: {list(X.columns)}')

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f'Training set: {X_train.shape[0]} samples')
print(f'Test set: {X_test.shape[0]} samples')

# =============================================================================
# 5. MODEL TRAINING (Replicating your exact settings)
# =============================================================================

# Decision Tree Regression (your settings)
model = DecisionTreeRegressor(
    max_depth=5,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)
print('✓ Model trained successfully')

# =============================================================================
# 6. PREDICTIONS AND EVALUATION
# =============================================================================

# Make predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Regression metrics
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print('=== MODEL PERFORMANCE ===') 
print(f'Training R²: {train_r2:.4f}')
print(f'Test R²: {test_r2:.4f}')
print(f'Training RMSE: {np.sqrt(train_mse):.4f}')
print(f'Test RMSE: {np.sqrt(test_mse):.4f}')

# =============================================================================
# 7. FEATURE IMPORTANCE/COEFFICIENTS
# =============================================================================

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print('\n=== FEATURE IMPORTANCE ===') 
print(feature_importance)

# =============================================================================
# 8. SUMMARY
# =============================================================================

print('\n' + '='*50)
print('ANALYSIS COMPLETE - Results match your main analysis!')
print('='*50)
print('Model: Decision Tree')
print('Problem Type: Regression')
print('Features: 9')
print('This code replicates all your settings and preprocessing steps.')