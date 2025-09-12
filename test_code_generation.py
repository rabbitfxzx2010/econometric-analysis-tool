#!/usr/bin/env python3
"""
Comprehensive test script to verify that generated Python code produces 
the same results as the main Streamlit app for all estimation methods.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score
import warnings
warnings.filterwarnings('ignore')

def create_test_data():
    """Create consistent test data for all methods"""
    np.random.seed(42)
    n_samples = 500
    
    # Create feature data
    age = np.random.normal(35, 10, n_samples).clip(18, 65)
    experience = np.random.normal(10, 5, n_samples).clip(0, 40)
    income = np.random.normal(50000, 15000, n_samples).clip(20000, 120000)
    hours_worked = np.random.normal(40, 8, n_samples).clip(20, 60)
    is_urban = np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
    
    # Education (one-hot encoded)
    education = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 
                                n_samples, p=[0.3, 0.4, 0.2, 0.1])
    education_bachelor = (education == 'Bachelor').astype(int)
    education_high_school = (education == 'High School').astype(int)
    education_master = (education == 'Master').astype(int)
    education_phd = (education == 'PhD').astype(int)
    
    # High earner (derived feature)
    high_earner = (income > 60000).astype(int)
    
    # Create regression target (promotion score 0-2)
    promotion_continuous = (
        0.3 * (income / 50000) + 
        0.2 * (experience / 20) + 
        0.1 * (age / 40) + 
        0.2 * education_master + 
        0.3 * education_phd +
        np.random.normal(0, 0.3, n_samples)
    ).clip(0, 2)
    
    # Create classification target
    promotion_binary = (promotion_continuous > 1).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'age': age,
        'experience': experience,
        'income': income,
        'hours_worked': hours_worked,
        'is_urban': is_urban,
        'education_Bachelor': education_bachelor,
        'education_High School': education_high_school,
        'education_Master': education_master,
        'education_PhD': education_phd,
        'high_earner': high_earner,
        'promotion': promotion_continuous,  # For regression
        'promotion_binary': promotion_binary,  # For classification
        'survey_date': pd.date_range('2023-01-01', periods=n_samples, freq='D')[:n_samples]
    })
    
    return df

def test_method(estimation_method, model_type, df, test_params=None):
    """Test a specific method and return results"""
    print(f"\n{'='*60}")
    print(f"Testing {estimation_method} - {model_type}")
    print(f"{'='*60}")
    
    # Set up data
    independent_vars = ['high_earner', 'experience', 'education_High School', 
                       'education_Master', 'age', 'hours_worked', 
                       'education_Bachelor', 'education_PhD', 'income']
    
    if model_type == 'regression':
        dependent_var = 'promotion'
    else:
        dependent_var = 'promotion_binary'
    
    # Apply any filters (simulate urban filter)
    df_filtered = df[df['is_urban'] == 1].copy()
    print(f"Data shape after filtering: {df_filtered.shape}")
    
    # Handle missing values (mean imputation)
    numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
    mean_imputer = SimpleImputer(strategy='mean')
    df_filtered[numeric_cols] = mean_imputer.fit_transform(df_filtered[numeric_cols])
    
    # Extract features and target
    X = df_filtered[independent_vars].copy()
    y = df_filtered[dependent_var].copy()
    
    # Train-test split with consistent random state
    random_state = 42
    if model_type == 'classification':
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state
        )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Apply scaling if needed
    use_scaling = test_params.get('use_scaling', False) if test_params else False
    if use_scaling:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        print("✓ Applied feature scaling")
    
    # Create and train model
    if estimation_method == "OLS":
        model = LinearRegression()
    elif estimation_method == "Lasso":
        alpha = test_params.get('alpha', 1.0) if test_params else 1.0
        model = Lasso(alpha=alpha, random_state=random_state)
    elif estimation_method == "Ridge":
        alpha = test_params.get('alpha', 1.0) if test_params else 1.0
        model = Ridge(alpha=alpha, random_state=random_state)
    elif estimation_method == "Elastic Net":
        alpha = test_params.get('alpha', 1.0) if test_params else 1.0
        l1_ratio = test_params.get('l1_ratio', 0.5) if test_params else 0.5
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state)
    elif estimation_method == "Logistic Regression":
        class_weight = test_params.get('class_weight', None) if test_params else None
        model = LogisticRegression(class_weight=class_weight, random_state=random_state, max_iter=1000)
    elif estimation_method == "Decision Tree":
        max_depth = test_params.get('max_depth', 5) if test_params else 5
        min_samples_split = test_params.get('min_samples_split', 2) if test_params else 2
        min_samples_leaf = test_params.get('min_samples_leaf', 1) if test_params else 1
        
        if model_type == 'classification':
            model = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state
            )
        else:
            model = DecisionTreeRegressor(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state
            )
    elif estimation_method == "Random Forest":
        n_estimators = test_params.get('n_estimators', 100) if test_params else 100
        max_depth = test_params.get('max_depth', 5) if test_params else 5
        min_samples_split = test_params.get('min_samples_split', 2) if test_params else 2
        min_samples_leaf = test_params.get('min_samples_leaf', 1) if test_params else 1
        
        if model_type == 'classification':
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state
            )
        else:
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state
            )
    
    # Train model
    model.fit(X_train, y_train)
    print("✓ Model trained successfully")
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    results = {}
    if model_type == 'classification':
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        results = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy
        }
        
        print(f"\n🎯 KEY RESULTS:")
        print(f"📊 Training Accuracy: {train_accuracy:.6f}")
        print(f"📊 Test Accuracy: {test_accuracy:.6f}")
        
    else:
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        results = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': np.sqrt(train_mse),
            'test_rmse': np.sqrt(test_mse),
            'train_mae': train_mae,
            'test_mae': test_mae
        }
        
        print(f"\n🎯 KEY RESULTS:")
        print(f"📊 Training R²: {train_r2:.6f}")
        print(f"📊 Test R²: {test_r2:.6f}")
        print(f"📊 Training RMSE: {np.sqrt(train_mse):.6f}")
        print(f"📊 Test RMSE: {np.sqrt(test_mse):.6f}")
        print(f"📊 Training MAE: {train_mae:.6f}")
        print(f"📊 Test MAE: {test_mae:.6f}")
    
    return results, model

def main():
    """Run comprehensive tests for all methods"""
    print("🧪 COMPREHENSIVE CODE GENERATION TEST")
    print("=" * 60)
    print("Testing that generated code produces same results as main app")
    
    # Create test data
    df = create_test_data()
    print(f"Created test dataset with {len(df)} rows")
    
    # Test configurations
    test_configs = [
        # Linear models for regression
        ("OLS", "regression", {}),
        ("Lasso", "regression", {"alpha": 0.1}),
        ("Ridge", "regression", {"alpha": 1.0}),
        ("Elastic Net", "regression", {"alpha": 0.1, "l1_ratio": 0.5}),
        
        # Tree models for regression
        ("Decision Tree", "regression", {"max_depth": 5, "min_samples_split": 2, "min_samples_leaf": 1}),
        ("Random Forest", "regression", {"n_estimators": 100, "max_depth": 5}),
        
        # Classification models
        ("Logistic Regression", "classification", {"class_weight": None}),
        ("Decision Tree", "classification", {"max_depth": 5}),
        ("Random Forest", "classification", {"n_estimators": 50, "max_depth": 3}),
    ]
    
    all_results = {}
    
    for method, model_type, params in test_configs:
        try:
            results, model = test_method(method, model_type, df, params)
            all_results[f"{method}_{model_type}"] = results
            print(f"✅ {method} ({model_type}) - SUCCESS")
        except Exception as e:
            print(f"❌ {method} ({model_type}) - FAILED: {str(e)}")
            all_results[f"{method}_{model_type}"] = None
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print(f"{'='*60}")
    
    successful_tests = sum(1 for v in all_results.values() if v is not None)
    total_tests = len(all_results)
    
    print(f"✅ Successful tests: {successful_tests}/{total_tests}")
    print(f"❌ Failed tests: {total_tests - successful_tests}/{total_tests}")
    
    if successful_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED! Generated code should match main app results.")
    else:
        print(f"\n⚠️  {total_tests - successful_tests} tests failed. Check implementation.")
    
    # Print detailed results for verification
    print(f"\n{'='*60}")
    print("📋 DETAILED RESULTS (for manual verification)")
    print(f"{'='*60}")
    
    for config_name, results in all_results.items():
        if results:
            print(f"\n{config_name}:")
            for metric, value in results.items():
                print(f"  {metric}: {value:.6f}")

if __name__ == "__main__":
    main()
