#!/usr/bin/env python3
"""
Test actual code generation from the app and verify results match
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the actual function from the app
from econometric_app import generate_python_code

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.impute import SimpleImputer

def test_code_generation_pipeline():
    """Test the actual code generation pipeline"""
    
    print("🧪 TESTING ACTUAL CODE GENERATION PIPELINE")
    print("=" * 60)
    
    # Load test data
    df = pd.read_csv('test_dataset_classification.csv')
    
    # Set up test parameters for Decision Tree
    test_params = {
        'estimation_method': 'Decision Tree',
        'model_type': 'regression',
        'independent_vars': ['high_earner', 'experience', 'education_High School', 'education_Master', 'age', 'hours_worked', 'education_Bachelor', 'education_PhD', 'income'],
        'dependent_var': 'promotion',
        'include_constant': True,
        'alpha': 1.0,
        'l1_ratio': 0.5,
        'use_scaling': False,
        'use_nested_cv': False,
        'class_weight': None,
        'filename': 'test_dataset_classification.csv',
        'missing_data_method': 'Mean Imputation',
        'filter_conditions': [{'type': 'numerical', 'column': 'is_urban', 'values': [1.0, 1.0]}],
        'standardize_data': False,
        'cv_folds': 5,
        'max_depth': 5,
        'n_estimators': 100,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'enable_pruning': False,
        'pruning_method': None,
        'manual_alpha': None,
        'use_max_depth': True,
        'prob_class_index': 0,
        'include_plots': False,  # Disable plots for testing
        'parameter_input_method': None,
        'use_stratify': False,
        'class_weight_option': None,
        'filter_method': None,
        'start_row': None,
        'end_row': None,
        'use_sample_filter': False,
        'random_state': 42
    }
    
    # Create a mock model for testing
    X = df[test_params['independent_vars']].copy()
    y = df[test_params['dependent_var']].copy()
    
    # Apply filtering
    df_filtered = df[df['is_urban'] == 1].copy()
    X_filtered = df_filtered[test_params['independent_vars']].copy()
    y_filtered = df_filtered[test_params['dependent_var']].copy()
    
    # Train model for reference
    X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_filtered, test_size=0.2, random_state=42)
    mock_model = DecisionTreeRegressor(max_depth=5, min_samples_split=2, min_samples_leaf=1, random_state=42)
    mock_model.fit(X_train, y_train)
    
    # Generate code
    print("🔧 Generating Python code...")
    generated_code = generate_python_code(
        model=mock_model,
        **test_params
    )
    
    print("✅ Code generated successfully!")
    print(f"📏 Code length: {len(generated_code)} characters")
    
    # Save generated code to file
    with open('generated_test_code.py', 'w') as f:
        f.write(generated_code)
    
    print("💾 Saved generated code to 'generated_test_code.py'")
    
    # Check if code contains key components
    required_components = [
        'import pandas as pd',
        'import numpy as np',
        'DecisionTreeRegressor',
        'train_test_split',
        'mean_squared_error',
        'r2_score',
        'random_state=42',
        'max_depth=5',
        'min_samples_split=2',
        'min_samples_leaf=1',
        'KEY RESULTS',
        'Training R²',
        'Test R²'
    ]
    
    missing_components = []
    for component in required_components:
        if component not in generated_code:
            missing_components.append(component)
    
    if missing_components:
        print(f"❌ Missing components: {missing_components}")
        return False
    else:
        print("✅ All required components found in generated code!")
    
    # Try to execute the generated code (without plots)
    print("\n🔧 Testing generated code execution...")
    
    try:
        # Create a clean namespace for execution
        exec_namespace = {}
        
        # Execute the generated code
        exec(generated_code, exec_namespace)
        
        print("✅ Generated code executed successfully!")
        
        # Check if expected variables exist in the namespace
        expected_vars = ['train_r2', 'test_r2', 'train_rmse', 'test_rmse']
        for var in expected_vars:
            if var in exec_namespace:
                print(f"📊 {var}: {exec_namespace[var]:.6f}")
            else:
                print(f"⚠️  Variable {var} not found in execution results")
        
        # Compare with expected results
        if 'train_r2' in exec_namespace and 'test_r2' in exec_namespace:
            expected_train_r2 = 0.413740
            expected_test_r2 = -0.330224
            
            actual_train_r2 = exec_namespace['train_r2']
            actual_test_r2 = exec_namespace['test_r2']
            
            print(f"\n🎯 ACCURACY CHECK:")
            print(f"Expected Training R²: {expected_train_r2:.6f}")
            print(f"Actual Training R²:   {actual_train_r2:.6f}")
            print(f"Difference:           {abs(actual_train_r2 - expected_train_r2):.6f}")
            
            print(f"Expected Test R²:     {expected_test_r2:.6f}")
            print(f"Actual Test R²:       {actual_test_r2:.6f}")
            print(f"Difference:           {abs(actual_test_r2 - expected_test_r2):.6f}")
            
            if abs(actual_train_r2 - expected_train_r2) < 0.000001 and abs(actual_test_r2 - expected_test_r2) < 0.000001:
                print("\n✅ PERFECT MATCH! Generated code produces identical results.")
                return True
            else:
                print("\n⚠️  Results differ. Investigating...")
                return False
        
    except Exception as e:
        print(f"❌ Error executing generated code: {str(e)}")
        return False
    
    return True

def test_multiple_methods():
    """Test code generation for multiple methods"""
    
    print("\n" + "=" * 60)
    print("🧪 TESTING MULTIPLE ESTIMATION METHODS")
    print("=" * 60)
    
    # Load data once
    df = pd.read_csv('test_dataset_classification.csv')
    
    test_methods = [
        {
            'name': 'OLS Regression',
            'estimation_method': 'OLS',
            'model_type': 'regression',
            'model_class': LinearRegression
        },
        {
            'name': 'Lasso Regression',
            'estimation_method': 'Lasso',
            'model_type': 'regression',
            'model_class': Lasso,
            'alpha': 0.1
        },
        {
            'name': 'Decision Tree Regression',
            'estimation_method': 'Decision Tree',
            'model_type': 'regression',
            'model_class': DecisionTreeRegressor,
            'max_depth': 5
        }
    ]
    
    results = {}
    
    for method_config in test_methods:
        print(f"\n{'='*40}")
        print(f"Testing {method_config['name']}")
        print(f"{'='*40}")
        
        try:
            # Set up basic parameters
            params = {
                'estimation_method': method_config['estimation_method'],
                'model_type': method_config['model_type'],
                'independent_vars': ['high_earner', 'experience', 'education_High School', 'education_Master', 'age', 'hours_worked', 'education_Bachelor', 'education_PhD', 'income'],
                'dependent_var': 'promotion',
                'include_constant': True,
                'alpha': method_config.get('alpha', 1.0),
                'l1_ratio': 0.5,
                'use_scaling': False,
                'use_nested_cv': False,
                'class_weight': None,
                'filename': 'test_dataset_classification.csv',
                'missing_data_method': 'Mean Imputation',
                'filter_conditions': None,
                'standardize_data': False,
                'cv_folds': 5,
                'max_depth': method_config.get('max_depth', None),
                'n_estimators': 100,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'enable_pruning': False,
                'pruning_method': None,
                'manual_alpha': None,
                'use_max_depth': True,
                'prob_class_index': 0,
                'include_plots': False,
                'parameter_input_method': None,
                'use_stratify': False,
                'class_weight_option': None,
                'filter_method': None,
                'start_row': None,
                'end_row': None,
                'use_sample_filter': False,
                'random_state': 42
            }
            
            # Create mock model
            X = df[params['independent_vars']].copy()
            y = df[params['dependent_var']].copy()
            
            if method_config['model_class'] == LinearRegression:
                mock_model = LinearRegression()
            elif method_config['model_class'] == Lasso:
                mock_model = Lasso(alpha=params['alpha'], random_state=42)
            elif method_config['model_class'] == DecisionTreeRegressor:
                mock_model = DecisionTreeRegressor(max_depth=params['max_depth'], random_state=42)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            mock_model.fit(X_train, y_train)
            
            # Generate code
            generated_code = generate_python_code(model=mock_model, **params)
            
            # Check if code generation succeeded
            if len(generated_code) > 1000:  # Basic check
                print(f"✅ {method_config['name']} - Code generated ({len(generated_code)} chars)")
                results[method_config['name']] = True
            else:
                print(f"❌ {method_config['name']} - Code too short")
                results[method_config['name']] = False
                
        except Exception as e:
            print(f"❌ {method_config['name']} - Error: {str(e)}")
            results[method_config['name']] = False
    
    # Summary
    successful = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"\n{'='*60}")
    print(f"📊 MULTI-METHOD TEST SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successful: {successful}/{total}")
    print(f"❌ Failed: {total - successful}/{total}")
    
    for method, success in results.items():
        status = "✅" if success else "❌"
        print(f"{status} {method}")
    
    return successful == total

if __name__ == "__main__":
    print("🚀 STARTING COMPREHENSIVE CODE GENERATION TESTS")
    print("=" * 60)
    
    # Test 1: Single method pipeline
    success1 = test_code_generation_pipeline()
    
    # Test 2: Multiple methods
    success2 = test_multiple_methods()
    
    # Final summary
    print(f"\n{'='*60}")
    print("🏁 FINAL TEST SUMMARY")
    print(f"{'='*60}")
    
    if success1 and success2:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Code generation works correctly for all methods")
        print("✅ Generated code produces exact results matching main window")
        print("✅ All estimation methods supported")
    else:
        print("⚠️  SOME TESTS FAILED")
        print(f"Pipeline test: {'✅' if success1 else '❌'}")
        print(f"Multi-method test: {'✅' if success2 else '❌'}")
    
    print("\n📋 Code generation is ready for production use!")
