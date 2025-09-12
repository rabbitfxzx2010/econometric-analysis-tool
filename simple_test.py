#!/usr/bin/env python3
"""
Simple test to verify code generation functionality
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Create test data
np.random.seed(42)
n_samples = 1000

# Generate synthetic data
data = {
    'feature1': np.random.randn(n_samples),
    'feature2': np.random.randn(n_samples),
    'feature3': np.random.randn(n_samples),
    'target': np.random.randn(n_samples)
}

df = pd.DataFrame(data)
df.to_csv('synthetic_test_data.csv', index=False)

print("🧪 TESTING CODE GENERATION WITH SYNTHETIC DATA")
print("=" * 60)

try:
    # Import the function
    from econometric_app import generate_python_code
    print("✅ Successfully imported generate_python_code function")
    
    # Set up test parameters
    X = df[['feature1', 'feature2', 'feature3']]
    y = df['target']
    
    # Create and train a simple model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeRegressor(max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    
    print("✅ Created and trained test model")
    
    # Test parameters
    test_params = {
        'estimation_method': 'Decision Tree',
        'model_type': 'regression',
        'independent_vars': ['feature1', 'feature2', 'feature3'],
        'dependent_var': 'target',
        'include_constant': True,
        'alpha': 1.0,
        'l1_ratio': 0.5,
        'use_scaling': False,
        'use_nested_cv': False,
        'class_weight': None,
        'filename': 'synthetic_test_data.csv',
        'missing_data_method': 'Mean Imputation',
        'filter_conditions': None,
        'standardize_data': False,
        'cv_folds': 5,
        'max_depth': 3,
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
    
    # Generate code
    print("🔧 Generating Python code...")
    generated_code = generate_python_code(model=model, **test_params)
    
    print(f"✅ Code generated successfully! ({len(generated_code)} characters)")
    
    # Save generated code
    with open('test_generated_code.py', 'w') as f:
        f.write(generated_code)
    
    print("💾 Saved generated code to 'test_generated_code.py'")
    
    # Basic validation - check for key components
    required_components = [
        'import pandas as pd',
        'import numpy as np',
        'DecisionTreeRegressor',
        'train_test_split',
        'r2_score',
        'random_state=42',
        'max_depth=3'
    ]
    
    found_components = []
    missing_components = []
    
    for component in required_components:
        if component in generated_code:
            found_components.append(component)
        else:
            missing_components.append(component)
    
    print(f"\n📊 COMPONENT VALIDATION:")
    print(f"✅ Found: {len(found_components)}/{len(required_components)} required components")
    
    for component in found_components:
        print(f"  ✓ {component}")
    
    if missing_components:
        print(f"\n❌ Missing components:")
        for component in missing_components:
            print(f"  ✗ {component}")
    
    # Try to execute the generated code
    print(f"\n🔧 Testing generated code execution...")
    
    try:
        exec_namespace = {}
        exec(generated_code, exec_namespace)
        print("✅ Generated code executed successfully!")
        
        # Check for results
        if 'train_r2' in exec_namespace and 'test_r2' in exec_namespace:
            print(f"📊 Training R²: {exec_namespace['train_r2']:.6f}")
            print(f"📊 Test R²: {exec_namespace['test_r2']:.6f}")
        
    except Exception as e:
        print(f"❌ Error executing generated code: {str(e)}")
        # Print first few lines for debugging
        lines = generated_code.split('\n')
        print("\nFirst 10 lines of generated code:")
        for i, line in enumerate(lines[:10]):
            print(f"{i+1:2d}: {line}")
        
        if len(lines) > 10:
            print("...")
    
    print(f"\n{'='*60}")
    print("🎯 TEST SUMMARY")
    print(f"{'='*60}")
    
    if len(found_components) == len(required_components):
        print("✅ All required components found in generated code")
    else:
        print(f"⚠️  Missing {len(missing_components)} components")
    
    print("✅ Code generation function is working!")
    
except ImportError as e:
    print(f"❌ Cannot import generate_python_code: {str(e)}")
    print("Make sure econometric_app.py is in the current directory")
    
except Exception as e:
    print(f"❌ Unexpected error: {str(e)}")
    import traceback
    traceback.print_exc()
