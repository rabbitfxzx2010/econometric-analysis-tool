#!/usr/bin/env python3
"""
Quick test for econometric_app.py
Tests imports, creates toy data, and verifies key functions work.
"""
import sys
import pandas as pd
import numpy as np
from sklearn.datasets import make_regression, make_classification

# Test 1: Import the module
print("🧪 Testing econometric_app import...")
try:
    import econometric_app
    print("✅ Import successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Create toy data
print("\n🧪 Creating toy datasets...")
try:
    # Regression data
    X_reg, y_reg = make_regression(n_samples=100, n_features=4, noise=0.1, random_state=42)
    reg_df = pd.DataFrame(X_reg, columns=['feature1', 'feature2', 'feature3', 'feature4'])
    reg_df['target'] = y_reg
    
    # Classification data
    X_cls, y_cls = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)
    cls_df = pd.DataFrame(X_cls, columns=['feature1', 'feature2', 'feature3', 'feature4'])
    cls_df['target'] = y_cls
    
    print(f"✅ Created regression data: {reg_df.shape}")
    print(f"✅ Created classification data: {cls_df.shape}")
except Exception as e:
    print(f"❌ Data creation failed: {e}")
    sys.exit(1)

# Test 3: Test key functions individually
print("\n🧪 Testing key functions...")

try:
    # Test fit_model with OLS
    from econometric_app import fit_model, calculate_regression_stats
    X_test = reg_df[['feature1', 'feature2', 'feature3', 'feature4']]
    y_test = reg_df['target']
    
    model = fit_model(X_test, y_test, 'OLS')
    stats = calculate_regression_stats(X_test, y_test, model, 'OLS')
    print(f"✅ OLS regression: R² = {stats['r_squared']:.3f}")
    
    # Test decision tree
    tree_model = fit_model(X_test, y_test, 'Decision Tree', 
                          model_type='regression', max_depth=3, enable_pruning=False)
    print("✅ Decision tree regression successful")
    
    # Test tree visualization
    from econometric_app import create_interactive_tree_plot
    feature_names = ['feature1', 'feature2', 'feature3', 'feature4']
    fig = create_interactive_tree_plot(tree_model, feature_names, max_depth=3)
    print("✅ Tree visualization successful")
    print(f"   Figure type: {type(fig)}")
    
except Exception as e:
    print(f"❌ Function test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Test with manual alpha
print("\n🧪 Testing manual alpha tree...")
try:
    tree_manual = fit_model(X_test, y_test, 'Decision Tree', 
                           model_type='regression', max_depth=5, 
                           enable_pruning=True, pruning_method="Manual Alpha", manual_alpha=0.01)
    fig_manual = create_interactive_tree_plot(tree_manual, feature_names, max_depth=5)
    print("✅ Manual alpha tree successful")
    print(f"   Tree nodes: {tree_manual.tree_.node_count}")
    
except Exception as e:
    print(f"❌ Manual alpha test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n🎉 All tests completed!")
print("\n📝 To run the full app locally:")
print("   streamlit run econometric_app.py")
