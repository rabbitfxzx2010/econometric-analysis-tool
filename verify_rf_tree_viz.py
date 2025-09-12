#!/usr/bin/env python3
"""
Simple test to verify Random Forest tree visualization code generation
"""

def test_tree_visualization_code():
    """Test that tree visualization code is generated for Random Forest"""
    
    # Read the econometric_app.py file to check the visualization function
    with open('econometric_app.py', 'r') as f:
        content = f.read()
    
    # Check if the Random Forest tree visualization code is present
    checks = [
        ("Random Forest tree viz section", "elif estimation_method == \"Random Forest\":"),
        ("Tree plotting code", "plot_tree(model.estimators_[0],"),
        ("Random Forest tree title", "Random Forest - Sample Tree"),
        ("Forest properties", "RANDOM FOREST PROPERTIES"),
        ("Number of trees info", "Number of Trees: {model.n_estimators}"),
    ]
    
    print("🔍 Checking Random Forest tree visualization code...")
    print("=" * 60)
    
    all_passed = True
    for description, search_string in checks:
        if search_string in content:
            print(f"✅ {description}: FOUND")
        else:
            print(f"❌ {description}: NOT FOUND")
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("🎉 All Random Forest tree visualization code is present!")
    else:
        print("⚠️ Some Random Forest tree visualization code is missing!")
    
    return all_passed

if __name__ == "__main__":
    test_tree_visualization_code()
