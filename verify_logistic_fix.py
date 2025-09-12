#!/usr/bin/env python3
"""
Simple test to verify Logistic Regression coefficient analysis code generation
"""

def test_logistic_regression_code():
    """Test that coefficient analysis code is generated for Logistic Regression"""
    
    # Read the econometric_app.py file to check the feature analysis function
    with open('econometric_app.py', 'r') as f:
        content = f.read()
    
    # Check if the Logistic Regression coefficient analysis code is present
    checks = [
        ("Logistic Regression in linear models", "\"Logistic Regression\"]:"),
        ("Coefficient handling code", "coef_values = model.coef_"),
        ("Binary classification handling", "if len(model.coef_.shape) == 1:"),
        ("Multiclass handling", "coef_values = model.coef_[0]"),
        ("Coefficient DataFrame creation", "coefficients = pd.DataFrame"),
        ("Top coefficients display", "TOP COEFFICIENTS"),
    ]
    
    print("🔍 Checking Logistic Regression coefficient analysis code...")
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
        print("🎉 All Logistic Regression coefficient analysis code is present!")
    else:
        print("⚠️ Some Logistic Regression coefficient analysis code is missing!")
    
    return all_passed

if __name__ == "__main__":
    test_logistic_regression_code()
