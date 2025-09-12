#!/usr/bin/env python3
"""
Test script to generate a Logistic Regression notebook with proper coefficient analysis
"""

import pandas as pd
import numpy as np
from econometric_app import generate_jupyter_notebook

def test_logistic_regression_notebook_generation():
    """Test generating a Logistic Regression notebook with coefficient analysis"""
    
    # Load test data
    df = pd.read_csv('test_dataset_classification.csv')
    
    # Define variables
    independent_vars = ['age', 'experience', 'education_High School', 'hours_worked', 'education_Bachelor', 'education_Master', 'income', 'education_PhD', 'high_earner']
    dependent_var = 'is_urban'
    
    # Parameters for Logistic Regression
    estimation_method = "Logistic Regression"
    model_type = "classification"  
    
    # Generate notebook
    notebook_content = generate_jupyter_notebook(
        estimation_method=estimation_method,
        model_type=model_type,
        independent_vars=independent_vars,
        dependent_var=dependent_var,
        test_size=0.2,
        missing_data_method="listwise_deletion",
        class_weight_option="None"
    )
    
    # Save notebook
    output_file = "test_logistic_regression_fixed.ipynb"
    with open(output_file, 'w') as f:
        f.write(notebook_content)
    
    print(f"✅ Logistic Regression notebook generated: {output_file}")
    
    # Check if coefficient analysis is included
    if "coefficients = pd.DataFrame" in notebook_content:
        print("✅ Coefficient analysis code found in notebook!")
    else:
        print("❌ Coefficient analysis code NOT found in notebook!")
    
    if "Feature analysis not implemented" in notebook_content:
        print("❌ Still showing 'not implemented' message!")
    else:
        print("✅ 'Not implemented' message removed!")
        
    if "TOP COEFFICIENTS" in notebook_content:
        print("✅ Coefficient display section found in notebook!")
    else:
        print("❌ Coefficient display section NOT found in notebook!")
        
    if "coef_values = model.coef_" in notebook_content:
        print("✅ Logistic regression coefficient handling found!")
    else:
        print("❌ Logistic regression coefficient handling NOT found!")

if __name__ == "__main__":
    test_logistic_regression_notebook_generation()
