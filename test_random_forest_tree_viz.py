#!/usr/bin/env python3
"""
Test script to generate a Random Forest notebook with tree visualization
"""

import pandas as pd
import numpy as np
from econometric_app import generate_jupyter_notebook

def test_random_forest_notebook_generation():
    """Test generating a Random Forest notebook with tree visualization"""
    
    # Load test data
    df = pd.read_csv('test_dataset_classification.csv')
    
    # Define variables
    independent_vars = ['age', 'experience', 'education_High School', 'hours_worked', 'education_Bachelor', 'education_Master', 'income', 'education_PhD', 'is_urban', 'high_earner']
    dependent_var = 'promotion'
    
    # Parameters for Random Forest
    estimation_method = "Random Forest"
    model_type = "regression"  
    
    # Generate notebook
    notebook_content = generate_jupyter_notebook(
        estimation_method=estimation_method,
        model_type=model_type,
        independent_vars=independent_vars,
        dependent_var=dependent_var,
        test_size=0.2,
        missing_data_method="listwise_deletion",
        max_depth=5,
        n_estimators=100,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight_option="balanced"
    )
    
    # Save notebook
    output_file = "test_random_forest_with_tree_viz.ipynb"
    with open(output_file, 'w') as f:
        f.write(notebook_content)
    
    print(f"✅ Random Forest notebook generated: {output_file}")
    
    # Check if tree visualization is included
    if "plot_tree(model.estimators_[0]" in notebook_content:
        print("✅ Tree visualization code found in notebook!")
    else:
        print("❌ Tree visualization code NOT found in notebook!")
    
    if "Random Forest - Sample Tree" in notebook_content:
        print("✅ Random Forest tree title found in notebook!")
    else:
        print("❌ Random Forest tree title NOT found in notebook!")
        
    if "RANDOM FOREST PROPERTIES" in notebook_content:
        print("✅ Random Forest properties section found in notebook!")
    else:
        print("❌ Random Forest properties section NOT found in notebook!")

if __name__ == "__main__":
    test_random_forest_notebook_generation()
