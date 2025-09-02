"""
Test Dataset Generator for Supervised Learning Tool
==================================================

Creates a comprehensive dataset for testing the econometric analysis app with:
- 500 samples, 5 variables
- Mix of continuous, categorical, and dummy variables
- Both regression and classification targets
- Missing values for testing imputation
- Realistic economic/business context

Author: Created for testing the Supervised Learning Tool
Date: September 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_test_dataset(n_samples=500):
    """
    Generate a comprehensive test dataset for the app.
    
    Parameters:
    - n_samples: Number of samples to generate (default: 500)
    
    Returns:
    - DataFrame with mixed variable types
    """
    
    print(f"🎯 Generating test dataset with {n_samples} samples...")
    
    # 1. Continuous variables
    # Age (20-65, normally distributed around 40)
    age = np.random.normal(40, 12, n_samples)
    age = np.clip(age, 20, 65)  # Clip to reasonable range
    
    # Experience (related to age but with some noise)
    experience = np.maximum(0, age - 22 + np.random.normal(0, 3, n_samples))
    experience = np.clip(experience, 0, 40)
    
    # Income (log-normal distribution, realistic income range)
    log_income = np.random.normal(10.5, 0.5, n_samples)  # Log of income
    income = np.exp(log_income)  # Convert back to income
    
    # 2. Categorical variable (converted to dummy)
    # Education level: High School, Bachelor, Master, PhD
    education_levels = ['High School', 'Bachelor', 'Master', 'PhD']
    education_probs = [0.3, 0.4, 0.25, 0.05]  # Realistic distribution
    education = np.random.choice(education_levels, n_samples, p=education_probs)
    
    # Convert education to dummy variables
    education_dummies = pd.get_dummies(education, prefix='education')
    
    # 3. Binary dummy variable
    # Urban vs Rural (1 = Urban, 0 = Rural)
    urban_prob = 0.7  # 70% urban
    is_urban = np.random.binomial(1, urban_prob, n_samples)
    
    # 4. Another continuous variable with some correlation
    # Hours worked per week (influenced by income and education)
    base_hours = 40
    education_effect = {
        'High School': -2,
        'Bachelor': 0, 
        'Master': 3,
        'PhD': 5
    }
    
    hours_worked = []
    for i in range(n_samples):
        edu_effect = education_effect[education[i]]
        income_effect = (np.log(income[i]) - 10.5) * 2  # Income effect
        hours = base_hours + edu_effect + income_effect + np.random.normal(0, 5)
        hours = np.clip(hours, 20, 80)  # Reasonable range
        hours_worked.append(hours)
    
    hours_worked = np.array(hours_worked)
    
    # Create the main dataset
    data = pd.DataFrame({
        'age': age,
        'experience': experience,
        'income': income,
        'hours_worked': hours_worked,
        'is_urban': is_urban
    })
    
    # Add education dummies
    data = pd.concat([data, education_dummies], axis=1)
    
    # 5. Create target variables for testing different models
    
    # REGRESSION TARGET: Salary (influenced by all variables)
    salary_base = (
        1000 +  # Base salary
        experience * 800 +  # Experience effect
        np.log(income) * 2000 +  # Income effect (proxy for wealth/background)
        hours_worked * 200 +  # Hours effect
        is_urban * 5000 +  # Urban premium
        (education == 'Bachelor').astype(int) * 8000 +
        (education == 'Master').astype(int) * 15000 +
        (education == 'PhD').astype(int) * 25000
    )
    
    # Add realistic noise
    salary = salary_base + np.random.normal(0, 8000, n_samples)
    salary = np.maximum(salary, 20000)  # Minimum salary
    data['salary'] = salary
    
    # CLASSIFICATION TARGET: High earner (binary)
    # Based on salary being above median
    salary_median = np.median(salary)
    data['high_earner'] = (salary > salary_median).astype(int)
    
    # CLASSIFICATION TARGET: Promotion (multi-class)
    # 0 = No promotion, 1 = Small promotion, 2 = Big promotion
    promotion_prob = (
        0.1 +  # Base probability
        experience * 0.01 +  # Experience effect
        (education == 'Master').astype(int) * 0.2 +
        (education == 'PhD').astype(int) * 0.3 +
        (hours_worked - 40) * 0.005 +  # Overtime effect
        is_urban * 0.1
    )
    
    promotion_prob = np.clip(promotion_prob, 0, 0.8)
    
    # Generate promotion categories
    promotion = []
    for prob in promotion_prob:
        rand_val = np.random.random()
        if rand_val < prob * 0.3:  # 30% of promotions are big
            promotion.append(2)  # Big promotion
        elif rand_val < prob:
            promotion.append(1)  # Small promotion  
        else:
            promotion.append(0)  # No promotion
    
    data['promotion'] = promotion
    
    # 6. Add some missing values for testing imputation
    print("📝 Adding missing values for imputation testing...")
    
    # Add missing values strategically
    missing_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
    data.loc[missing_indices[:len(missing_indices)//2], 'income'] = np.nan
    
    missing_indices_2 = np.random.choice(n_samples, size=int(n_samples * 0.03), replace=False)
    data.loc[missing_indices_2, 'hours_worked'] = np.nan
    
    # 7. Add a date variable for testing date filtering
    start_date = datetime(2020, 1, 1)
    dates = []
    for i in range(n_samples):
        random_days = np.random.randint(0, 1461)  # 4 years
        dates.append(start_date + timedelta(days=random_days))
    
    data['survey_date'] = dates
    
    # Round numeric variables for readability (handle missing values)
    data['age'] = data['age'].round(0).astype(int)
    data['experience'] = data['experience'].round(1)
    # Handle income with missing values
    data['income'] = data['income'].round(0)  # Don't convert to int yet due to NaN
    data['hours_worked'] = data['hours_worked'].round(1)
    data['salary'] = data['salary'].round(0).astype(int)
    
    return data

def save_test_datasets():
    """Generate and save test datasets in multiple formats"""
    
    print("🚀 Creating comprehensive test dataset for Supervised Learning Tool")
    print("=" * 60)
    
    # Generate the main dataset
    df = generate_test_dataset(500)
    
    # Display summary
    print("\n📊 Dataset Summary:")
    print(f"Shape: {df.shape}")
    print(f"Variables: {list(df.columns)}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    
    # Show data types
    print("\n📋 Variable Types:")
    print(df.dtypes)
    
    # Show first few rows
    print("\n👀 First 5 rows:")
    print(df.head())
    
    # Show summary statistics
    print("\n📈 Summary Statistics:")
    print(df.describe())
    
    # Save in multiple formats
    print("\n💾 Saving datasets...")
    
    # CSV format (most common)
    df.to_csv('test_dataset.csv', index=False)
    print("✅ Saved: test_dataset.csv")
    
    # Excel format
    df.to_excel('test_dataset.xlsx', index=False)
    print("✅ Saved: test_dataset.xlsx")
    
    # Create smaller subset for quick testing
    df_small = df.head(100).copy()
    df_small.to_csv('test_dataset_small.csv', index=False)
    print("✅ Saved: test_dataset_small.csv (100 samples)")
    
    # Create regression-only dataset (remove classification targets)
    df_regression = df.drop(['high_earner', 'promotion'], axis=1).copy()
    df_regression.to_csv('test_dataset_regression.csv', index=False)
    print("✅ Saved: test_dataset_regression.csv")
    
    # Create classification-only dataset (remove salary)
    df_classification = df.drop(['salary'], axis=1).copy()
    df_classification.to_csv('test_dataset_classification.csv', index=False)
    print("✅ Saved: test_dataset_classification.csv")
    
    print("\n🎯 Test Scenarios:")
    print("1. 📈 REGRESSION: Use 'salary' as dependent variable")
    print("2. 🎯 BINARY CLASSIFICATION: Use 'high_earner' as dependent variable") 
    print("3. 🎲 MULTI-CLASS: Use 'promotion' as dependent variable")
    print("4. 🔧 FEATURE TESTING: Try different combinations of independent variables")
    print("5. 🌳 TREE MODELS: Test Decision Tree and Random Forest with all targets")
    print("6. 🧪 MISSING VALUES: Test imputation methods")
    print("7. 📅 DATE FILTERING: Use 'survey_date' for sample filtering")
    
    print("\n🧪 Recommended Test Cases:")
    print("=" * 40)
    print("• OLS Regression: salary ~ age + experience + education_* + is_urban")
    print("• Logistic Regression: high_earner ~ income + hours_worked + education_*")
    print("• Decision Tree: promotion ~ all variables")
    print("• Random Forest: salary ~ all variables (test tree visualization)")
    print("• Lasso/Ridge: salary ~ all variables (test regularization)")
    print("• Missing data: Include income and hours_worked (have missing values)")
    
    return df

if __name__ == "__main__":
    # Generate and save the datasets
    dataset = save_test_datasets()
    print("\n🎉 Test dataset generation complete!")
    print("📁 Files ready for testing the Supervised Learning Tool")
