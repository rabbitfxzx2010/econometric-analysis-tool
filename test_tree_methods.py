# Test Tree-based Methods
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Load test data
df = pd.read_csv('test_dataset_classification.csv')
print(f'Original dataset shape: {df.shape}')

# Apply urban filter
df = df[(df['is_urban'] >= 1.0) & (df['is_urban'] <= 1.0)]
print(f'After filtering shape: {df.shape}')

# Handle missing values
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
mean_imputer = SimpleImputer(strategy='mean')
df[numeric_cols] = mean_imputer.fit_transform(df[numeric_cols])

# Define variables
independent_vars = ['high_earner', 'experience', 'education_High School', 'education_Master', 'age', 'hours_worked', 'education_Bachelor', 'education_PhD', 'income']

# Test regression tree methods
print(f"\n{'='*60}")
print(f"TESTING TREE METHODS - REGRESSION")
print(f"{'='*60}")

dependent_var = 'promotion'
X = df[independent_vars].copy()
y = df[dependent_var].copy()

tree_methods = [
    ('Decision Tree', DecisionTreeRegressor(max_depth=5, min_samples_split=2, min_samples_leaf=1, random_state=42)),
    ('Random Forest', RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_split=2, min_samples_leaf=1, random_state=42))
]

for method_name, model in tree_methods:
    print(f"\n{'='*40}")
    print(f"Testing {method_name} (Regression)")
    print(f"{'='*40}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    print(f"🎯 {method_name} Results:")
    print(f"📊 Training R²: {train_r2:.6f}")
    print(f"📊 Test R²: {test_r2:.6f}")
    print(f"📊 Training RMSE: {train_rmse:.6f}")
    print(f"📊 Test RMSE: {test_rmse:.6f}")
    print(f"📊 Training MAE: {train_mae:.6f}")
    print(f"📊 Test MAE: {test_mae:.6f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n🔥 Top 3 Features:")
    for idx, row in feature_importance.head(3).iterrows():
        print(f"📈 {row['feature']:<20}: {row['importance']:.6f}")

# Test classification tree methods
print(f"\n{'='*60}")
print(f"TESTING TREE METHODS - CLASSIFICATION")
print(f"{'='*60}")

# Use binary target
y_binary = (df['promotion'] > 1).astype(int)

tree_classifiers = [
    ('Decision Tree', DecisionTreeClassifier(max_depth=5, min_samples_split=2, min_samples_leaf=1, random_state=42)),
    ('Random Forest', RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=2, min_samples_leaf=1, random_state=42))
]

for method_name, model in tree_classifiers:
    print(f"\n{'='*40}")
    print(f"Testing {method_name} (Classification)")
    print(f"{'='*40}")
    
    # Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42, stratify=y_binary)
    
    # Train model
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    print(f"🎯 {method_name} Results:")
    print(f"📊 Training Accuracy: {train_acc:.6f}")
    print(f"📊 Test Accuracy: {test_acc:.6f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n🔥 Top 3 Features:")
    for idx, row in feature_importance.head(3).iterrows():
        print(f"📈 {row['feature']:<20}: {row['importance']:.6f}")

print(f"\n✅ All tree methods tested successfully!")

# Verify Decision Tree gives expected results
print(f"\n{'='*60}")
print(f"VERIFICATION: Decision Tree Regression should match notebook")
print(f"{'='*60}")

# This should match the notebook results exactly
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeRegressor(max_depth=5, min_samples_split=2, min_samples_leaf=1, random_state=42)
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"Expected Training R²: 0.413740")
print(f"Actual Training R²:   {train_r2:.6f}")
print(f"Expected Test R²:     -0.330224")
print(f"Actual Test R²:       {test_r2:.6f}")

if abs(train_r2 - 0.413740) < 0.000001 and abs(test_r2 - (-0.330224)) < 0.000001:
    print("✅ PERFECT MATCH with notebook results!")
else:
    print("⚠️  Results don't match - need to investigate")
