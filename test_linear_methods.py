# Test Linear Regression Methods
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
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
dependent_var = 'promotion'

X = df[independent_vars].copy()
y = df[dependent_var].copy()

# Test all linear methods
methods = [
    ('OLS', LinearRegression()),
    ('Lasso', Lasso(alpha=1.0, random_state=42)),
    ('Ridge', Ridge(alpha=1.0, random_state=42)),
    ('Elastic Net', ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42))
]

for method_name, model in methods:
    print(f"\n{'='*50}")
    print(f"Testing {method_name}")
    print(f"{'='*50}")
    
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
    
    print(f"🎯 {method_name} Results:")
    print(f"📊 Training R²: {train_r2:.6f}")
    print(f"📊 Test R²: {test_r2:.6f}")
    print(f"📊 Training RMSE: {train_rmse:.6f}")
    print(f"📊 Test RMSE: {test_rmse:.6f}")

# Test classification
print(f"\n{'='*50}")
print(f"Testing Logistic Regression (Classification)")
print(f"{'='*50}")

# Use binary target
y_binary = (df['promotion'] > 1).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42, stratify=y_binary)

# Train model
log_model = LogisticRegression(random_state=42, max_iter=1000)
log_model.fit(X_train, y_train)

# Predictions
y_train_pred = log_model.predict(X_train)
y_test_pred = log_model.predict(X_test)

# Metrics
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print(f"🎯 Logistic Regression Results:")
print(f"📊 Training Accuracy: {train_acc:.6f}")
print(f"📊 Test Accuracy: {test_acc:.6f}")

print(f"\n✅ All linear methods tested successfully!")
