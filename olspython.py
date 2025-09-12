# Generated Python code for your analysis
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load your data
# df = pd.read_csv('your_data.csv')

# Define variables
X = df[['brw_2024']]
y = df['brw_2020']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LinearRegression()

# Fit model and make predictions
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'R-squared: {r2:.4f}')
print(f'RMSE: {np.sqrt(mse):.4f}')

# Model coefficients
if hasattr(model, 'coef_'):
    print('\nCoefficients:')
    for i, var in enumerate(['brw_2024']):
        coef = model.coef_[i] if model.coef_.ndim == 1 else model.coef_[0][i]
        print(f'{var}: {coef:.4f}')