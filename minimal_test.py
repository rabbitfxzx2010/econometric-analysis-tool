import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor  
from sklearn.linear_model import LinearRegression

st.title("Minimal Test - Decision Tree Coefficient Fix")

# Create sample data
np.random.seed(42)
X = np.random.randn(100, 3)
y = X[:, 0] + 2*X[:, 1] + np.random.randn(100) * 0.1

# Create DataFrame
df = pd.DataFrame(X, columns=['Feature1', 'Feature2', 'Feature3'])
df['Target'] = y

st.write("Sample Data:")
st.dataframe(df.head())

# Test models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42, n_estimators=10)
}

for name, model in models.items():
    st.subheader(f"Testing {name}")
    
    # Fit model
    model.fit(X, y)
    
    # Test coefficient access with our fix
    if hasattr(model, 'coef_'):
        st.success(f"✓ {name} has coefficients")
        try:
            coefficients = model.coef_
            st.write(f"Coefficients: {coefficients}")
        except Exception as e:
            st.error(f"Error accessing coefficients: {e}")
    else:
        st.info(f"ℹ️ {name} does not have coefficients (uses feature importance instead)")
        try:
            if hasattr(model, 'feature_importances_'):
                st.write(f"Feature importances: {model.feature_importances_}")
        except Exception as e:
            st.error(f"Error accessing feature importances: {e}")

st.success("🎉 All tests completed without DecisionTreeRegressor coefficient errors!")
