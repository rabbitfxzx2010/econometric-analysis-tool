import streamlit as st
import pandas as pd
import numpy as np

st.title("Basic Test - Package Availability Check")

st.write("Testing available packages...")

# Test pandas and numpy (which should be available from streamlit installation)
try:
    import pandas as pd
    import numpy as np
    st.success("✓ pandas and numpy are available")
    
    # Create sample data
    np.random.seed(42)
    data = np.random.randn(10, 3)
    df = pd.DataFrame(data, columns=['A', 'B', 'C'])
    st.write("Sample data:")
    st.dataframe(df)
    
except ImportError as e:
    st.error(f"Error with basic packages: {e}")

# Test sklearn
try:
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    st.success("✓ scikit-learn is available")
    
    # Test our coefficient fix logic
    X = np.random.randn(50, 2)
    y = np.random.randn(50)
    
    models = [
        ("Linear Regression", LinearRegression()),
        ("Decision Tree", DecisionTreeRegressor(random_state=42))
    ]
    
    for name, model in models:
        model.fit(X, y)
        st.write(f"**{name}:**")
        
        # This is our fixed logic
        if hasattr(model, 'coef_'):
            st.write(f"  ✓ Has coefficients: {model.coef_}")
        else:
            st.write(f"  ℹ️ No coefficients (tree-based model)")
            if hasattr(model, 'feature_importances_'):
                st.write(f"  ✓ Feature importances: {model.feature_importances_}")
    
except ImportError as e:
    st.error(f"scikit-learn not available: {e}")

# Test other packages
packages_to_test = ['matplotlib', 'seaborn', 'statsmodels']
for pkg in packages_to_test:
    try:
        __import__(pkg)
        st.success(f"✓ {pkg} is available")
    except ImportError:
        st.warning(f"⚠️ {pkg} is not available")

st.info("Package availability check complete!")
