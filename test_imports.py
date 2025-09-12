#!/usr/bin/env python3

print("Testing imports...")

try:
    import streamlit as st
    print("✓ Streamlit imported successfully")
except ImportError as e:
    print(f"✗ Streamlit import failed: {e}")

try:
    import pandas as pd
    print("✓ Pandas imported successfully")
except ImportError as e:
    print(f"✗ Pandas import failed: {e}")

try:
    import numpy as np
    print("✓ NumPy imported successfully")
except ImportError as e:
    print(f"✗ NumPy import failed: {e}")

try:
    import matplotlib.pyplot as plt
    print("✓ Matplotlib imported successfully")
except ImportError as e:
    print(f"✗ Matplotlib import failed: {e}")

try:
    import seaborn as sns
    print("✓ Seaborn imported successfully")
except ImportError as e:
    print(f"✗ Seaborn import failed: {e}")

try:
    import sklearn
    print("✓ Scikit-learn imported successfully")
except ImportError as e:
    print(f"✗ Scikit-learn import failed: {e}")

try:
    import statsmodels
    print("✓ Statsmodels imported successfully")
except ImportError as e:
    print(f"✗ Statsmodels import failed: {e}")

print("Import test completed.")
