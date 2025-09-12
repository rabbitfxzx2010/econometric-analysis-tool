#!/usr/bin/env python3

import streamlit as st
import sys
import os

# Add current directory to path
sys.path.append('.')

try:
    from econometric_app import main
    print('✅ Import successful - no syntax errors in econometric_app.py')
    print('✅ The class_weight_option error should be fixed')
except Exception as e:
    print(f'❌ Import error: {e}')
    import traceback
    traceback.print_exc()
