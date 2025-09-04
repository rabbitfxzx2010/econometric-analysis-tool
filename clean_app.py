#!/usr/bin/env python3
"""
Script to clean up the corrupted econometric_app.py file
"""

# Read the file
with open('econometric_app.py', 'r') as f:
    lines = f.readlines()

# Find corrupted sections and clean them
clean_lines = []
skip_corrupted = False
in_function = False

for i, line in enumerate(lines):
    line_num = i + 1
    
    # Skip corrupted sections between line 958 and 1822
    if line_num == 958 and 'return fig' in line:
        clean_lines.append(line)
        # Skip to line 1823 where clean content starts
        skip_corrupted = True
        continue
    elif line_num == 1823 and 'def calculate_regression_stats' in line:
        skip_corrupted = False
        clean_lines.append(line)
        continue
    elif skip_corrupted:
        continue
    else:
        clean_lines.append(line)

# Write cleaned file
with open('econometric_app.py', 'w') as f:
    f.writelines(clean_lines)

print(f"Cleaned up file! Removed corrupted section.")
