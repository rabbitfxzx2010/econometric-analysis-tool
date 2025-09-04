#!/usr/bin/env python3
"""
Script to fix the corrupted tree function in econometric_app.py
"""

import re

# Read the clean tree function from tree_fix.py
with open('tree_fix.py', 'r') as f:
    clean_tree_function = f.read()

# Read the corrupted main file
with open('econometric_app_backup.py', 'r') as f:
    content = f.read()

# Find the first occurrence of the tree function and replace it
def replace_tree_function(content, clean_function):
    # Pattern to match the function definition through its end
    pattern = r'def create_interactive_tree_plot\(.*?\):(.*?)(?=\ndef [a-zA-Z_]|\nclass [a-zA-Z_]|\n# |$)'
    
    # Find the function
    match = re.search(pattern, content, re.DOTALL)
    if match:
        # Replace with clean function
        new_content = content[:match.start()] + clean_function + content[match.end():]
        return new_content
    else:
        print("Tree function not found!")
        return content

# Replace the function
fixed_content = replace_tree_function(content, clean_tree_function)

# Write the fixed content
with open('econometric_app.py', 'w') as f:
    f.write(fixed_content)

print("Tree function replaced successfully!")
