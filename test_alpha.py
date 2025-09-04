import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_regression

# Create sample data
X, y = make_regression(n_samples=100, n_features=4, random_state=42)

print('Testing different alpha values...')

# Test different alpha values
alphas = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]

for alpha in alphas:
    try:
        model = DecisionTreeRegressor(ccp_alpha=alpha, random_state=42)
        model.fit(X, y)
        
        n_nodes = model.tree_.node_count
        n_leaves = model.get_n_leaves()
        max_depth = model.get_depth()
        
        print(f'Alpha {alpha:>4}: {n_nodes:>3} nodes, {n_leaves:>3} leaves, depth {max_depth:>2}')
        
        # Check if tree is essentially empty
        if n_nodes <= 1:
            print(f'  ⚠️  WARNING: Tree with alpha {alpha} has only {n_nodes} node(s)')
        
        # Test tree structure integrity
        if hasattr(model.tree_, 'children_left'):
            left_children = model.tree_.children_left
            right_children = model.tree_.children_right
            features = model.tree_.feature
            
            print(f'  Tree structure: {len(left_children)} nodes in arrays')
            
            # Check for any anomalies
            for i in range(len(left_children)):
                if left_children[i] == right_children[i] and left_children[i] != -1:
                    print(f'    ⚠️  Node {i}: left and right children are the same: {left_children[i]}')
                
    except Exception as e:
        print(f'Alpha {alpha:>4}: ERROR - {e}')

print('\nRecommendations for alpha values:')
print('• 0.0-0.01: No pruning to light pruning')
print('• 0.01-0.05: Moderate pruning (recommended range)')
print('• 0.05-0.1: Heavy pruning')
print('• >0.1: Very heavy pruning (may result in very small trees)')
