def create_interactive_tree_plot(model, feature_names, class_names=None, max_depth=None):
    """
    Create an interactive decision tree visualization with probability-based heatmap coloring.
    Fixed version with visible text and no bottom cutoff.
    """
    tree = model.tree_
    feature = tree.feature
    threshold = tree.threshold
    children_left = tree.children_left
    children_right = tree.children_right
    value = tree.value
    impurity = tree.impurity
    n_node_samples = tree.n_node_samples
    
    # Get total samples for proportion calculation
    total_samples = int(n_node_samples[0])
    
    # Calculate positions with much wider spacing to prevent overlap
    def get_tree_positions(node=0, x=0, y=0, level=0, positions=None, spacing_factor=25.0):
        if positions is None:
            positions = {}
            
        positions[node] = (x, y)
        
        if max_depth is not None and level >= max_depth:
            return positions
        
        if int(children_left[node]) != int(children_right[node]):  # Not a leaf
            # Use much wider spacing with very slow reduction to prevent overlap
            spacing = spacing_factor / (level * 0.2 + 1)  # Much slower spacing reduction
            
            if children_left[node] >= 0:
                left_child = int(children_left[node])
                get_tree_positions(left_child, x - spacing, y - 3.5, level + 1, positions, spacing_factor)
            
            if children_right[node] >= 0:
                right_child = int(children_right[node])
                get_tree_positions(right_child, x + spacing, y - 3.5, level + 1, positions, spacing_factor)
        
        return positions
    
    positions = get_tree_positions()
    
    # Calculate probability ranges for color scaling
    all_probabilities = []
    for n in range(tree.node_count):
        if max_depth is not None and positions[n][1] < -max_depth * 3.5:
            continue
        
        if class_names is not None:  # Classification
            class_probs = value[n][0] / np.sum(value[n][0])
            max_prob = float(np.max(class_probs))
            all_probabilities.append(max_prob)
        else:  # Regression - normalize prediction values
            predicted_value = float(value[n][0][0])
            all_probabilities.append(predicted_value)
    
    if all_probabilities:
        min_prob, max_prob = min(all_probabilities), max(all_probabilities)
    else:
        min_prob, max_prob = 0.0, 1.0
    
    # Create the plot
    fig = go.Figure()
    
    # Add edges with labels
    for node in range(tree.node_count):
        if max_depth is not None and positions[node][1] < -max_depth * 3.5:
            continue
            
        if children_left[node] >= 0:  # Has left child
            left_child = int(children_left[node])
            if left_child in positions:
                x0, y0 = positions[node]
                x1, y1 = positions[left_child]
                
                # Add edge line
                fig.add_trace(go.Scatter(
                    x=[x0, x1], y=[y0, y1],
                    mode='lines',
                    line=dict(color='black', width=2),
                    hoverinfo='none',
                    showlegend=False
                ))
                
                # Add "yes" label on left branch
                mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2
                fig.add_annotation(
                    x=mid_x - 0.2, y=mid_y + 0.2,
                    text="<b>yes</b>",
                    showarrow=False,
                    font=dict(size=12, color='black', family='Arial Bold'),
                    bgcolor='rgba(255,255,255,0.9)',
                    bordercolor='black',
                    borderwidth=1,
                    borderpad=2
                )
        
        if children_right[node] >= 0:  # Has right child
            right_child = int(children_right[node])
            if right_child in positions:
                x0, y0 = positions[node]
                x1, y1 = positions[right_child]
                
                # Add edge line
                fig.add_trace(go.Scatter(
                    x=[x0, x1], y=[y0, y1],
                    mode='lines',
                    line=dict(color='black', width=2),
                    hoverinfo='none',
                    showlegend=False
                ))
                
                # Add "no" label on right branch
                mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2
                fig.add_annotation(
                    x=mid_x + 0.2, y=mid_y + 0.2,
                    text="<b>no</b>",
                    showarrow=False,
                    font=dict(size=12, color='black', family='Arial Bold'),
                    bgcolor='rgba(255,255,255,0.9)',
                    bordercolor='black',
                    borderwidth=1,
                    borderpad=2
                )
    
    # Create nodes with probability-based coloring
    for node in range(tree.node_count):
        if max_depth is not None and positions[node][1] < -max_depth * 3.5:
            continue
            
        x, y = positions[node]
        
        # Node information
        samples = int(n_node_samples[node])
        proportion = samples / total_samples
        
        is_leaf = int(children_left[node]) == int(children_right[node])
        
        if is_leaf:  # Leaf node
            if class_names is not None:  # Classification
                predicted_class = int(np.argmax(value[node][0]))
                class_probs = value[node][0] / np.sum(value[node][0])
                predicted_probability = float(class_probs[predicted_class])
                
                # Calculate color intensity based on probability (0.0 to 1.0)
                if max_prob > min_prob:
                    color_intensity = (predicted_probability - min_prob) / (max_prob - min_prob)
                else:
                    color_intensity = predicted_probability
                
                # Create heatmap-style colors: light blue (low) to deep red (high)
                if color_intensity < 0.2:
                    node_color = f'rgba(240, 248, 255, 0.95)'  # Very light blue
                    text_color = 'black'
                elif color_intensity < 0.4:
                    node_color = f'rgba(173, 216, 230, 0.95)'  # Light blue
                    text_color = 'black'
                elif color_intensity < 0.6:
                    node_color = f'rgba(255, 140, 105, 0.95)'  # Coral/salmon
                    text_color = 'black'
                elif color_intensity < 0.8:
                    node_color = f'rgba(255, 69, 58, 0.95)'   # Red-orange
                    text_color = 'white'
                else:
                    node_color = f'rgba(220, 20, 20, 0.95)'   # Deep red
                    text_color = 'white'
                
                # Node text: GUARANTEED VISIBLE probability and percentage
                node_text = f"<b>{predicted_probability:.3f}</b><br><b>{proportion*100:.0f}%</b>"
                
                hover_text = (f"<b>LEAF NODE</b><br>"
                            f"Predicted Class: <b>{class_names[predicted_class]}</b><br>"
                            f"Probability: <b>{predicted_probability:.3f}</b><br>"
                            f"Samples: {samples:,}<br>"
                            f"Percentage: {proportion*100:.1f}%<br>"
                            f"Confidence Level: {'High' if predicted_probability > 0.8 else 'Medium' if predicted_probability > 0.6 else 'Low'}")
                            
            else:  # Regression
                predicted_value = float(value[node][0][0])
                
                # Normalize for color intensity
                if max_prob > min_prob:
                    color_intensity = (predicted_value - min_prob) / (max_prob - min_prob)
                else:
                    color_intensity = 0.5
                
                # Heatmap colors for regression
                if color_intensity < 0.2:
                    node_color = f'rgba(240, 248, 255, 0.95)'  # Very light blue
                    text_color = 'black'
                elif color_intensity < 0.4:
                    node_color = f'rgba(173, 216, 230, 0.95)'  # Light blue
                    text_color = 'black'
                elif color_intensity < 0.6:
                    node_color = f'rgba(255, 140, 105, 0.95)'  # Coral
                    text_color = 'black'
                elif color_intensity < 0.8:
                    node_color = f'rgba(255, 69, 58, 0.95)'   # Red-orange
                    text_color = 'white'
                else:
                    node_color = f'rgba(220, 20, 20, 0.95)'   # Deep red
                    text_color = 'white'
                
                node_text = f"<b>{predicted_value:.3f}</b><br><b>{proportion*100:.0f}%</b>"
                
                hover_text = (f"<b>LEAF NODE</b><br>"
                            f"Predicted Value: <b>{predicted_value:.3f}</b><br>"
                            f"Samples: {samples:,}<br>"
                            f"Percentage: {proportion*100:.1f}%")
        
        else:  # Internal node
            feature_name = feature_names[int(feature[node])]
            threshold_val = float(threshold[node])
            
            if class_names is not None:  # Classification
                predicted_class = int(np.argmax(value[node][0]))
                class_probs = value[node][0] / np.sum(value[node][0])
                predicted_probability = float(class_probs[predicted_class])
                
                # Calculate color intensity
                if max_prob > min_prob:
                    color_intensity = (predicted_probability - min_prob) / (max_prob - min_prob)
                else:
                    color_intensity = predicted_probability
                
                # Lighter heatmap colors for internal nodes
                if color_intensity < 0.2:
                    node_color = f'rgba(248, 248, 255, 0.9)'  # Ghost white
                    text_color = 'black'
                elif color_intensity < 0.4:
                    node_color = f'rgba(230, 240, 250, 0.9)'  # Light blue
                    text_color = 'black'
                elif color_intensity < 0.6:
                    node_color = f'rgba(255, 200, 180, 0.9)'  # Light coral
                    text_color = 'black'
                elif color_intensity < 0.8:
                    node_color = f'rgba(255, 160, 140, 0.9)'  # Medium coral
                    text_color = 'black'
                else:
                    node_color = f'rgba(240, 100, 80, 0.9)'   # Deep coral
                    text_color = 'white'
                
                # Node text: probability and percentage
                node_text = f"<b>{predicted_probability:.3f}</b><br><b>{proportion*100:.0f}%</b>"
                
                # Decision rule as separate annotation below the node
                decision_text = f"<b>{feature_name} < {threshold_val:.2f}</b>"
                
                hover_text = (f"<b>DECISION NODE</b><br>"
                            f"Split Rule: <b>{feature_name} < {threshold_val:.3f}</b><br>"
                            f"Current Best Probability: <b>{predicted_probability:.3f}</b><br>"
                            f"Samples: {samples:,}<br>"
                            f"Percentage: {proportion*100:.1f}%")
                            
            else:  # Regression
                predicted_value = float(value[node][0][0])
                
                if max_prob > min_prob:
                    color_intensity = (predicted_value - min_prob) / (max_prob - min_prob)
                else:
                    color_intensity = 0.5
                
                # Lighter heatmap colors for internal regression nodes
                if color_intensity < 0.2:
                    node_color = f'rgba(248, 248, 255, 0.9)'  # Ghost white
                    text_color = 'black'
                elif color_intensity < 0.4:
                    node_color = f'rgba(230, 240, 250, 0.9)'  # Light blue
                    text_color = 'black'
                elif color_intensity < 0.6:
                    node_color = f'rgba(255, 200, 180, 0.9)'  # Light coral
                    text_color = 'black'
                elif color_intensity < 0.8:
                    node_color = f'rgba(255, 160, 140, 0.9)'  # Medium coral
                    text_color = 'black'
                else:
                    node_color = f'rgba(240, 100, 80, 0.9)'   # Deep coral
                    text_color = 'white'
                
                node_text = f"<b>{predicted_value:.3f}</b><br><b>{proportion*100:.0f}%</b>"
                
                decision_text = f"<b>{feature_name} < {threshold_val:.2f}</b>"
                
                hover_text = (f"<b>DECISION NODE</b><br>"
                            f"Split Rule: <b>{feature_name} < {threshold_val:.3f}</b><br>"
                            f"Current Value: <b>{predicted_value:.3f}</b><br>"
                            f"Samples: {samples:,}<br>"
                            f"Percentage: {proportion*100:.1f}%")
        
        # Create BIGGER rectangular nodes with better proportions
        node_width = 1.8   # Even bigger width
        node_height = 1.2  # Bigger height for better text visibility
        
        # Add rectangle with probability-based coloring
        fig.add_shape(
            type="rect",
            x0=x - node_width/2, y0=y - node_height/2,
            x1=x + node_width/2, y1=y + node_height/2,
            fillcolor=node_color,
            line=dict(color='black', width=2)
        )
        
        # Add node text with MAXIMUM visibility
        fig.add_annotation(
            x=x, y=y,
            text=node_text,
            showarrow=False,
            font=dict(size=24, color=text_color, family='Arial Black'),  # VERY large, bold font
            bgcolor='rgba(255,255,255,0.4)' if text_color == 'white' else 'rgba(0,0,0,0.2)',  # More visible background
            bordercolor=text_color,
            borderwidth=2
        )
        
        # Add decision rule below internal nodes
        if not is_leaf:
            fig.add_annotation(
                x=x, y=y - 1.0,
                text=decision_text,
                showarrow=False,
                font=dict(size=11, color='black', family='Arial Bold'),
                bgcolor='rgba(255,255,255,0.95)',
                bordercolor='black',
                borderwidth=1,
                borderpad=3
            )
        
        # Add invisible scatter point for hover with larger area
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='markers',
            marker=dict(size=50, color='rgba(0,0,0,0)'),  # Larger hover area
            hovertext=hover_text,
            hoverinfo='text',
            hoverlabel=dict(
                bgcolor='rgba(255,255,255,0.95)',
                bordercolor='black',
                font=dict(size=12, color='black', family='Arial')
            ),
            showlegend=False,
            name=f'Node_{node}'
        ))
    
    # Add color scale legend on the right
    if class_names is not None:
        # Classification probability color scale with heatmap colors
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(
                size=1,
                color=[min_prob, max_prob],
                colorscale=[
                    [0, 'rgb(240,248,255)'],      # Very light blue
                    [0.2, 'rgb(173,216,230)'],    # Light blue
                    [0.4, 'rgb(255,140,105)'],    # Coral
                    [0.6, 'rgb(255,69,58)'],      # Red-orange
                    [1, 'rgb(220,20,20)']         # Deep red
                ],
                showscale=True,
                colorbar=dict(
                    title="<b>Probability Level</b>",
                    titleside="right",
                    thickness=35,
                    len=0.8,
                    x=1.02,
                    tickvals=[min_prob, min_prob + 0.2*(max_prob-min_prob), min_prob + 0.4*(max_prob-min_prob), min_prob + 0.6*(max_prob-min_prob), min_prob + 0.8*(max_prob-min_prob), max_prob],
                    ticktext=[f'{min_prob:.2f}<br><span style="font-size:10px">Very Low</span>', 
                             f'{min_prob + 0.2*(max_prob-min_prob):.2f}<br><span style="font-size:10px">Low</span>', 
                             f'{min_prob + 0.4*(max_prob-min_prob):.2f}<br><span style="font-size:10px">Medium</span>',
                             f'{min_prob + 0.6*(max_prob-min_prob):.2f}<br><span style="font-size:10px">High</span>',
                             f'{min_prob + 0.8*(max_prob-min_prob):.2f}<br><span style="font-size:10px">Very High</span>',
                             f'{max_prob:.2f}<br><span style="font-size:10px">Max</span>'],
                    tickfont=dict(size=11, family='Arial Bold')
                )
            ),
            showlegend=False,
            name='Probability Scale'
        ))
    else:
        # Regression value color scale with matching heatmap colors
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(
                size=1,
                color=[min_prob, max_prob],
                colorscale=[
                    [0, 'rgb(240,248,255)'],      # Very light blue
                    [0.2, 'rgb(173,216,230)'],    # Light blue
                    [0.4, 'rgb(255,140,105)'],    # Coral
                    [0.6, 'rgb(255,69,58)'],      # Red-orange
                    [1, 'rgb(220,20,20)']         # Deep red
                ],
                showscale=True,
                colorbar=dict(
                    title="<b>Predicted Value</b>",
                    titleside="right",
                    thickness=35,
                    len=0.8,
                    x=1.02,
                    tickfont=dict(size=11, family='Arial Bold')
                )
            ),
            showlegend=False,
            name='Value Scale'
        ))
    
    # Update layout with MAXIMUM spacing and margins to prevent issues
    fig.update_layout(
        title=dict(
            text="Decision Tree Probability Heatmap<br><sub>Probabilities and percentages displayed - Colors show confidence levels - Hover for details</sub>",
            font=dict(size=18, color='black'),
            x=0.5,
            xanchor='center'
        ),
        showlegend=False,
        hovermode='closest',
        margin=dict(b=250, l=150, r=250, t=150),  # VERY large margins to prevent cutoff
        annotations=[
            dict(
                text="HEATMAP STYLE GUIDE:<br>" +
                     "LIGHT BLUE = Low probability - CORAL/ORANGE = Medium probability - DEEP RED = High probability<br>" +
                     "Numbers show: Top = probability, Bottom = sample percentage<br>" +
                     "Decision rules shown below internal nodes",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.5, y=-0.15,
                xanchor='center', yanchor='top',
                font=dict(color='rgb(60,60,60)', size=12),
                bgcolor='rgba(248,248,248,0.9)',
                bordercolor='gray',
                borderwidth=1,
                borderpad=8
            )
        ],
        xaxis=dict(
            showgrid=False, 
            zeroline=False, 
            showticklabels=False,
            fixedrange=False  # Allow horizontal scrolling
        ),
        yaxis=dict(
            showgrid=False, 
            zeroline=False, 
            showticklabels=False,
            fixedrange=True  # Disable vertical zooming/panning
        ),
        plot_bgcolor='rgba(250,250,250,1)',
        paper_bgcolor='white',
        # Make figure much wider and taller to prevent cutoff
        width=6000,   # Much wider for better spacing and no overlap
        height=3500,  # Much taller to prevent bottom cutoff
        dragmode='pan'  # Enable horizontal scrolling
    )
    
    return fig
