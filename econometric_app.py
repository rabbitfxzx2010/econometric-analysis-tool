import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_squared_error, log_loss
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import StringIO
import openpyxl
from datetime import datetime, date
import pytz
import json
import requests
import os

# Usage tracking functions
def track_app_usage():
    """
    Track app usage by recording session starts, page views, and user actions.
    Creates and maintains usage statistics in a local JSON file.
    """
    usage_file = "app_usage_stats.json"
    # Use US Central Time
    central_tz = pytz.timezone('US/Central')
    current_time = datetime.now(central_tz)
    today = current_time.strftime("%Y-%m-%d")
    current_hour = current_time.hour
    
    # Initialize or load existing usage data
    if os.path.exists(usage_file):
        try:
            with open(usage_file, "r") as f:
                usage_data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            usage_data = {}
    else:
        usage_data = {}
    
    # Initialize structure if needed
    if "total_sessions" not in usage_data:
        usage_data["total_sessions"] = 0
    if "daily_stats" not in usage_data:
        usage_data["daily_stats"] = {}
    if "hourly_distribution" not in usage_data:
        usage_data["hourly_distribution"] = {str(i): 0 for i in range(24)}
    if "feature_usage" not in usage_data:
        usage_data["feature_usage"] = {
            "file_uploads": 0,
            "model_runs": 0,
            "visualizations_created": 0,
            "downloads": 0
        }
    if "first_use" not in usage_data:
        usage_data["first_use"] = current_time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Update daily stats
    if today not in usage_data["daily_stats"]:
        usage_data["daily_stats"][today] = {
            "sessions": 0,
            "unique_users": set(),
            "models_run": 0,
            "files_uploaded": 0
        }
    
    # Track session if it's a new session (use Streamlit session state)
    if "session_tracked" not in st.session_state:
        st.session_state.session_tracked = True
        usage_data["total_sessions"] += 1
        usage_data["daily_stats"][today]["sessions"] += 1
        usage_data["hourly_distribution"][str(current_hour)] += 1
        
        # Update last access
        usage_data["last_access"] = current_time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Convert sets to lists for JSON serialization
    for day_data in usage_data["daily_stats"].values():
        if isinstance(day_data["unique_users"], set):
            day_data["unique_users"] = list(day_data["unique_users"])
    
    # Save updated usage data
    try:
        with open(usage_file, "w") as f:
            json.dump(usage_data, f, indent=2)
    except Exception as e:
        st.error(f"Failed to save usage data: {e}")
    
    return usage_data

def track_feature_usage(feature_name):
    """
    Track specific feature usage (file upload, model run, etc.)
    """
    usage_file = "app_usage_stats.json"
    today = date.today().strftime("%Y-%m-%d")
    
    # Load or initialize usage data safely
    usage_data = {}
    if os.path.exists(usage_file):
        try:
            with open(usage_file, "r") as f:
                usage_data = json.load(f)
        except Exception as e:
            # Corrupt file or unreadable — start fresh but keep user informed
            st.warning(f"Could not read usage file, reinitializing usage stats: {e}")
            usage_data = {}

    # Ensure structure exists
    if "feature_usage" not in usage_data:
        usage_data["feature_usage"] = {
            "file_uploads": 0,
            "model_runs": 0,
            "visualizations_created": 0,
            "downloads": 0
        }
    if "daily_stats" not in usage_data:
        usage_data["daily_stats"] = {}

    # Update feature usage safely
    if feature_name in usage_data["feature_usage"]:
        usage_data["feature_usage"][feature_name] += 1
    else:
        # If unknown feature, create and increment
        usage_data["feature_usage"][feature_name] = usage_data["feature_usage"].get(feature_name, 0) + 1

    # Update daily feature usage counters
    if today not in usage_data["daily_stats"]:
        usage_data["daily_stats"][today] = {
            "sessions": 0,
            "unique_users": [],
            "models_run": 0,
            "files_uploaded": 0
        }

    if feature_name == "model_runs":
        usage_data["daily_stats"][today]["models_run"] = usage_data["daily_stats"][today].get("models_run", 0) + 1
    elif feature_name == "file_uploads":
        usage_data["daily_stats"][today]["files_uploaded"] = usage_data["daily_stats"][today].get("files_uploaded", 0) + 1

    # Save updated data
    try:
        with open(usage_file, "w") as f:
            json.dump(usage_data, f, indent=2)
    except Exception as e:
        st.warning(f"Failed to save usage data: {e}")

def display_usage_analytics():
    """
    Display comprehensive usage analytics for the app owner.
    """
    usage_file = "app_usage_stats.json"
    
    if not os.path.exists(usage_file):
        st.warning("No usage data available yet. Analytics will appear after the app has been used.")
        return
    
    try:
        with open(usage_file, "r") as f:
            usage_data = json.load(f)
    except:
        st.error("Unable to load usage data.")
        return
    
    st.markdown("# 📊 App Usage Analytics Dashboard")
    st.markdown("---")
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Sessions",
            value=usage_data.get("total_sessions", 0),
            help="Total number of app sessions since tracking began"
        )
    
    with col2:
        total_models = usage_data.get("feature_usage", {}).get("model_runs", 0)
        st.metric(
            label="Models Run",
            value=total_models,
            help="Total number of machine learning models executed"
        )
    
    with col3:
        total_uploads = usage_data.get("feature_usage", {}).get("file_uploads", 0)
        st.metric(
            label="File Uploads",
            value=total_uploads,
            help="Total number of datasets uploaded"
        )
    
    with col4:
        total_viz = usage_data.get("feature_usage", {}).get("visualizations_created", 0)
        st.metric(
            label="Visualizations",
            value=total_viz,
            help="Total number of plots and visualizations created"
        )
    
    # Time period info
    st.markdown("### 📅 Usage Period")
    col1, col2 = st.columns(2)
    
    with col1:
        first_use = usage_data.get("first_use", "Unknown")
        st.info(f"**First Use:** {first_use}")
    
    with col2:
        last_access = usage_data.get("last_access", "Unknown")
        st.info(f"**Last Access:** {last_access}")
    
    # Daily usage chart
    daily_stats = usage_data.get("daily_stats", {})
    if daily_stats:
        st.markdown("### 📈 Daily Usage Trends")
        
        # Prepare data for daily chart
        dates = list(daily_stats.keys())
        sessions = [daily_stats[date]["sessions"] for date in dates]
        models = [daily_stats[date]["models_run"] for date in dates]
        uploads = [daily_stats[date]["files_uploaded"] for date in dates]
        
        daily_df = pd.DataFrame({
            "Date": dates,
            "Sessions": sessions,
            "Models Run": models,
            "Files Uploaded": uploads
        })
        
        # Convert Date to datetime for better plotting
        daily_df["Date"] = pd.to_datetime(daily_df["Date"])
        daily_df = daily_df.sort_values("Date")
        
        # Create daily usage chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=daily_df["Date"],
            y=daily_df["Sessions"],
            mode='lines+markers',
            name='Sessions',
            line=dict(color='blue', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=daily_df["Date"],
            y=daily_df["Models Run"],
            mode='lines+markers',
            name='Models Run',
            line=dict(color='green', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=daily_df["Date"],
            y=daily_df["Files Uploaded"],
            mode='lines+markers',
            name='Files Uploaded',
            line=dict(color='orange', width=3)
        ))
        
        fig.update_layout(
            title="Daily Usage Activity",
            xaxis_title="Date",
            yaxis_title="Count",
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Hourly distribution
    hourly_dist = usage_data.get("hourly_distribution", {})
    if hourly_dist and sum(hourly_dist.values()) > 0:
        st.markdown("### 🕐 Hourly Usage Distribution")
        
        hours = list(range(24))
        counts = [hourly_dist.get(str(h), 0) for h in hours]
        
        fig = go.Figure(data=go.Bar(
            x=[f"{h:02d}:00" for h in hours],
            y=counts,
            marker_color='lightblue',
            text=counts,
            textposition='outside'
        ))
        
        fig.update_layout(
            title="App Usage by Hour of Day",
            xaxis_title="Hour",
            yaxis_title="Number of Sessions",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature usage breakdown
    feature_usage = usage_data.get("feature_usage", {})
    if feature_usage and sum(feature_usage.values()) > 0:
        st.markdown("### 🔧 Feature Usage Breakdown")
        
        feature_names = list(feature_usage.keys())
        feature_counts = list(feature_usage.values())
        
        fig = go.Figure(data=go.Pie(
            labels=[name.replace("_", " ").title() for name in feature_names],
            values=feature_counts,
            hole=0.4
        ))
        
        fig.update_layout(
            title="Most Used Features",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent activity table
    if daily_stats:
        st.markdown("### 📋 Recent Daily Activity")
        
        # Get last 7 days of data
        recent_dates = sorted(daily_stats.keys())[-7:]
        recent_data = []
        
        for date in recent_dates:
            day_data = daily_stats[date]
            recent_data.append({
                "Date": date,
                "Sessions": day_data["sessions"],
                "Models Run": day_data["models_run"],
                "Files Uploaded": day_data["files_uploaded"]
            })
        
        recent_df = pd.DataFrame(recent_data)
        st.dataframe(recent_df, use_container_width=True)
    
    # Raw data expander
    with st.expander("🔍 View Raw Usage Data"):
        st.json(usage_data)

# Email feedback function with daily limit
def send_feedback_email(feedback_text):
    """
    Send feedback via Formspree service (no backend required).
    Also saves feedback locally as backup.
    """
    try:
        import requests
        import os
        from datetime import datetime, date
        
        # Check daily limit (5 emails per day)
        today = date.today().strftime("%Y-%m-%d")
        count_file = f"email_count_{today}.txt"
        
        current_count = 0
        if os.path.exists(count_file):
            try:
                with open(count_file, "r") as f:
                    current_count = int(f.read().strip())
            except:
                current_count = 0
        
        if current_count >= 5:
            return False  # Daily limit reached
        
        # Save feedback locally as backup
        timestamp = datetime.now(pytz.timezone('US/Central')).strftime("%Y-%m-%d %H:%M:%S CST")
        feedback_entry = f"\n--- Feedback submitted on {timestamp} ---\n{feedback_text}\n"
        
        with open("user_feedback.txt", "a", encoding="utf-8") as f:
            f.write(feedback_entry)
        
        # Send via Formspree (free service, no backend needed)
        # Use a simpler approach that works better with Gmail
        formspree_url = "https://formspree.io/f/xjkeegpn"  # Your actual Formspree endpoint
        
        # Create a clean, Gmail-friendly email format
        email_data = {
            "name": "Econometric Analysis Tool",
            "email": "feedback@econometrictool.app",  # Use a professional-looking email
            "subject": f"📊 New Feedback - {timestamp}",
            "message": f"""📊 FEEDBACK RECEIVED FROM ECONOMETRIC ANALYSIS TOOL

✅ Content: {feedback_text}

📅 Submitted: {timestamp}
🔧 Source: Streamlit Supervised Learning Tool
🎯 App: Econometric Analysis Tool

---
Recipients: r_z79@txstate.edu, zhangren080@gmail.com

This message was automatically generated by the feedback system.
            """,
            "_replyto": "feedback@econometrictool.app"
        }
        
        try:
            # Send email via Formspree
            response = requests.post(formspree_url, data=email_data, timeout=10)
            
            if response.status_code == 200:
                # Email sent successfully, increment counter
                with open(count_file, "w") as f:
                    f.write(str(current_count + 1))
                return True
            else:
                # Formspree failed, but feedback saved locally
                return True  # Don't show error to user
                
        except requests.exceptions.RequestException:
            # Network error, but feedback saved locally
            return True  # Don't show error to user
            
    except Exception as e:
        # Any other error, feedback still saved locally
        return True
        feedback_entry = f"\n--- Feedback submitted on {timestamp} ---\n{feedback_text}\n"
        
        with open("user_feedback.txt", "a", encoding="utf-8") as f:
            f.write(feedback_entry)
        
        # Increment count
        with open(count_file, "w") as f:
            f.write(str(current_count + 1))
        
        # Send email notification
        try:
            # Use a simple email service (for testing purposes)
            # In production, you would configure SMTP properly
            
            # For now, we'll use a simple approach that works with Gmail
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = "noreply@streamlit.app"  # Placeholder sender
            msg['To'] = "r_z79@txstate.edu, zhangren080@gmail.com"
            msg['Subject'] = f"App Feedback - {timestamp}"
            
            body = f"""New feedback received from your Supervised Learning Tool:

Feedback: {feedback_text}

Timestamp: {timestamp}
Source: Streamlit App
"""
            msg.attach(MIMEText(body, 'plain'))
            
            # For actual email sending, you would need SMTP configuration
            # This is commented out until you set up email credentials
            # 
            # server = smtplib.SMTP('smtp.gmail.com', 587)
            # server.starttls()
            # server.login("your_email@gmail.com", "your_app_password")
            # server.sendmail("your_email@gmail.com", ["r_z79@txstate.edu", "zhangren080@gmail.com"], msg.as_string())
            # server.quit()
            
        except Exception as e:
            pass  # Email failed, but feedback is still saved locally
        
        # TODO: Uncomment when you set up email credentials in Streamlit secrets
        # gmail_user = st.secrets["email"]["gmail_user"]
        # gmail_password = st.secrets["email"]["gmail_password"]
        # 
        # msg = MIMEMultipart()
        # msg['From'] = gmail_user
        # msg['To'] = "r_z79@txstate.edu"
        # msg['Subject'] = f"App Feedback - {timestamp}"
        # 
        # body = f"New feedback received:\n\n{feedback_text}\n\nTimestamp: {timestamp}"
        # msg.attach(MIMEText(body, 'plain'))
        # 
        # server = smtplib.SMTP('smtp.gmail.com', 587)
        # server.starttls()
        # server.login(gmail_user, gmail_password)
        # text = msg.as_string()
        # server.sendmail(gmail_user, "r_z79@txstate.edu", text)
        # server.quit()
        
        return True
        
    except Exception as e:
        return True  # Graceful fallback

def create_interactive_tree_plot(model, feature_names, class_names=None, max_depth=None, prob_class_index=0):
    """
    Clean, simple decision tree visualization with proper node sizes and all depths visible.
    """
    tree = model.tree_
    
    # Add validation for heavily pruned trees
    if tree.node_count <= 0:
        # Return empty figure for empty trees
        fig = go.Figure()
        fig.add_annotation(
            x=0.5, y=0.5,
            xref='paper', yref='paper',
            text="Tree is empty (heavily pruned)",
            showarrow=False,
            font=dict(size=14, color='orange'),
            xanchor='center', yanchor='middle'
        )
        fig.update_layout(
            title="Empty Decision Tree",
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=300
        )
        return fig
    
    # Validate prob_class_index for binary classification
    if class_names is not None and len(class_names) == 2:
        if prob_class_index not in [0, 1]:
            prob_class_index = 0  # Default to class 0 if invalid
    elif class_names is not None:
        # For multiclass, default to 0
        prob_class_index = 0
    
    # Simple recursive function to calculate node positions with spacing for variable-sized nodes
    def calculate_positions(node_id=0, x=0, y=0, level=0, h_spacing=160):
        if node_id < 0 or node_id >= tree.node_count:
            return {}
        
        positions = {node_id: (x, y)}
        
        # Stop if we've reached max depth
        if max_depth is not None and level >= max_depth:
            return positions
        
        # Get children
        left_child = tree.children_left[node_id]
        right_child = tree.children_right[node_id]
        
        # If this is not a leaf node
        if left_child != right_child:
            # Spacing increases for higher levels (bigger nodes) and decreases for lower levels
            level_factor = max(1.0 - (level * 0.2), 0.4)  # Reduces spacing as depth increases
            spacing = max(h_spacing * level_factor / (1 + level * 0.2), 45)  # Min 45 to prevent overlap
            child_y = y - 24  # More vertical spacing for edge labels and larger top nodes
            
            # Add left child
            if left_child >= 0:
                left_positions = calculate_positions(left_child, x - spacing, child_y, level + 1, h_spacing)
                positions.update(left_positions)
            
            # Add right child  
            if right_child >= 0:
                right_positions = calculate_positions(right_child, x + spacing, child_y, level + 1, h_spacing)
                positions.update(right_positions)
        
        return positions
    
    # Calculate all node positions
    positions = calculate_positions()
    
    # Calculate level-based node sizing (same size within each level)
    levels = {}
    for node_id, (x, y) in positions.items():
        level = int(round((15 - y) / 22))  # Calculate level from y position
        if level not in levels:
            levels[level] = []
        levels[level].append(node_id)
    
    max_level = max(levels.keys()) if levels else 0
    
    # Simple, clean node sizing based on your reference image proportions
    # Much more aggressive width reduction for levels 3-5 to prevent overlapping
    
    level_sizes = {}
    for level in range(max_level + 1):
        # Base sizing that matches your reference image proportions
        # Much smaller widths for deeper levels
        if level == 0:  # Root level
            width = 120
            height = 40
            font_size = 14
        elif level == 1:  # Second level
            width = 100
            height = 35
            font_size = 12
        elif level == 2:  # Third level  
            width = 85
            height = 30
            font_size = 11
        elif level == 3:  # Fourth level - much smaller
            width = 40
            height = 22
            font_size = 8
        elif level == 4:  # Fifth level - very small
            width = 30
            height = 18
            font_size = 7
        else:  # Sixth level and deeper - extremely small
            width = 25
            height = 15
            font_size = 6
        
        level_sizes[level] = {
            'width': width,
            'height': height,
            'font_size': font_size
        }
    
    # Get Y range with generous padding
    all_y = [pos[1] for pos in positions.values()]
    min_y = min(all_y) - 25
    max_y = max(all_y) + 15
    
    # Create figure
    fig = go.Figure()
    
    # Draw edges first
    for node_id in positions:
        if node_id >= tree.node_count:
            continue
            
        left_child = tree.children_left[node_id]
        right_child = tree.children_right[node_id]
        
        x, y = positions[node_id]
        
        # Draw edge to left child
        if left_child >= 0 and left_child in positions:
            x_child, y_child = positions[left_child]
            fig.add_trace(go.Scatter(
                x=[x, x_child], y=[y, y_child],
                mode='lines',
                line=dict(color='black', width=2),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Draw edge to right child
        if right_child >= 0 and right_child in positions:
            x_child, y_child = positions[right_child]
            fig.add_trace(go.Scatter(
                x=[x, x_child], y=[y, y_child],
                mode='lines', 
                line=dict(color='black', width=2),
                showlegend=False,
                hoverinfo='skip'
            ))
    
    # Calculate color scaling values
    all_values = []
    for node_id in positions:
        if node_id >= tree.node_count:
            continue
        if class_names is not None:  # Classification
            probs = tree.value[node_id][0] / tree.value[node_id][0].sum()
            # Use specified class probability, with better validation
            if prob_class_index < len(probs):
                all_values.append(float(probs[prob_class_index]))
            else:
                # Fallback to class 0 if index is invalid
                all_values.append(float(probs[0]))
        else:  # Regression
            all_values.append(float(tree.value[node_id][0][0]))
    
    min_val = min(all_values) if all_values else 0
    max_val = max(all_values) if all_values else 1
    
    # Create feature mapping with proper abbreviations BEFORE drawing nodes
    feature_mapping = {}
    used_features = []
    abbrev_count = {}
    
    # Collect all features used in the tree
    for node_id in positions:
        if node_id >= tree.node_count:
            continue
        if tree.children_left[node_id] != tree.children_right[node_id]:
            # Validate feature index before accessing feature_names
            feature_idx = tree.feature[node_id]
            if 0 <= feature_idx < len(feature_names):
                feature_name = feature_names[feature_idx]
                if feature_name not in used_features:
                    used_features.append(feature_name)
            else:
                # Skip invalid feature indices (can happen with heavy pruning)
                continue
    
    # Create abbreviations with numbering for duplicates
    for feature_name in used_features:
        # Create base abbreviation (first 3 letters)
        if len(feature_name) >= 3:
            base_abbrev = feature_name[:3].lower()
        else:
            base_abbrev = feature_name.lower()
        
        # Handle duplicates by adding numbers
        if base_abbrev in abbrev_count:
            abbrev_count[base_abbrev] += 1
            final_abbrev = f"{base_abbrev}{abbrev_count[base_abbrev]}"
        else:
            abbrev_count[base_abbrev] = 1
            # Check if there will be future conflicts
            future_conflicts = [f for f in used_features if f != feature_name and len(f) >= 3 and f[:3].lower() == base_abbrev]
            if future_conflicts:
                final_abbrev = f"{base_abbrev}1"
            else:
                final_abbrev = base_abbrev
        
        feature_mapping[feature_name] = final_abbrev
    
    # Draw nodes
    for node_id in positions:
        if node_id >= tree.node_count:
            continue
            
        x, y = positions[node_id]
        
        # Get level-specific sizing
        level = int(round((15 - y) / 22))
        node_sizing = level_sizes.get(level, level_sizes[0])
        node_width = node_sizing['width']
        node_height = node_sizing['height']
        font_size = node_sizing['font_size']
        
        # Calculate node value and percentage
        samples = int(tree.n_node_samples[node_id])
        total_samples = int(tree.n_node_samples[0])
        percentage = (samples / total_samples) * 100
        
        if class_names is not None:  # Classification
            probs = tree.value[node_id][0] / tree.value[node_id][0].sum()
            # Use specified class probability, with better validation
            if prob_class_index < len(probs):
                main_value = float(probs[prob_class_index])
            else:
                # Fallback to class 0 if index is invalid
                main_value = float(probs[0])
        else:  # Regression
            main_value = float(tree.value[node_id][0][0])
        
        # Calculate color based on value (keeping your good color scheme)
        if max_val > min_val:
            color_intensity = (main_value - min_val) / (max_val - min_val)
        else:
            color_intensity = 0.5
        
        # Your approved color scheme: dark blue (low) to dark red (high) with better text visibility
        if color_intensity < 0.2:
            node_color = 'rgba(30, 70, 150, 0.95)'   # Dark blue
            text_color = 'white'  # White text on dark blue
        elif color_intensity < 0.4:
            node_color = 'rgba(173, 216, 230, 0.95)' # Light blue
            text_color = 'black'  # Black text on light blue
        elif color_intensity < 0.6:
            node_color = 'rgba(255, 140, 105, 0.95)' # Light coral
            text_color = 'black'  # Black text on coral
        elif color_intensity < 0.8:
            node_color = 'rgba(255, 69, 58, 0.95)'   # Red-orange
            text_color = 'white'  # White text on red-orange
        else:
            node_color = 'rgba(220, 20, 20, 0.95)'   # Dark red
            text_color = 'white'  # White text on dark red
        
        # Draw rectangular node with level-specific sizing
        # node_width and node_height already set above based on level
        
        fig.add_shape(
            type="rect",
            x0=x - node_width/2, y0=y - node_height/2,
            x1=x + node_width/2, y1=y + node_height/2,
            fillcolor=node_color,
            line=dict(color='black', width=2)
        )
        
        # Create detailed hover information
        if class_names is not None:  # Classification
            probs = tree.value[node_id][0] / tree.value[node_id][0].sum()
            class_probs_text = "<br>".join([f"{class_names[i]}: {prob:.3f}" for i, prob in enumerate(probs)])
            
            # Show which probability is being displayed
            displayed_class = class_names[prob_class_index] if len(class_names) > prob_class_index else "Max Class"
            displayed_prob_text = f"<br><b>Displayed Probability ({displayed_class}): {main_value:.3f}</b>"
            
            hover_text = f"""
            <b>Node {node_id}</b><br>
            Level: {level}<br>
            Samples: {samples}<br>
            Percentage: {percentage:.1f}%<br>
            {displayed_prob_text}<br>
            <br><b>All Class Probabilities:</b><br>
            {class_probs_text}<br>
            <br>Predicted Class: {class_names[probs.argmax()]}<br>
            Confidence: {probs.max():.3f}
            """
        else:  # Regression
            hover_text = f"""
            <b>Node {node_id}</b><br>
            Level: {level}<br>
            Samples: {samples}<br>
            Percentage: {percentage:.1f}%<br>
            <br>Predicted Value: {main_value:.3f}<br>
            Mean Squared Error: {tree.impurity[node_id]:.3f}
            """
        
        # Add invisible scatter point for hover functionality
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='markers',
            marker=dict(size=max(node_width, node_height), color='rgba(0,0,0,0)', opacity=0),
            hovertext=hover_text,
            hoverinfo='text',
            showlegend=False,
            name=''
        ))
        
        # Add node text: value on top line, percentage on bottom line (2 decimal places)
        node_text = f"<b>{main_value:.2f}</b><br><b>{percentage:.0f}%</b>"
        
        fig.add_annotation(
            x=x, y=y,
            text=node_text,
            showarrow=False,
            font=dict(size=font_size, color=text_color, family='Arial Bold'),  # Level-specific font size and optimal text color
            borderwidth=0
        )
        
        # Add decision rule on the LEFT edge with variable ABOVE and value BELOW
        if tree.children_left[node_id] != tree.children_right[node_id]:
            # Validate feature index before accessing feature_names
            feature_idx = tree.feature[node_id]
            if 0 <= feature_idx < len(feature_names):
                feature_name = feature_names[feature_idx]
                threshold = tree.threshold[node_id]
                
                # Use the abbreviation from our mapping
                abbrev = feature_mapping.get(feature_name, feature_name[:3].lower())
                
                # Simplified edge labeling: variable name on LEFT, threshold on RIGHT
                left_child = tree.children_left[node_id]
                right_child = tree.children_right[node_id]
            
            # LEFT edge: show variable name (green = below threshold)
            if left_child >= 0 and left_child in positions:
                x_child, y_child = positions[left_child]
                mid_x = (x + x_child) / 2
                mid_y = (y + y_child) / 2
                
                # Variable name on left edge (green = below)
                fig.add_annotation(
                    x=mid_x, y=mid_y,
                    text=f"<b>{abbrev}</b>",
                    showarrow=False,
                    font=dict(size=12, color='green', family='Arial Bold'),
                    bgcolor='rgba(255,255,255,0.9)',
                    bordercolor='green',
                    borderwidth=1,
                    borderpad=2
                )
            
            # RIGHT edge: show threshold value (red = above threshold)
            if right_child >= 0 and right_child in positions:
                x_child, y_child = positions[right_child]
                mid_x = (x + x_child) / 2
                mid_y = (y + y_child) / 2
                
                # Threshold value on right edge (red = above)
                fig.add_annotation(
                    x=mid_x, y=mid_y,
                    text=f"<b>{threshold:.1f}</b>",
                    showarrow=False,
                    font=dict(size=12, color='red', family='Arial Bold'),
                    bgcolor='rgba(255,255,255,0.9)',
                    bordercolor='red',
                    borderwidth=1,
                    borderpad=2
                )
    
    # Add heatmap color legend
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(
            size=1,
            color=[min_val, max_val],
            colorscale=[
                [0, 'rgb(30,70,150)'],        # Dark blue for small values
                [0.25, 'rgb(173,216,230)'],   # Light blue
                [0.5, 'rgb(255,140,105)'],    # Light coral (middle)
                [0.75, 'rgb(255,69,58)'],     # Red-orange
                [1, 'rgb(220,20,20)']         # Dark red for big values
            ],
            showscale=True,
            colorbar=dict(
                title=dict(
                    text="<b>Probability/Value</b>",
                    font=dict(size=14, family='Arial Bold')
                ),
                thickness=20,
                len=0.6,
                x=1.02,
                tickfont=dict(size=11, family='Arial')
            )
        ),
        showlegend=False,
        name='Color Scale'
    ))
    
    # Update layout for proper display with legend, wider spacing to prevent overlap
    fig.update_layout(
        title="Decision Tree Visualization",
        showlegend=False,
        hovermode='closest',
        margin=dict(b=150, l=150, r=150, t=100),  # More space on left for threshold values
        xaxis=dict(
            showgrid=False,
            zeroline=False, 
            showticklabels=False,
            range=[min([p[0] for p in positions.values()]) - 40, 
                   max([p[0] for p in positions.values()]) + 40]  # More horizontal space
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False, 
            range=[min_y, max_y]
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=1600,  # Wider to accommodate better spacing
        height=1200   # Taller for better vertical spacing
    )
    
    # Add feature mapping note at the bottom if there are abbreviated features
    if feature_mapping:
        note_text = "Feature abbreviations: " + ", ".join([f"{abbrev} = {full}" for full, abbrev in feature_mapping.items()])
        fig.add_annotation(
            x=0.5, y=-0.12,
            xref='paper', yref='paper',
            text=note_text,
            showarrow=False,
            font=dict(size=11, color='black', family='Arial'),
            xanchor='center'
        )

    # Always return the figure (previously indented inside the if-block)
    return fig

def create_confusion_matrix_plot(y_true, y_pred, class_names=None):
    """
    Create an interactive confusion matrix visualization using Plotly
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(cm))]
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=class_names,
        y=class_names,
        colorscale='Blues',
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 16},
        showscale=True
    ))
    
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted Class",
        yaxis_title="Actual Class",
        width=500,
        height=400
    )
    
    return fig

def create_coefficients_plot(model, feature_names):
    """
    Create a visualization of logistic regression coefficients
    """
    if hasattr(model, 'coef_') and hasattr(model, 'intercept_'):
        coeffs = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
        intercept = model.intercept_[0] if isinstance(model.intercept_, np.ndarray) else model.intercept_
        
        # Create coefficient dataframe
        coef_df = pd.DataFrame({
            'Feature': feature_names + ['Intercept'],
            'Coefficient': np.append(coeffs, intercept),
            'Abs_Coefficient': np.append(np.abs(coeffs), np.abs(intercept))
        }).sort_values('Abs_Coefficient', ascending=True)
        
        # Create horizontal bar plot
        colors = ['red' if x < 0 else 'blue' for x in coef_df['Coefficient']]
        
        fig = go.Figure(data=go.Bar(
            y=coef_df['Feature'],
            x=coef_df['Coefficient'],
            orientation='h',
            marker_color=colors,
            text=[f"{x:.3f}" for x in coef_df['Coefficient']],
            textposition='outside'
        ))
        
        fig.update_layout(
            title="Logistic Regression Coefficients",
            xaxis_title="Coefficient Value",
            yaxis_title="Features",
            height=max(400, len(feature_names) * 30 + 100),
            margin=dict(l=150, r=50, t=50, b=50)
        )
        
        return fig
    return None

def create_actual_vs_predicted_plot(y_true, y_pred, y_pred_proba=None):
    """
    Create actual vs predicted visualization for classification
    """
    fig = go.Figure()
    
    # Scatter plot of actual vs predicted
    fig.add_trace(go.Scatter(
        x=list(range(len(y_true))),
        y=y_true,
        mode='markers',
        name='Actual',
        marker=dict(color='blue', size=8, opacity=0.6),
        hovertemplate='Index: %{x}<br>Actual: %{y}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=list(range(len(y_pred))),
        y=y_pred,
        mode='markers',
        name='Predicted',
        marker=dict(color='red', size=8, opacity=0.6, symbol='x'),
        hovertemplate='Index: %{x}<br>Predicted: %{y}<extra></extra>'
    ))
    
    # Add probability visualization if available
    if y_pred_proba is not None:
        fig.add_trace(go.Scatter(
            x=list(range(len(y_pred_proba))),
            y=y_pred_proba,
            mode='markers',
            name='Probability',
            marker=dict(color='green', size=6, opacity=0.4),
            yaxis='y2',
            hovertemplate='Index: %{x}<br>Probability: %{y:.3f}<extra></extra>'
        ))
        
        # Add secondary y-axis for probabilities
        fig.update_layout(
            yaxis2=dict(
                title="Predicted Probability",
                overlaying='y',
                side='right',
                range=[0, 1]
            )
        )
    
    fig.update_layout(
        title="Actual vs Predicted Classes",
        xaxis_title="Sample Index",
        yaxis_title="Class (0/1)",
        hovermode='closest',
        height=400
    )
    
    return fig

def create_forest_importance_plot(model, feature_names):
    """
    Create a feature importance plot for Random Forest models.
    """
    importance = model.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=True)
    
    # Create horizontal bar plot
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=feature_importance['importance'],
        y=feature_importance['feature'],
        orientation='h',
        marker=dict(color='skyblue', opacity=0.8),
        text=feature_importance['importance'].round(3),
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Feature Importance (Random Forest)",
        xaxis_title="Importance",
        yaxis_title="Features",
        height=max(400, len(feature_names) * 50),
        margin=dict(l=150, r=50, t=50, b=50)
    )
    
    return fig

def create_pruning_visualization(pruning_info):
    """Create visualization of the cost complexity pruning path"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Validate input
    if not isinstance(pruning_info, dict):
        # Return a simple error figure
        fig = go.Figure()
        fig.add_annotation(
            x=0.5, y=0.5,
            xref='paper', yref='paper',
            text="Error: Invalid pruning information",
            showarrow=False,
            font=dict(size=14, color='red'),
            xanchor='center', yanchor='middle'
        )
        fig.update_layout(
            title="Pruning Visualization Error",
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=200
        )
        return fig
    
    if 'manual_alpha' in pruning_info:
        # For manual alpha, create a simple display
        try:
            manual_alpha_value = pruning_info['manual_alpha']
            fig = go.Figure()
            fig.add_annotation(
                x=0.5, y=0.5,
                xref='paper', yref='paper',
                text=f"Manual Alpha Used: {manual_alpha_value:.6f}",
                showarrow=False,
                font=dict(size=16, color='black'),
                xanchor='center', yanchor='middle'
            )
            fig.update_layout(
                title="Cost Complexity Pruning: Manual Alpha",
                showlegend=False,
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=200
            )
            return fig
        except Exception as e:
            # Return error figure if manual alpha processing fails
            fig = go.Figure()
            fig.add_annotation(
                x=0.5, y=0.5,
                xref='paper', yref='paper',
                text=f"Error processing manual alpha: {str(e)}",
                showarrow=False,
                font=dict(size=12, color='red'),
                xanchor='center', yanchor='middle'
            )
            fig.update_layout(
                title="Manual Alpha Error",
                showlegend=False,
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=200
            )
            return fig
    
    # For cross-validation results
    ccp_alphas = pruning_info.get('ccp_alphas', [])
    cv_scores = pruning_info.get('cv_scores', [])
    optimal_alpha = pruning_info.get('optimal_alpha', 0)
    
    if len(ccp_alphas) == 0:
        # No pruning path available
        fig = go.Figure()
        fig.add_annotation(
            x=0.5, y=0.5,
            xref='paper', yref='paper',
            text="No pruning path available (tree may be already optimal)",
            showarrow=False,
            font=dict(size=14, color='orange'),
            xanchor='center', yanchor='middle'
        )
        fig.update_layout(
            title="Cost Complexity Pruning: No Path Available",
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=200
        )
        return fig
    
    # Create the pruning path visualization
    fig = go.Figure()
    
    # Plot CV scores vs alpha
    fig.add_trace(go.Scatter(
        x=ccp_alphas,
        y=cv_scores,
        mode='lines+markers',
        name='Cross-Validation Score',
        line=dict(color='blue', width=2),
        marker=dict(size=6, color='blue')
    ))
    
    # Highlight optimal alpha
    optimal_idx = list(ccp_alphas).index(optimal_alpha) if optimal_alpha in ccp_alphas else 0
    fig.add_trace(go.Scatter(
        x=[optimal_alpha],
        y=[cv_scores[optimal_idx]],
        mode='markers',
        name='Optimal α',
        marker=dict(size=12, color='red', symbol='star')
    ))
    
    # Add vertical line at optimal alpha
    fig.add_vline(
        x=optimal_alpha,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Optimal α = {optimal_alpha:.6f}"
    )
    
    fig.update_layout(
        title="Cost Complexity Pruning: Cross-Validation Results",
        xaxis_title="Cost Complexity Parameter (α)",
        yaxis_title="Cross-Validation Score",
        showlegend=True,
        height=400,
        hovermode='x unified'
    )
    
    return fig

def display_pruning_info(estimation_method):
    """Display pruning information if available"""
    if hasattr(st.session_state, 'pruning_info'):
        method_key = estimation_method.lower().replace(' ', '_')
        if method_key in st.session_state.pruning_info:
            pruning_info = st.session_state.pruning_info[method_key]
            
            st.markdown("---")
            st.markdown('<h2 class="subheader">🌿 Cost Complexity Pruning Results</h2>', unsafe_allow_html=True)
            
            # Display key metrics
            col1, col2, col3 = st.columns(3)
            
            if 'manual_alpha' in pruning_info:
                with col1:
                    st.metric("Pruning Method", "Manual Alpha")
                with col2:
                    st.metric("Alpha Value", f"{pruning_info['manual_alpha']:.6f}")
                with col3:
                    st.metric("Status", "Applied")
            else:
                optimal_alpha = pruning_info.get('optimal_alpha', 0)
                optimal_score = pruning_info.get('optimal_score', 0)
                n_alphas = pruning_info.get('n_alphas_tested', 0)
                
                with col1:
                    st.metric("Optimal Alpha", f"{optimal_alpha:.6f}")
                with col2:
                    st.metric("CV Score", f"{optimal_score:.4f}")
                with col3:
                    st.metric("Alphas Tested", n_alphas)
            
            # Create and display visualization
            try:
                pruning_fig = create_pruning_visualization(pruning_info)
                if pruning_fig is not None:
                    st.plotly_chart(pruning_fig, use_container_width=True)
                else:
                    st.error("Error: Could not create pruning visualization")
            except Exception as e:
                st.error(f"Error creating pruning visualization: {str(e)}")
                # Create a simple fallback display
                if 'manual_alpha' in pruning_info:
                    st.info(f"Manual Alpha Used: {pruning_info['manual_alpha']:.6f}")
                else:
                    st.info("Pruning information available but visualization failed")
            
            # Explanation
            with st.expander("📖 Understanding Cost Complexity Pruning"):
                st.markdown("""
                **Cost Complexity Pruning** follows Algorithm 8.1 from ISLR:
                
                1. **Grow Large Tree**: Start with a large tree grown on training data
                2. **Generate Subtrees**: Apply cost complexity pruning to obtain sequence of subtrees as function of α
                3. **Cross-Validation**: Use K-fold CV to choose optimal α that minimizes prediction error
                4. **Final Model**: Return the subtree corresponding to chosen α
                
                **Key Points:**
                - **Higher α** = More pruning (simpler tree)
                - **Lower α** = Less pruning (more complex tree)
                - **Optimal α** balances bias-variance tradeoff
                - **CV Score** indicates model performance with given α
                """)

def calculate_regression_stats(X, y, model, method='OLS', fit_intercept=True):
    """Calculate comprehensive regression statistics for different methods"""
    # Predictions
    y_pred = model.predict(X)
    
    # Basic statistics
    n = len(y)
    k = X.shape[1]  # number of features
    
    # R-squared - use sklearn's score method for regularized models as it's more accurate
    if method in ['Lasso', 'Ridge', 'Elastic Net']:
        r_squared = model.score(X, y)  # sklearn's R² calculation
    else:
        # Manual calculation for other methods
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
    
    # Adjusted R-squared (account for constant term)
    k_adj = k + (1 if fit_intercept else 0)  # Add 1 for intercept if included
    adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k_adj)
    
    # Mean Squared Error and Root Mean Squared Error
    ss_res = np.sum((y - y_pred) ** 2)  # Recalculate for MSE
    mse = ss_res / (n - k_adj) if method == 'OLS' else ss_res / n
    rmse = np.sqrt(mse)
    
    # Calculate residuals
    residuals = y - y_pred
    
    # For OLS, calculate standard errors and statistical tests
    if method == 'OLS':
        # Standard errors of coefficients
        if fit_intercept:
            X_with_intercept = np.column_stack([np.ones(n), X])
            # Handle intercept properly for different model types
            intercept_val = float(model.intercept_) if hasattr(model, 'intercept_') else 0.0
            coefficients = np.concatenate([[intercept_val], model.coef_])
        else:
            X_with_intercept = X
            coefficients = model.coef_
        
        try:
            # Variance-covariance matrix
            var_cov_matrix = mse * np.linalg.inv(X_with_intercept.T @ X_with_intercept)
            std_errors = np.sqrt(np.diag(var_cov_matrix))
            
            # T-statistics
            t_stats = coefficients / std_errors
            
            # P-values (two-tailed test)
            p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - k_adj))
            
        except np.linalg.LinAlgError:
            # If matrix is singular, return NaN values
            std_errors = np.full(k_adj, np.nan)
            t_stats = np.full(k_adj, np.nan)
            p_values = np.full(k_adj, np.nan)
        
        # F-statistic (adjust for intercept)
        if fit_intercept:
            f_stat = (r_squared / k) / ((1 - r_squared) / (n - k - 1))
            f_p_value = 1 - stats.f.cdf(f_stat, k, n - k - 1)
        else:
            f_stat = (r_squared / k) / ((1 - r_squared) / (n - k))
            f_p_value = 1 - stats.f.cdf(f_stat, k, n - k)
    else:
        # For regularized methods, we don't calculate traditional statistical tests
        k_adj = k + (1 if fit_intercept else 0)
        std_errors = np.full(k_adj, np.nan)
        t_stats = np.full(k_adj, np.nan)
        p_values = np.full(k_adj, np.nan)
        f_stat = np.nan
        f_p_value = np.nan
    
    return {
        'r_squared': r_squared,
        'adj_r_squared': adj_r_squared,
        'mse': mse,
        'rmse': rmse,
        'std_errors': std_errors,
        't_stats': t_stats,
        'p_values': p_values,
        'f_stat': f_stat,
        'f_p_value': f_p_value,
        'residuals': residuals,
        'fitted_values': y_pred,
        'n_obs': n,
        'n_features': k,
        'method': method
    }

def find_optimal_ccp_alpha(X, y, model_class, cv_folds=5, **model_params):
    """
    Implement cost complexity pruning algorithm based on Algorithm 8.1
    
    This function implements the steps from Algorithm 8.1:
    1. Grow a large tree on training data
    2. Apply cost complexity pruning to obtain subtrees
    3. Use K-fold cross-validation to choose optimal alpha
    4. Return the optimal alpha value
    """
    
    # Step 1: Grow a large tree (without depth limit for initial tree)
    large_tree_params = model_params.copy()
    large_tree_params['max_depth'] = None  # Remove depth limit for initial large tree
    large_tree = model_class(**large_tree_params)
    large_tree.fit(X, y)
    
    # Step 2: Get cost complexity pruning path (sequence of alpha values and corresponding subtrees)
    path = large_tree.cost_complexity_pruning_path(X, y)
    ccp_alphas = path.ccp_alphas
    
    # Remove the last alpha (which gives empty tree)
    ccp_alphas = ccp_alphas[:-1]
    
    if len(ccp_alphas) == 0:
        return 0.0, {}
    
    # Step 3: Use K-fold cross-validation to choose optimal alpha
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = []
    
    for alpha in ccp_alphas:
        fold_scores = []
        
        # Step 3a & 3b: For each fold, train on K-1 folds and evaluate on kth fold
        for train_idx, val_idx in kf.split(X):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train model with current alpha
            model_params_alpha = model_params.copy()
            model_params_alpha['ccp_alpha'] = alpha
            fold_model = model_class(**model_params_alpha)
            fold_model.fit(X_train_fold, y_train_fold)
            
            # Evaluate on validation fold
            y_pred = fold_model.predict(X_val_fold)
            
            # Use appropriate metric based on model type
            if hasattr(fold_model, 'predict_proba'):  # Classification
                try:
                    y_pred_proba = fold_model.predict_proba(X_val_fold)
                    score = -log_loss(y_val_fold, y_pred_proba)  # Negative for maximization
                except:
                    score = accuracy_score(y_val_fold, y_pred)
            else:  # Regression
                score = -mean_squared_error(y_val_fold, y_pred)  # Negative MSE for maximization
            
            fold_scores.append(score)
        
        # Average scores across folds for this alpha
        cv_scores.append(np.mean(fold_scores))
    
    # Step 4: Choose alpha that minimizes average error (maximizes average score)
    optimal_idx = np.argmax(cv_scores)
    optimal_alpha = ccp_alphas[optimal_idx]
    
    # Return results including pruning information
    pruning_info = {
        'ccp_alphas': ccp_alphas,
        'cv_scores': cv_scores,
        'optimal_alpha': optimal_alpha,
        'optimal_score': cv_scores[optimal_idx],
        'n_alphas_tested': len(ccp_alphas)
    }
    
    return optimal_alpha, pruning_info

def fit_model(X, y, method, alpha=1.0, l1_ratio=0.5, fit_intercept=True, **kwargs):
    """Fit model based on selected method"""
    if method == 'OLS':
        model = LinearRegression(fit_intercept=fit_intercept)
    elif method == 'Lasso':
        model = Lasso(alpha=alpha, fit_intercept=fit_intercept, random_state=42)
    elif method == 'Ridge':
        model = Ridge(alpha=alpha, fit_intercept=fit_intercept, random_state=42)
    elif method == 'Elastic Net':
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=fit_intercept, random_state=42)
    elif method == 'Logistic Regression':
        # Get class balancing option from kwargs
        class_weight = kwargs.get('class_weight', None)
        model = LogisticRegression(fit_intercept=fit_intercept, random_state=42, max_iter=1000, class_weight=class_weight)
    elif method == 'Decision Tree':
        model_type = kwargs.get('model_type', 'regression')
        max_depth = kwargs.get('max_depth', None)
        min_samples_split = kwargs.get('min_samples_split', 2)
        min_samples_leaf = kwargs.get('min_samples_leaf', 1)
        enable_pruning = kwargs.get('enable_pruning', False)
        cv_folds = kwargs.get('cv_folds', 5)
        pruning_method = kwargs.get('pruning_method', 'Automatic (CV)')
        manual_alpha = kwargs.get('manual_alpha', None)
        
        # Base model parameters
        base_params = {
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'random_state': 42
        }
        
        # Determine model class
        if model_type == 'classification':
            model_class = DecisionTreeClassifier
        else:
            model_class = DecisionTreeRegressor
        
        # Apply pruning if enabled
        if enable_pruning:
            if pruning_method == "Manual Alpha" and manual_alpha is not None:
                # Use manual alpha
                optimal_alpha = manual_alpha
                pruning_info = {'manual_alpha': manual_alpha}
            else:
                # Use cross-validation to find optimal alpha
                optimal_alpha, pruning_info = find_optimal_ccp_alpha(
                    X, y, model_class, cv_folds, **base_params
                )
            
            # Add ccp_alpha to model parameters
            base_params['ccp_alpha'] = optimal_alpha
            
            # Store pruning info for later display
            if not hasattr(st.session_state, 'pruning_info'):
                st.session_state.pruning_info = {}
            st.session_state.pruning_info['decision_tree'] = pruning_info
        
        # Create and fit the model
        model = model_class(**base_params)
        
    elif method == 'Random Forest':
        model_type = kwargs.get('model_type', 'regression')
        n_estimators = kwargs.get('n_estimators', 100)
        max_depth = kwargs.get('max_depth', None)
        min_samples_split = kwargs.get('min_samples_split', 2)
        min_samples_leaf = kwargs.get('min_samples_leaf', 1)
        enable_pruning = kwargs.get('enable_pruning', False)
        cv_folds = kwargs.get('cv_folds', 5)
        pruning_method = kwargs.get('pruning_method', 'Automatic (CV)')
        manual_alpha = kwargs.get('manual_alpha', None)
        
        # Base model parameters
        base_params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'random_state': 42
        }
        
        # Determine model class
        if model_type == 'classification':
            model_class = RandomForestClassifier
        else:
            model_class = RandomForestRegressor
        
        # Apply pruning if enabled
        if enable_pruning:
            if pruning_method == "Manual Alpha" and manual_alpha is not None:
                # Use manual alpha
                optimal_alpha = manual_alpha
                pruning_info = {'manual_alpha': manual_alpha}
            else:
                # Use cross-validation to find optimal alpha (using a single tree for alpha estimation)
                single_tree_params = {k: v for k, v in base_params.items() if k != 'n_estimators'}
                single_tree_class = DecisionTreeClassifier if model_type == 'classification' else DecisionTreeRegressor
                optimal_alpha, pruning_info = find_optimal_ccp_alpha(
                    X, y, single_tree_class, cv_folds, **single_tree_params
                )
            
            # Add ccp_alpha to model parameters (applies to all trees in the forest)
            base_params['ccp_alpha'] = optimal_alpha
            
            # Store pruning info for later display
            if not hasattr(st.session_state, 'pruning_info'):
                st.session_state.pruning_info = {}
            st.session_state.pruning_info['random_forest'] = pruning_info
        
        # Create and fit the model
        model = model_class(**base_params)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    model.fit(X, y)
    return model

def calculate_classification_metrics(X, y, model, method='Logistic Regression'):
    """Calculate comprehensive classification metrics"""
    y_pred = model.predict(X)
    
    # Handle both binary and multiclass classification for predict_proba
    if hasattr(model, 'predict_proba'):
        y_pred_proba_full = model.predict_proba(X)
        # For binary classification, use positive class probability
        if y_pred_proba_full.shape[1] == 2:
            y_pred_proba = y_pred_proba_full[:, 1]
        else:
            # For multiclass, use max probability across classes
            y_pred_proba = np.max(y_pred_proba_full, axis=1)
    else:
        y_pred_proba = None
    
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='binary', zero_division=0)
    recall = recall_score(y, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y, y_pred, average='binary', zero_division=0)
    
    # ROC AUC if probabilities are available
    roc_auc = roc_auc_score(y, y_pred_proba) if y_pred_proba is not None else None
    
    # Calculate residuals for classification (difference between actual and predicted probabilities)
    residuals = y - y_pred_proba if y_pred_proba is not None else y - y_pred
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'fitted_values': y_pred_proba if y_pred_proba is not None else y_pred,  # Add fitted values for plotting
        'residuals': residuals,  # Add residuals for plotting
        'n_obs': len(y),
        'method': method
    }

def optimize_regularization_parameters(X, y, method, fit_intercept=True, cv_folds=5):
    """Use nested cross-validation to find optimal regularization parameters"""
    
    # Define parameter grids
    if method == 'Lasso':
        param_grid = {'alpha': [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]}
        model = Lasso(fit_intercept=fit_intercept, random_state=42, max_iter=2000)
    elif method == 'Ridge':
        param_grid = {'alpha': [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]}
        model = Ridge(fit_intercept=fit_intercept, random_state=42)
    elif method == 'Elastic Net':
        param_grid = {
            'alpha': [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0],
            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
        }
        model = ElasticNet(fit_intercept=fit_intercept, random_state=42, max_iter=2000)
    else:
        raise ValueError(f"Nested CV not supported for method: {method}")
    
    # Use GridSearchCV with cross-validation
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        return_train_score=True
    )
    
    grid_search.fit(X, y)
    
    return {
        'best_params': grid_search.best_params_,
        'best_score': -grid_search.best_score_,  # Convert back to positive MSE
        'best_model': grid_search.best_estimator_,
        'cv_results': grid_search.cv_results_,
        'param_grid': param_grid
    }

def main():
    # Initialize usage tracking (must be called early)
    usage_data = track_app_usage()
    
    # Initialize session state for model run counter (needed for tracking even when hidden)
    if 'models_run_count' not in st.session_state:
        st.session_state.models_run_count = 0
    
    # Check for owner access to analytics
    show_analytics_option = False
    if "show_analytics" not in st.session_state:
        st.session_state.show_analytics = False

    # Navigation options
    nav_options = ["📊 Main App"]
    if show_analytics_option or st.session_state.show_analytics:
        nav_options.append("📈 Usage Analytics (Owner)")

    page = st.sidebar.selectbox(
        "Navigation",
        nav_options,
        help="Select Main App for normal use" + (" or Usage Analytics to view app usage statistics" if show_analytics_option else "")
    )
    
    if page == "📈 Usage Analytics (Owner)" and st.session_state.show_analytics:
        # Display usage analytics dashboard
        display_usage_analytics()
        return
    
    # Main header
    st.markdown('<h1 class="main-header">📊 Supervised Learning Tool: Regression and Classification</h1>', unsafe_allow_html=True)
    
    # About section first
    st.markdown("**About:** This webapp is created by Ren Zhang. Visit my [personal webpage](https://renzhang.weebly.com/) for more information. Please leave your feedback below:")
    
    # Feedback system with Google Sheets integration
    with st.expander("💬 Leave Feedback", expanded=False):
        feedback_text = st.text_area(
            "Your feedback helps improve this tool:",
            placeholder="Share your thoughts, suggestions, or report any issues...",
            height=100
        )
        if st.button("📤 Submit Feedback"):
            if feedback_text.strip():
                success = send_feedback_email(feedback_text)
                if success:
                    st.success("✅ Thank you for your feedback! It has been submitted.")
                else:
                    st.warning("⚠️ Daily feedback limit reached. Please try again tomorrow.")
            else:
                st.warning("⚠️ Please enter some feedback before submitting.")
    
    st.markdown("---")
    
    # Concise description
    st.markdown("""
    **Upload CSV/Excel data and perform advanced regression and classification analysis with multiple machine learning models.**
    
    **Available Models:** OLS, Logistic Regression, Lasso, Ridge, Elastic Net, Decision Trees, Random Forest
    
    **Key Features:** Multi-column data filtering • Interactive variable selection • Missing data handling • Nested cross-validation for parameter optimization • Comprehensive statistics & visualizations • Classification metrics & ROC curves
    
    **Perfect for:** Econometric analysis, predictive modeling, educational purposes, and exploratory data analysis.
    """)
    st.markdown("---")
    
    # Initialize default values for variables used in main area
    plot_var = 'None'
    plot_type = None
    
    # Sidebar for file upload and variable selection
    st.sidebar.header("📁 Data Upload & Variable Selection")
    
    # File upload with support for CSV and Excel
    uploaded_file = st.sidebar.file_uploader(
        "Choose a CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        help="Upload a CSV or Excel file containing your econometric data"
    )
    
    if uploaded_file is not None:
        # Track file upload
        track_feature_usage("file_uploads")
        
        try:
            # Determine file type and read accordingly
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                df = pd.read_csv(uploaded_file)
                sheet_name = None
            else:  # Excel file
                # Read Excel file and get sheet names
                excel_file = pd.ExcelFile(uploaded_file)
                sheet_names = excel_file.sheet_names
                
                # Let user choose sheet
                if len(sheet_names) > 1:
                    sheet_name = st.sidebar.selectbox(
                        "📋 Select Excel Sheet",
                        sheet_names,
                        help="Choose which sheet to analyze"
                    )
                else:
                    sheet_name = sheet_names[0]
                
                df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
            
            # Display basic information about the dataset
            st.sidebar.success(f"✅ File uploaded successfully!")
            if sheet_name:
                st.sidebar.info(f"Sheet: {sheet_name}")
            st.sidebar.info(f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
            
            # Initialize df_filtered as a copy of df at the start
            df_filtered = df.copy()
            
            # ===== SAMPLE SELECTION SECTION (MOVED TO TOP) =====
            st.sidebar.markdown("---")
            st.sidebar.header("🎯 Sample Selection")
            
            # Option to filter observations
            use_sample_filter = st.sidebar.checkbox(
                "Filter Sample Observations",
                help="Check this to select specific observations for analysis"
            )
            
            if use_sample_filter:
                # Filter method selection
                filter_method = st.sidebar.radio(
                    "Filter Method:",
                    ["Row Range", "Column Values"],
                    help="Choose how to filter your data"
                )
                
                if filter_method == "Row Range":
                    # Row range selection
                    start_row = st.sidebar.number_input(
                        "Start Row",
                        min_value=1,
                        max_value=len(df),
                        value=1,
                        help="First row to include (1-indexed)"
                    )
                    
                    end_row = st.sidebar.number_input(
                        "End Row",
                        min_value=start_row,
                        max_value=len(df),
                        value=len(df),
                        help="Last row to include"
                    )
                    
                    # Apply row filter
                    df_filtered = df.iloc[start_row-1:end_row].copy()
                
                elif filter_method == "Column Values":
                    # Allow multiple column filtering
                    st.sidebar.markdown("**🎯 Multiple Column Filtering**")
                    
                    # Add/Remove filter option
                    if 'active_filters' not in st.session_state:
                        st.session_state.active_filters = []
                    
                    # Interface to add new filters
                    st.sidebar.markdown("**Add New Filter:**")
                    new_filter_column = st.sidebar.selectbox(
                        "Select Column:",
                        ['None'] + df.columns.tolist(),
                        help="Choose a column to add a new filter"
                    )
                    
                    if new_filter_column != 'None' and st.sidebar.button("➕ Add Filter"):
                        if new_filter_column not in [f['column'] for f in st.session_state.active_filters]:
                            st.session_state.active_filters.append({
                                'column': new_filter_column,
                                'type': None,
                                'values': None
                            })
                    
                    # Show active filters and allow configuration
                    if st.session_state.active_filters:
                        st.sidebar.markdown("**Active Filters:**")
                        filters_to_remove = []
                        
                        for i, filter_config in enumerate(st.session_state.active_filters):
                            filter_column = filter_config['column']
                            column_series = df[filter_column]
                            
                            # Create expander for each filter
                            with st.sidebar.expander(f"🔧 {filter_column}", expanded=True):
                                
                                # Remove filter button
                                if st.button(f"❌ Remove", key=f"remove_filter_{i}"):
                                    filters_to_remove.append(i)
                                    continue
                                
                                # Detect column type and apply appropriate filtering
                                is_date_column = False
                                # First check if it's a numeric dtype - if so, not a date
                                if pd.api.types.is_numeric_dtype(column_series):
                                    is_date_column = False
                                else:
                                    # Only try date conversion for non-numeric columns
                                    try:
                                        # Sample a few values and check if they look like dates
                                        sample_data = column_series.dropna().head(10)
                                        if len(sample_data) > 0:
                                            pd.to_datetime(sample_data, errors='raise')
                                            # Additional check: make sure it's not just numbers
                                            if sample_data.dtype == 'object' or 'date' in str(sample_data.dtype).lower():
                                                is_date_column = True
                                    except:
                                        is_date_column = False
                                
                                if is_date_column:
                                    # DATE FILTERING
                                    date_series = pd.to_datetime(column_series, errors='coerce')
                                    min_date = date_series.min().date()
                                    max_date = date_series.max().date()
                                    
                                    st.write(f"📅 Date range: {min_date} to {max_date}")
                                    
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        start_date = st.date_input(
                                            "From",
                                            value=min_date,
                                            min_value=min_date,
                                            max_value=max_date,
                                            key=f"start_date_{i}"
                                        )
                                    with col2:
                                        end_date = st.date_input(
                                            "To", 
                                            value=max_date,
                                            min_value=min_date,
                                            max_value=max_date,
                                            key=f"end_date_{i}"
                                        )
                                    
                                    filter_config['type'] = 'date'
                                    filter_config['values'] = (start_date, end_date)
                                
                                elif column_series.dtype in ['object', 'category']:
                                    # CATEGORICAL FILTERING
                                    unique_values = column_series.unique()
                                    unique_values = [v for v in unique_values if pd.notna(v)]
                                    
                                    selected_values = st.multiselect(
                                        "Select values:",
                                        unique_values,
                                        default=unique_values[:3] if len(unique_values) <= 5 else unique_values[:2],
                                        key=f"cat_values_{i}"
                                    )
                                    
                                    filter_config['type'] = 'categorical'
                                    filter_config['values'] = selected_values
                                
                                else:
                                    # NUMERICAL FILTERING
                                    min_val = float(column_series.min())
                                    max_val = float(column_series.max())
                                    
                                    st.write(f"🔢 Range: {min_val:.2f} to {max_val:.2f}")
                                    
                                    input_method = st.radio(
                                        "Input method:",
                                        ["Slider", "Manual"],
                                        key=f"input_method_{i}"
                                    )
                                    
                                    if input_method == "Slider":
                                        value_range = st.slider(
                                            "Range",
                                            min_value=min_val,
                                            max_value=max_val,
                                            value=(min_val, max_val),
                                            key=f"slider_{i}"
                                        )
                                    else:
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            range_min = st.number_input(
                                                "Min",
                                                value=min_val,
                                                key=f"min_{i}"
                                            )
                                        with col2:
                                            range_max = st.number_input(
                                                "Max", 
                                                value=max_val,
                                                key=f"max_{i}"
                                            )
                                        value_range = (range_min, range_max)
                                    
                                    filter_config['type'] = 'numerical'
                                    filter_config['values'] = value_range
                        
                        # Remove filters marked for removal
                        for idx in sorted(filters_to_remove, reverse=True):
                            st.session_state.active_filters.pop(idx)
                        
                        # Apply all filters
                        df_filtered = df.copy()
                        for filter_config in st.session_state.active_filters:
                            if filter_config['values'] is not None:
                                column = filter_config['column']
                                
                                if filter_config['type'] == 'date':
                                    start_date, end_date = filter_config['values']
                                    if start_date <= end_date:
                                        date_series = pd.to_datetime(df_filtered[column], errors='coerce')
                                        mask = (date_series.dt.date >= start_date) & (date_series.dt.date <= end_date)
                                        df_filtered = df_filtered[mask]
                                
                                elif filter_config['type'] == 'categorical':
                                    selected_values = filter_config['values']
                                    if selected_values:
                                        df_filtered = df_filtered[df_filtered[column].isin(selected_values)]
                                
                                elif filter_config['type'] == 'numerical':
                                    min_val, max_val = filter_config['values']
                                    if min_val <= max_val:
                                        df_filtered = df_filtered[
                                            (df_filtered[column] >= min_val) & 
                                            (df_filtered[column] <= max_val)
                                        ]
                        
                        st.sidebar.info(f"📊 Active filters: {len(st.session_state.active_filters)}")
                    else:
                        df_filtered = df.copy()
                
                st.sidebar.info(f"Filtered sample: {len(df_filtered)} rows")
            
            # Missing value handling
            st.sidebar.markdown("---")
            st.sidebar.header("🔧 Missing Value Handling")
            
            missing_method = st.sidebar.selectbox(
                "How to handle missing values?",
                ["Listwise Deletion", "Mean Imputation", "Median Imputation", "Mode Imputation", "KNN Imputation"],
                help="Choose how to handle missing data"
            )
            
            # ===== ENHANCED DATA VISUALIZATION SECTION (AFTER FILTERING) =====
            st.markdown('<h2 class="subheader">📈 Data Visualization</h2>', unsafe_allow_html=True)
            st.write("Explore your data with various visualization options:")
            
            # Get current data for plotting (make sure it's always filtered data)
            current_df = df_filtered.copy()
            numeric_cols = current_df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Show info about current dataset being plotted
            if len(current_df) != len(df):
                st.info(f"📊 **Plotting filtered data**: {len(current_df)} rows (out of {len(df)} total rows)")
            else:
                st.info(f"📊 **Plotting full dataset**: {len(current_df)} rows")
            
            if len(numeric_cols) >= 1:
                # Enhanced controls in main area
                col_plot_type, col_var1, col_var2, col_standardize = st.columns([1, 1, 1, 1])
                
                with col_plot_type:
                    plot_type = st.selectbox(
                        "📊 Plot Type:",
                        ["None", "Histogram", "Box Plot", "Line Plot (Multiple)", "Scatter Plot", "Correlation Heatmap"],
                        help="Choose visualization type"
                    )
                
                with col_standardize:
                    standardize_data = st.checkbox(
                        "📐 Standardize Data",
                        value=False,
                        help="Standardize data: (x - mean) / std_dev"
                    )
                
                with col_var1:
                    if plot_type in ["Histogram", "Box Plot"]:
                        plot_var = st.selectbox(
                            "📋 Variable:",
                            ['None'] + numeric_cols,
                            help="Select a numeric variable"
                        )
                    elif plot_type == "Line Plot (Multiple)":
                        plot_vars = st.multiselect(
                            "📋 Y Variables:",
                            numeric_cols,
                            help="Select multiple Y variables for line plot"
                        )
                        plot_var = None
                    elif plot_type == "Scatter Plot":
                        plot_var = st.selectbox(
                            "📋 X Variable:",
                            ['None'] + numeric_cols,
                            help="Select X-axis variable"
                        )
                    elif plot_type == "Correlation Heatmap":
                        plot_vars = st.multiselect(
                            "📋 Variables (Multiple):",
                            numeric_cols,
                            default=numeric_cols[:5] if len(numeric_cols) >= 2 else numeric_cols,
                            help="Select variables for correlation analysis"
                        )
                        plot_var = None
                    else:
                        plot_var = None
                
                with col_var2:
                    if plot_type == "Scatter Plot":
                        plot_var2 = st.selectbox(
                            "📋 Y Variable:",
                            ['None'] + [col for col in numeric_cols if col != plot_var],
                            help="Select Y-axis variable"
                        )
                    elif plot_type == "Line Plot (Multiple)":
                        # Get all columns (numeric and non-numeric) for X-axis
                        all_cols = current_df.columns.tolist()
                        x_axis_var = st.selectbox(
                            "📋 X-axis:",
                            ["Index"] + all_cols,
                            help="Select X-axis variable (Index = row numbers)"
                        )
                    else:
                        st.write("")  # Empty space
                
                # Display plots based on selection
                if plot_type != "None":
                    st.markdown("---")
                    
                    # Prepare data (standardized or original)
                    plot_df = current_df.copy()
                    data_suffix = ""
                    
                    if standardize_data:
                        # Standardize numeric columns: (x - mean) / std
                        for col in numeric_cols:
                            if current_df[col].std() != 0:  # Avoid division by zero
                                plot_df[col] = (current_df[col] - current_df[col].mean()) / current_df[col].std()
                            else:
                                plot_df[col] = current_df[col] - current_df[col].mean()  # Just center if std=0
                        data_suffix = " (Standardized)"
                        st.info("📐 **Data is standardized**: (value - mean) / standard_deviation")
                    
                    if plot_type == "Histogram" and plot_var != 'None':
                        st.markdown(f"**📊 Distribution of {plot_var}{data_suffix}**")
                        fig = px.histogram(plot_df, x=plot_var, title=f"Distribution of {plot_var}{data_suffix}", nbins=30)
                        x_title = f"{plot_var}{' (Standardized)' if standardize_data else ''}"
                        fig.update_layout(xaxis_title=x_title, yaxis_title="Frequency")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Stats (show both original and standardized if applicable)
                        col1, col2, col3, col4 = st.columns(4)
                        if standardize_data:
                            with col1: st.metric("Mean (Std)", f"{plot_df[plot_var].mean():.4f}")
                            with col2: st.metric("Std Dev (Std)", f"{plot_df[plot_var].std():.4f}")
                            with col3: st.metric("Original Mean", f"{current_df[plot_var].mean():.2f}")
                            with col4: st.metric("Original Std", f"{current_df[plot_var].std():.2f}")
                        else:
                            with col1: st.metric("Mean", f"{plot_df[plot_var].mean():.2f}")
                            with col2: st.metric("Median", f"{plot_df[plot_var].median():.2f}")
                            with col3: st.metric("Std Dev", f"{plot_df[plot_var].std():.2f}")
                            with col4: st.metric("Range", f"{plot_df[plot_var].max() - plot_df[plot_var].min():.2f}")
                    
                    elif plot_type == "Box Plot" and plot_var != 'None':
                        st.markdown(f"**📊 Box Plot of {plot_var}{data_suffix}**")
                        y_title = f"{plot_var}{' (Standardized)' if standardize_data else ''}"
                        fig = px.box(plot_df, y=plot_var, title=f"Box Plot of {plot_var}{data_suffix}")
                        fig.update_layout(yaxis_title=y_title)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Outlier info (use plot_df for standardized data)
                        Q1 = plot_df[plot_var].quantile(0.25)
                        Q3 = plot_df[plot_var].quantile(0.75)
                        IQR = Q3 - Q1
                        outliers = plot_df[(plot_df[plot_var] < Q1 - 1.5*IQR) | (plot_df[plot_var] > Q3 + 1.5*IQR)]
                        
                        col1, col2, col3, col4 = st.columns(4)
                        if standardize_data:
                            with col1: st.metric("Q1 (Std)", f"{Q1:.4f}")
                            with col2: st.metric("Q3 (Std)", f"{Q3:.4f}")
                            with col3: st.metric("IQR (Std)", f"{IQR:.4f}")
                            with col4: st.metric("Outliers", len(outliers))
                        else:
                            with col1: st.metric("Q1", f"{Q1:.2f}")
                            with col2: st.metric("Q3", f"{Q3:.2f}")
                            with col3: st.metric("IQR", f"{IQR:.2f}")
                            with col4: st.metric("Outliers", len(outliers))
                    
                    elif plot_type == "Line Plot (Multiple)" and 'plot_vars' in locals() and len(plot_vars) > 0:
                        x_axis_title = "Observation Index" if x_axis_var == "Index" else x_axis_var
                        st.markdown(f"**📊 Line Plot: {', '.join(plot_vars)} vs {x_axis_title}{data_suffix}**")
                        
                        # Create line plot with multiple series
                        fig = px.line(title=f"Multiple Variables Line Plot{data_suffix}")
                        
                        # Determine X-axis data
                        if x_axis_var == "Index":
                            x_data = plot_df.index
                            x_label = "Observation Index"
                        else:
                            x_data = plot_df[x_axis_var]
                            x_label = f"{x_axis_var}"
                        
                        # Add each Y variable as a separate line
                        for var in plot_vars:
                            fig.add_scatter(x=x_data, y=plot_df[var], mode='lines', name=var)
                        
                        y_title = f"Values{' (Standardized)' if standardize_data else ''}"
                        fig.update_layout(xaxis_title=x_label, yaxis_title=y_title)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show selected variables info
                        vars_info = f"📊 X-axis: {x_axis_title} | Y-variables: {', '.join(plot_vars)}"
                        if standardize_data:
                            vars_info += " (Y-variables standardized to mean=0, std=1)"
                        st.info(vars_info)
                    
                    elif plot_type == "Scatter Plot" and plot_var != 'None' and 'plot_var2' in locals() and plot_var2 != 'None':
                        st.markdown(f"**📊 Scatter Plot: {plot_var} vs {plot_var2}{data_suffix}**")
                        
                        x_title = f"{plot_var}{' (Standardized)' if standardize_data else ''}"
                        y_title = f"{plot_var2}{' (Standardized)' if standardize_data else ''}"
                        
                        fig = px.scatter(plot_df, x=plot_var, y=plot_var2, 
                                       title=f"Scatter Plot: {plot_var} vs {plot_var2}{data_suffix}")
                        fig.update_layout(xaxis_title=x_title, yaxis_title=y_title)
                        
                        # Add correlation info (use plot_df for consistency)
                        correlation = plot_df[plot_var].corr(plot_df[plot_var2])
                        fig.update_layout(
                            annotations=[dict(x=0.02, y=0.98, xref="paper", yref="paper",
                                            text=f"Correlation: {correlation:.3f}", showarrow=False,
                                            bgcolor="white", bordercolor="black")]
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        col1, col2 = st.columns(2)
                        with col1: st.metric("Correlation", f"{correlation:.4f}")
                        with col2: st.metric("R²", f"{correlation**2:.4f}")
                        
                        if standardize_data:
                            st.info("📐 **Note**: Correlation remains the same for standardized data")
                    
                    elif plot_type == "Correlation Heatmap" and 'plot_vars' in locals() and len(plot_vars) >= 2:
                        st.markdown(f"**📊 Correlation Matrix of {len(plot_vars)} Variables{data_suffix}**")
                        
                        # Calculate correlation matrix (using plot_df for consistency, though correlation is same)
                        corr_matrix = plot_df[plot_vars].corr()
                        
                        # Create heatmap
                        title_text = f"Correlation Matrix Heatmap{data_suffix}"
                        fig = px.imshow(corr_matrix, 
                                      text_auto=True, 
                                      aspect="auto",
                                      title=title_text,
                                      color_continuous_scale="RdBu_r",
                                      zmin=-1, zmax=1)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show correlation matrix as table
                        st.write("**Correlation Matrix Values:**")
                        st.dataframe(corr_matrix.round(4), use_container_width=True)
                        
                        if standardize_data:
                            st.info("📐 **Note**: Correlation values are identical for standardized data")
                    
                    # Help messages for incomplete selections
                    elif plot_type in ["Histogram", "Box Plot"] and plot_var == 'None':
                        st.info("👆 Please select a variable to display the plot.")
                    elif plot_type == "Line Plot (Multiple)" and ('plot_vars' not in locals() or len(plot_vars) == 0):
                        st.info("👆 Please select at least one Y variable for the line plot.")
                    elif plot_type == "Scatter Plot" and (plot_var == 'None' or 'plot_var2' not in locals() or plot_var2 == 'None'):
                        st.info("👆 Please select both X and Y variables for the scatter plot.")
                    elif plot_type == "Correlation Heatmap" and ('plot_vars' not in locals() or len(plot_vars) < 2):
                        st.info("👆 Please select at least 2 variables for correlation analysis.")
            else:
                st.warning("⚠️ No numeric columns found in the dataset for visualization.")
            
            # Main content area
            st.markdown("---")
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown('<h2 class="subheader">📋 Dataset Overview</h2>', unsafe_allow_html=True)
                
                # Display first few rows of current data (original or filtered)
                if 'df_filtered' in locals():
                    st.write("**First 5 rows of your data:**")
                    st.dataframe(df_filtered.head(), use_container_width=True)
                    
                    # Display basic statistics for current data
                    st.write("**Descriptive Statistics:**")
                    st.dataframe(df_filtered.describe(), use_container_width=True)
                else:
                    st.write("**First 5 rows of your data:**")
                    st.dataframe(df.head(), use_container_width=True)
                    
                    # Display basic statistics
                    st.write("**Descriptive Statistics:**")
                    st.dataframe(df.describe(), use_container_width=True)
            
            with col2:
                st.markdown('<h2 class="subheader">📊 Data Series Information</h2>', unsafe_allow_html=True)
                
                # Display column information for current data
                current_df = df_filtered if 'df_filtered' in locals() else df
                col_info = pd.DataFrame({
                    'Column': current_df.columns,
                    'Data Type': current_df.dtypes.astype(str),  # Convert to string to avoid Arrow conversion issues
                    'Non-Null Count': current_df.count(),
                    'Null Count': current_df.isnull().sum(),
                    'Unique Values': current_df.nunique()
                })
                st.dataframe(col_info, use_container_width=True)
            
            # Get numeric columns only from the current dataset
            current_df = df_filtered if 'df_filtered' in locals() else df
            # Include both numeric and boolean columns (boolean columns are useful for regression)
            numeric_columns = current_df.select_dtypes(include=[np.number, bool]).columns.tolist()
            
            if len(numeric_columns) < 2:
                st.error("❌ Please upload a dataset with at least 2 numeric columns for regression analysis.")
                return
            
            # Update numeric columns for the final filtered data (include boolean)
            numeric_columns = df_filtered.select_dtypes(include=[np.number, bool]).columns.tolist()
            
            # Variable selection in sidebar
            st.sidebar.markdown("---")
            st.sidebar.header("🎯 Regression Setup")
            
            # Dependent variable selection
            dependent_var = st.sidebar.selectbox(
                "Select Dependent Variable (Y)",
                numeric_columns,
                help="Choose the variable you want to predict/explain"
            )
            
            # Independent variables selection with enhanced features
            available_independent = [col for col in numeric_columns if col != dependent_var]
            
            st.sidebar.markdown("**Independent Variables (X):**")
            
            # Initialize session state for selected variables
            if 'selected_independent' not in st.session_state:
                st.session_state.selected_independent = []
            
            # Search and Add Interface
            st.sidebar.markdown("*Add Variables:*")
            search_term = st.sidebar.text_input(
                "🔍 Search & Add Variables",
                placeholder="Type to filter variable names...",
                help="Search for variables by name, then click Add"
            )
            
            # Filter variables based on search
            if search_term:
                filtered_vars = [var for var in available_independent if search_term.lower() in var.lower()]
                filtered_vars = [var for var in filtered_vars if var not in st.session_state.selected_independent]
            else:
                filtered_vars = [var for var in available_independent if var not in st.session_state.selected_independent]
            
            # Add variable interface
            if filtered_vars:
                col1, col2 = st.sidebar.columns([3, 1])
                with col1:
                    var_to_add = st.selectbox(
                        "Select variable to add:",
                        [""] + filtered_vars,
                        key="var_to_add"
                    )
                with col2:
                    st.write("")  # spacing
                    if st.button("➕ Add", disabled=not var_to_add):
                        if var_to_add and var_to_add not in st.session_state.selected_independent:
                            st.session_state.selected_independent.append(var_to_add)
                            st.rerun()
            
            # Quick selection buttons
            col1, col2 = st.sidebar.columns(2)
            with col1:
                if st.button("✅ Add All", help="Add all available variables"):
                    st.session_state.selected_independent = list(set(st.session_state.selected_independent + available_independent))
                    st.rerun()
            with col2:
                if st.button("❌ Clear All", help="Remove all variables"):
                    st.session_state.selected_independent = []
                    st.rerun()
            
            # Selected Variables Display and Removal Interface
            st.sidebar.markdown("*Selected Variables:*")
            if st.session_state.selected_independent:
                st.sidebar.success(f"✅ {len(st.session_state.selected_independent)} variables selected")
                
                # Display selected variables with remove buttons
                for i, var in enumerate(st.session_state.selected_independent):
                    col1, col2 = st.sidebar.columns([4, 1])
                    with col1:
                        st.write(f"📊 {var}")
                    with col2:
                        if st.button("❌", key=f"remove_var_{i}", help=f"Remove {var}"):
                            st.session_state.selected_independent.remove(var)
                            st.rerun()
            else:
                st.sidebar.info("No variables selected yet")
            
            # Set independent_vars for compatibility with rest of code
            independent_vars = st.session_state.selected_independent
            
            # Constant term option
            st.sidebar.markdown("**Intercept/Constant:**")
            include_constant = st.sidebar.checkbox(
                "Include Constant Term",
                value=True,
                help="Include an intercept (constant) in the regression"
            )
            
            # Estimation method selection
            st.sidebar.markdown("---")
            st.sidebar.header("⚙️ Estimation Method")
            
            # Detect if dependent variable is categorical (for classification)
            unique_values = df_filtered[dependent_var].dropna().unique() if dependent_var else []
            n_unique = len(unique_values)
            is_binary = n_unique == 2
            is_categorical = n_unique <= 10 and all(isinstance(x, (int, float)) and x == int(x) for x in unique_values if pd.notna(x))
            
            # Method categories - always include all methods
            method_options = ["OLS", "Logistic Regression", "Decision Tree", "Random Forest", "Lasso", "Ridge", "Elastic Net"]
            
            if is_binary:
                st.sidebar.info("🎯 Binary dependent variable detected - Classification methods recommended")
            elif is_categorical and n_unique > 2:
                st.sidebar.warning(f"📊 Multi-class dependent variable detected ({n_unique} classes: {sorted(unique_values)}) - Classification methods recommended")
                st.sidebar.info("💡 For Decision Trees: Use Classification mode for proper probability estimates")
            
            estimation_method = st.sidebar.selectbox(
                "Choose Estimation Method",
                method_options,
                help="Select the regression/classification method to use"
            )
            
            # Method-specific parameters
            if estimation_method in ["Lasso", "Ridge", "Elastic Net"]:
                st.sidebar.markdown("**Regularization Parameters:**")
                
                # Option to use nested cross-validation
                use_nested_cv = st.sidebar.checkbox(
                    "🔄 Use Nested Cross-Validation",
                    value=False,
                    help="Automatically find optimal regularization parameters using cross-validation"
                )
                
                if use_nested_cv:
                    cv_folds = st.sidebar.selectbox(
                        "CV Folds",
                        [3, 5, 10],
                        index=1,
                        help="Number of cross-validation folds"
                    )
                    st.sidebar.info("Parameters will be optimized automatically")
                    alpha = 1.0  # Will be overridden by CV
                    l1_ratio = 0.5  # Will be overridden by CV for Elastic Net
                else:
                    # Manual parameter setting with flexible input options
                    st.sidebar.markdown("**Manual Parameter Setting:**")
                    
                    # Alpha parameter with both slider and number input
                    parameter_input_method = st.sidebar.radio(
                        "Parameter Input Method:",
                        options=["Slider (0.001-10)", "Number Input (Any Value)"],
                        index=0,
                        help="Choose how to set the regularization parameter"
                    )
                    
                    if parameter_input_method == "Slider (0.001-10)":
                        alpha = st.sidebar.slider(
                            "Regularization Strength (α)",
                            min_value=0.001,
                            max_value=10.0,
                            value=1.0,
                            step=0.001,
                            help="Higher values increase regularization"
                        )
                    else:
                        alpha = st.sidebar.number_input(
                            "Regularization Strength (α)",
                            min_value=0.0001,
                            max_value=10000.0,
                            value=1.0,
                            step=0.1,
                            format="%.4f",
                            help="Enter any positive value for regularization strength"
                        )
                    cv_folds = 5  # Default value
                
                if estimation_method == "Elastic Net" and not use_nested_cv:
                    l1_ratio = st.sidebar.slider(
                        "L1 Ratio (Lasso vs Ridge)",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.5,
                        step=0.01,
                        help="0 = Ridge, 1 = Lasso, 0.5 = Equal mix"
                    )
                elif estimation_method == "Elastic Net":
                    l1_ratio = 0.5  # Will be optimized by CV
                else:
                    l1_ratio = 0.5
            else:
                alpha = 1.0
                l1_ratio = 0.5
                use_nested_cv = False  # Default for non-regularized methods
            
            # Logistic Regression specific parameters
            if estimation_method == "Logistic Regression":
                st.sidebar.markdown("**Classification Parameters:**")
                
                # Feature scaling option
                use_scaling = st.sidebar.checkbox(
                    "🔧 Standardize Features",
                    value=False,
                    help="Standardize features (recommended for logistic regression)"
                )
                
                # Class balancing options
                class_weight_option = st.sidebar.selectbox(
                    "Class Weight Strategy:",
                    options=["None", "Balanced", "Custom"],
                    index=0,
                    help="Handle class imbalance. 'Balanced' automatically adjusts weights inversely proportional to class frequencies."
                )
                
                if class_weight_option == "Balanced":
                    class_weight = "balanced"
                elif class_weight_option == "Custom" and is_binary:
                    col1, col2 = st.sidebar.columns(2)
                    with col1:
                        class_0_weight = st.number_input("Class 0 Weight", min_value=0.1, value=1.0, step=0.1)
                    with col2:
                        class_1_weight = st.number_input("Class 1 Weight", min_value=0.1, value=1.0, step=0.1)
                    class_weight = {0: class_0_weight, 1: class_1_weight}
                else:
                    class_weight = None
                    
                # Stratified sampling for train-test split
                if is_binary or is_categorical:
                    use_stratify = st.sidebar.checkbox(
                        "📊 Use Stratified Sampling",
                        value=False,
                        help="Maintain class proportions in train-test splits"
                    )
                else:
                    use_stratify = False
            else:
                use_scaling = False
                class_weight = None
                use_stratify = False
            
            # Tree-based method parameters
            if estimation_method in ["Decision Tree", "Random Forest"]:
                st.sidebar.markdown("**Tree Parameters:**")
                
                # Maximum depth with flexible input
                use_max_depth = st.sidebar.checkbox(
                    "Limit Tree Depth",
                    value=True,
                    help="Limit the maximum depth of the tree (recommended for better visualization)"
                )
                
                if use_max_depth:
                    max_depth = st.sidebar.number_input(
                        "Maximum Depth",
                        min_value=1,
                        max_value=25,
                        value=5,
                        step=1,
                        help="Enter any integer between 1 and 25 for maximum tree depth"
                    )
                else:
                    max_depth = None
                    st.sidebar.info("⚠️ Unlimited depth may create very large trees")
                
                min_samples_split = st.sidebar.slider(
                    "Min Samples Split",
                    min_value=2,
                    max_value=20,
                    value=2,
                    help="Minimum samples required to split an internal node"
                )
                
                min_samples_leaf = st.sidebar.slider(
                    "Min Samples Leaf",
                    min_value=1,
                    max_value=10,
                    value=1,
                    help="Minimum samples required to be at a leaf node"
                )
                
                # Cost Complexity Pruning
                st.sidebar.markdown("**Pruning Parameters:**")
                enable_pruning = st.sidebar.checkbox(
                    "Enable Cost Complexity Pruning",
                    value=False,
                    help="Apply cost complexity pruning using cross-validation to find optimal alpha"
                )
                
                if enable_pruning:
                    cv_folds = st.sidebar.slider(
                        "Cross-Validation Folds",
                        min_value=3,
                        max_value=10,
                        value=5,
                        help="Number of folds for cross-validation to select optimal alpha"
                    )
                    
                    pruning_method = st.sidebar.radio(
                        "Pruning Selection Method:",
                        options=["Automatic (CV)", "Manual Alpha"],
                        index=0,
                        help="Choose automatic selection via cross-validation or manual alpha setting"
                    )
                    
                    if pruning_method == "Manual Alpha":
                        st.sidebar.markdown("**Cost Complexity Alpha (α)**")
                        
                        # Get automatic alpha for reference if possible
                        auto_max = None  # Initialize to avoid scope issues
                        try:
                            # Try to calculate automatic alpha for guidance
                            if 'X_train' in st.session_state and 'y_train' in st.session_state:
                                from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
                                temp_model = DecisionTreeRegressor(random_state=42)
                                temp_model.fit(st.session_state.X_train, st.session_state.y_train)
                                path = temp_model.cost_complexity_pruning_path(st.session_state.X_train, st.session_state.y_train)
                                ccp_alphas = path.ccp_alphas[:-1]  # Remove last (empty tree)
                                if len(ccp_alphas) > 0:
                                    # Get optimal alpha using same CV logic
                                    from econometric_app import find_optimal_ccp_alpha
                                    optimal_alpha, _ = find_optimal_ccp_alpha(
                                        st.session_state.X_train, st.session_state.y_train, 
                                        DecisionTreeRegressor, cv_folds, random_state=42
                                    )
                                    auto_min, auto_max = ccp_alphas.min(), ccp_alphas.max()
                                    st.sidebar.markdown(f"*💡 For this dataset, automatic method explores α range: {auto_min:.6f} to {auto_max:.6f}*")
                                    st.sidebar.markdown(f"*💡 Automatic optimal α: {optimal_alpha:.6f} - use similar magnitude*")
                                else:
                                    st.sidebar.markdown("*💡 Use α values in similar magnitude to automatic optimal values*")
                            else:
                                st.sidebar.markdown("*💡 Use α values in similar magnitude to automatic optimal values*")
                        except:
                            st.sidebar.markdown("*💡 Use α values in similar magnitude to automatic optimal values*")
                        
                        st.sidebar.markdown("*Higher values = more pruning (smaller trees)*")
                        # Use a text input so users can type any positive number (no spinner buttons)
                        alpha_text = st.sidebar.text_input(
                            "Enter alpha value (positive number):",
                            value="0.01",
                            help="Enter any positive numeric value. Leave blank to cancel manual alpha."
                        )

                        manual_alpha = None
                        if alpha_text is not None and alpha_text.strip() != "":
                            try:
                                parsed_alpha = float(alpha_text)
                                if parsed_alpha < 0:
                                    st.sidebar.error("Alpha must be a non-negative number.")
                                    manual_alpha = None
                                else:
                                    manual_alpha = parsed_alpha
                                    # If automatic alpha range was computed above, warn when manual alpha is much larger
                                    if auto_max is not None:
                                        try:
                                            if manual_alpha > max(auto_max * 5, auto_max + 1e-12):
                                                st.sidebar.warning(
                                                    f"⚠️ Manual α ({manual_alpha:.6g}) is much larger than the automatic range max ({auto_max:.6g}).\n"
                                                    "Very large values may over-prune the tree and cause the visualization to fail."
                                                )
                                            elif manual_alpha > auto_max:
                                                st.sidebar.info(
                                                    f"Note: Manual α ({manual_alpha:.6g}) is larger than the automatic range max ({auto_max:.6g}). This may produce a smaller tree."
                                                )
                                        except Exception:
                                            pass
                            except ValueError:
                                st.sidebar.error("Please enter a valid numeric value for alpha.")
                                manual_alpha = None
                    else:
                        manual_alpha = None
                else:
                    cv_folds = 5
                    pruning_method = "Automatic (CV)"
                    manual_alpha = None
                
                if estimation_method == "Random Forest":
                    n_estimators = st.sidebar.slider(
                        "Number of Trees",
                        min_value=10,
                        max_value=500,
                        value=100,
                        step=10,
                        help="Number of trees in the forest"
                    )
                else:
                    n_estimators = 100
                
                # Probability Display Settings for Binary Classification
                if is_binary:
                    st.sidebar.markdown("---")
                    st.sidebar.markdown("**🎯 Probability Display (Binary Classification):**")
                    prob_class_choice = st.sidebar.radio(
                        "Show probability for:",
                        options=["Class 0 (typically negative/false)", "Class 1 (typically positive/true)"],
                        index=0,
                        help="Choose which class probability to display on tree nodes for binary classification"
                    )
                    prob_class_index = 0 if "Class 0" in prob_class_choice else 1
                else:
                    prob_class_index = 0  # Default for non-binary variables
                
            else:
                max_depth = None
                min_samples_split = 2
                min_samples_leaf = 1
                n_estimators = 100
                prob_class_index = 0  # Default for non-tree methods
                enable_pruning = False  # Default for non-tree methods
                cv_folds = 5
                pruning_method = "Automatic (CV)"
                manual_alpha = None
            
            # Missing Values Summary for Selected Variables (show in main area after variables are selected)
            if dependent_var and independent_vars:
                st.markdown("---")
                st.markdown('<h2 class="subheader">🔍 Selected Variables Missing Values Summary</h2>', unsafe_allow_html=True)
                
                selected_vars = [dependent_var] + independent_vars
                current_df = df_filtered if 'df_filtered' in locals() else df
                
                missing_summary = []
                for var in selected_vars:
                    if var in current_df.columns:
                        missing_count = current_df[var].isnull().sum()
                        total_count = len(current_df)
                        missing_pct = (missing_count / total_count) * 100 if total_count > 0 else 0
                        
                        missing_summary.append({
                            'Variable': var,
                            'Role': 'Dependent (Y)' if var == dependent_var else 'Independent (X)',
                            'Missing Count': missing_count,
                            'Missing %': f"{missing_pct:.1f}%",
                            'Available': total_count - missing_count
                        })
                
                if missing_summary:
                    missing_df = pd.DataFrame(missing_summary)
                    st.dataframe(missing_df, use_container_width=True)
                    
                    # Summary alert
                    total_missing = sum([row['Missing Count'] for row in missing_summary])
                    if total_missing > 0:
                        st.warning(f"⚠️ Total missing values in selected variables: {total_missing}")
                    else:
                        st.success("✅ No missing values in selected variables")
            
            if independent_vars:
                # Run regression button
                if st.sidebar.button(f"🔬 Run {estimation_method} Regression", type="primary"):
                    
                    # Track model execution - increment counters
                    st.session_state.models_run_count += 1
                    
                    # Update persistent usage statistics
                    usage_file = "app_usage_stats.json"
                    try:
                        if os.path.exists(usage_file):
                            with open(usage_file, "r") as f:
                                usage_data = json.load(f)
                        else:
                            usage_data = {}
                        
                        # Increment total models run
                        usage_data["total_models_run"] = usage_data.get("total_models_run", 0) + 1
                        
                        # Save updated data
                        with open(usage_file, "w") as f:
                            json.dump(usage_data, f, indent=2)
                            
                    except Exception:
                        # If file operations fail, continue with analysis
                        pass
                    
                    # Track feature usage
                    track_feature_usage("model_runs")
                    track_feature_usage(f"model_{estimation_method.lower().replace(' ', '_')}")
                    
                    # Prepare data for regression - handle missing values
                    y_raw = df_filtered[dependent_var]
                    X_raw = df_filtered[independent_vars]
                    
                    # Convert boolean columns to numeric (True=1, False=0)
                    bool_cols = X_raw.select_dtypes(include=[bool]).columns
                    if len(bool_cols) > 0:
                        X_raw = X_raw.copy()
                        X_raw[bool_cols] = X_raw[bool_cols].astype(int)
                        st.info(f"**Data Processing**: Converted {len(bool_cols)} boolean columns to numeric (True=1, False=0)")
                    
                    # Show missing value summary before processing
                    total_obs = len(df_filtered)
                    missing_y = y_raw.isnull().sum()
                    missing_X = X_raw.isnull().sum().sum()
                    
                    if missing_y > 0 or missing_X > 0:
                        st.info(f"**Missing Value Summary**: {missing_y} missing in Y, {missing_X} missing in X variables")
                    
                    # Handle missing values based on selected method
                    if missing_method == "Listwise Deletion":
                        # Original method - keep only complete cases
                        y = y_raw.dropna()
                        X = X_raw.dropna()
                        common_index = y.index.intersection(X.index)
                        y = y.loc[common_index]
                        X = X.loc[common_index]
                        data_info = f"Complete case analysis: {len(y)} observations used out of {total_obs}"
                        
                    else:
                        # Imputation methods
                        # First align indices to work with same observations
                        combined_data = pd.concat([y_raw, X_raw], axis=1).dropna(how='all')
                        y_temp = combined_data[dependent_var]
                        X_temp = combined_data[independent_vars]
                        
                        # Apply imputation
                        if missing_method == "Mean Imputation":
                            imputer = SimpleImputer(strategy='mean')
                        elif missing_method == "Median Imputation":
                            imputer = SimpleImputer(strategy='median')
                        elif missing_method == "Mode Imputation":
                            imputer = SimpleImputer(strategy='most_frequent')
                        elif missing_method == "KNN Imputation":
                            imputer = KNNImputer(n_neighbors=5)
                        
                        # Impute Y variable if it has missing values
                        if y_temp.isnull().sum() > 0:
                            y_imputed = imputer.fit_transform(y_temp.values.reshape(-1, 1)).flatten()
                            y = pd.Series(y_imputed, index=y_temp.index)
                        else:
                            y = y_temp
                        
                        # Impute X variables if they have missing values
                        if X_temp.isnull().sum().sum() > 0:
                            X_imputed = imputer.fit_transform(X_temp)
                            X = pd.DataFrame(X_imputed, columns=X_temp.columns, index=X_temp.index)
                        else:
                            X = X_temp
                        
                        data_info = f"{missing_method}: {len(y)} observations used, missing values imputed"
                    
                    if len(y) < len(independent_vars) + 1:
                        st.error("❌ Insufficient data points for regression. Need more observations than variables.")
                        return
                    
                    # Display data processing info
                    st.info(f"📊 **Data Processing**: {data_info}")
                    
                    # Determine model type for tree/forest methods
                    model_type = 'classification' if estimation_method in ['Logistic Regression', 'Decision Tree', 'Random Forest'] and (is_binary or is_categorical) else 'regression'
                    
                    # Handle nested cross-validation for regularized methods
                    cv_results = None
                    if estimation_method in ["Lasso", "Ridge", "Elastic Net"] and use_nested_cv:
                        st.info("🔄 **Optimizing parameters using nested cross-validation...**")
                        
                        # Use the original or scaled data for parameter optimization
                        X_for_cv = X
                        if estimation_method in ["Lasso", "Ridge", "Elastic Net"]:
                            scaler = StandardScaler()
                            X_for_cv = pd.DataFrame(
                                scaler.fit_transform(X), 
                                columns=X.columns, 
                                index=X.index
                            )
                        
                        cv_results = optimize_regularization_parameters(X_for_cv, y, estimation_method, include_constant, cv_folds)
                        
                        # Extract optimized parameters
                        if 'alpha' in cv_results['best_params']:
                            alpha = cv_results['best_params']['alpha']
                        if 'l1_ratio' in cv_results['best_params']:
                            l1_ratio = cv_results['best_params']['l1_ratio']
                        
                        st.success(f"✅ **Optimal parameters found**: {cv_results['best_params']}")
                        st.info(f"📊 **Cross-validation MSE**: {cv_results['best_score']:.6f}")
                    
                    # For regularized methods or when use_scaling is enabled, standardize features
                    if estimation_method in ["Lasso", "Ridge", "Elastic Net"] or (estimation_method == "Logistic Regression" and use_scaling):
                        scaler = StandardScaler()
                        X_scaled = pd.DataFrame(
                            scaler.fit_transform(X), 
                            columns=X.columns, 
                            index=X.index
                        )
                        # Fit the model on scaled data
                        model = fit_model(X_scaled, y, estimation_method, alpha, l1_ratio, include_constant,
                                        model_type=model_type, max_depth=max_depth, 
                                        min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                        n_estimators=n_estimators, enable_pruning=enable_pruning,
                                        cv_folds=cv_folds, pruning_method=pruning_method, manual_alpha=manual_alpha,
                                        class_weight=class_weight)
                        
                        # Calculate stats on scaled data
                        if model_type == 'classification':
                            stats_dict = calculate_classification_metrics(X_scaled, y, model, estimation_method)
                        else:
                            stats_dict = calculate_regression_stats(X_scaled, y, model, estimation_method, include_constant)
                        X_for_plotting = X_scaled  # Use scaled data for plotting
                    else:
                        # Fit the model on original data
                        model = fit_model(X, y, estimation_method, alpha, l1_ratio, include_constant,
                                        model_type=model_type, max_depth=max_depth,
                                        min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                        n_estimators=n_estimators, enable_pruning=enable_pruning,
                                        cv_folds=cv_folds, pruning_method=pruning_method, manual_alpha=manual_alpha,
                                        class_weight=class_weight)
                        # Calculate stats on original data
                        if model_type == 'classification':
                            stats_dict = calculate_classification_metrics(X, y, model, estimation_method)
                        else:
                            stats_dict = calculate_regression_stats(X, y, model, estimation_method, include_constant)
                        X_for_plotting = X  # Use original data for plotting
                    
                    # Display results
                    if model_type == 'classification':
                        st.markdown(f'<h2 class="subheader">📈 {estimation_method} Classification Results</h2>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<h2 class="subheader">📈 {estimation_method} Regression Results</h2>', unsafe_allow_html=True)
                    
                    # Method-specific information
                    if estimation_method != "OLS":
                        method_info = {
                            "Lasso": "L1 regularization - promotes sparsity by setting some coefficients to zero",
                            "Ridge": "L2 regularization - shrinks coefficients towards zero but keeps all variables",
                            "Elastic Net": "Combines L1 and L2 regularization for balanced variable selection and shrinkage",
                            "Logistic Regression": "Models probability of binary outcomes using logistic function",
                            "Decision Tree": "Creates decision rules through recursive partitioning of feature space",
                            "Random Forest": "Ensemble of decision trees with voting/averaging for robust predictions"
                        }
                        if estimation_method in method_info:
                            st.info(f"**{estimation_method}**: {method_info[estimation_method]}")
                        
                        if estimation_method in ["Lasso", "Ridge", "Elastic Net"]:
                            if use_nested_cv and cv_results:
                                st.info(f"**Cross-Validation Results**: Optimal α = {alpha:.4f}" + 
                                       (f", L1 ratio = {l1_ratio:.2f}" if estimation_method == "Elastic Net" else "") +
                                       f" (CV MSE: {cv_results['best_score']:.6f})")
                            else:
                                st.info(f"**Regularization Parameters**: α = {alpha}" + 
                                       (f", L1 ratio = {l1_ratio}" if estimation_method == "Elastic Net" else ""))
                        elif estimation_method in ["Decision Tree", "Random Forest"]:
                            tree_params = f"**Tree Parameters**: Max Depth = {max_depth}, Min Split = {min_samples_split}, Min Leaf = {min_samples_leaf}"
                            if estimation_method == "Random Forest":
                                tree_params += f", Trees = {n_estimators}"
                            st.info(tree_params)
                    
                    # Model summary - different for classification vs regression
                    if model_type == 'classification':
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Accuracy", f"{float(stats_dict['accuracy']):.4f}")
                        with col2:
                            st.metric("Precision", f"{float(stats_dict['precision']):.4f}")
                        with col3:
                            st.metric("Recall", f"{float(stats_dict['recall']):.4f}")
                        with col4:
                            st.metric("F1-Score", f"{float(stats_dict['f1_score']):.4f}")
                        
                        if stats_dict['roc_auc'] is not None:
                            st.metric("ROC AUC", f"{float(stats_dict['roc_auc']):.4f}")
                        
                        # Classification-specific visualizations
                        st.markdown('<h2 class="subheader">📊 Classification Visualizations</h2>', unsafe_allow_html=True)
                        
                        # Create tabs for classification plots
                        class_tab1, class_tab2, class_tab3 = st.tabs(["Confusion Matrix", "Model Coefficients", "Actual vs Predicted"])
                        
                        with class_tab1:
                            # Confusion Matrix
                            y_pred = model.predict(X_for_plotting)
                            class_names = [str(c) for c in model.classes_] if hasattr(model, 'classes_') else None
                            confusion_fig = create_confusion_matrix_plot(y, y_pred, class_names)
                            st.plotly_chart(confusion_fig, use_container_width=True)
                            track_feature_usage("visualizations_created")
                            st.caption("Confusion matrix shows the number of correct and incorrect predictions for each class.")
                        
                        with class_tab2:
                            # Coefficients plot (only for logistic regression)
                            if estimation_method == "Logistic Regression":
                                coef_fig = create_coefficients_plot(model, independent_vars)
                                if coef_fig:
                                    st.plotly_chart(coef_fig, use_container_width=True)
                                    st.caption("Coefficient values show the impact of each feature. Positive values increase the probability of the positive class.")
                                else:
                                    st.info("Coefficient visualization not available for this model type.")
                            else:
                                st.info("Coefficient visualization is only available for Logistic Regression models.")
                        
                        with class_tab3:
                            # Actual vs Predicted with probabilities
                            y_pred_proba = None
                            if hasattr(model, 'predict_proba'):
                                y_pred_proba_all = model.predict_proba(X_for_plotting)
                                # For binary classification, use probability of positive class
                                if y_pred_proba_all.shape[1] == 2:
                                    y_pred_proba = y_pred_proba_all[:, 1]
                                else:
                                    # For multiclass, use max probability
                                    y_pred_proba = np.max(y_pred_proba_all, axis=1)
                            
                            actual_pred_fig = create_actual_vs_predicted_plot(y, y_pred, y_pred_proba)
                            st.plotly_chart(actual_pred_fig, use_container_width=True)
                            st.caption("Comparison of actual vs predicted classes. Green dots show prediction probabilities when available.")
                    
                    else:
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("R-squared", f"{float(stats_dict['r_squared']):.4f}")
                        with col2:
                            st.metric("Adj. R-squared", f"{float(stats_dict['adj_r_squared']):.4f}")
                        with col3:
                            st.metric("RMSE", f"{float(stats_dict['rmse']):.4f}")
                        with col4:
                            st.metric("Observations", int(stats_dict['n_obs']))
                    
                    # Model-specific results display
                    if estimation_method in ["Decision Tree", "Random Forest"]:
                        # Tree models don't have coefficients - show feature importance and tree structure
                        st.write("**Feature Importance:**")
                        
                        # Feature importance table
                        importance_data = []
                        for i, var_name in enumerate(independent_vars):
                            importance_data.append({
                                'Feature': var_name,
                                'Importance': model.feature_importances_[i],
                                'Importance %': f"{model.feature_importances_[i]*100:.2f}%"
                            })
                        
                        importance_df = pd.DataFrame(importance_data).sort_values('Importance', ascending=False)
                        st.dataframe(importance_df, use_container_width=True)
                        
                        # Tree visualization
                        st.markdown('<h2 class="subheader">🌳 Tree Structure</h2>', unsafe_allow_html=True)
                        
                        if estimation_method == "Decision Tree":
                            # Create interactive decision tree plot
                            max_depth_display = min(5, model.get_depth())  # Use default depth
                            
                            # Determine if classification or regression
                            if hasattr(model, 'classes_'):
                                class_names = [str(c) for c in model.classes_]
                            else:
                                class_names = None
                            
                            tree_fig = create_interactive_tree_plot(
                                model, 
                                independent_vars, 
                                class_names=class_names,
                                max_depth=max_depth_display,
                                prob_class_index=prob_class_index
                            )
                            
                            # Display tree matching expected image format
                            # Display dynamic info message based on probability selection
                            if class_names and len(class_names) == 2:
                                selected_class = class_names[prob_class_index]
                                st.info(f"🌳 Tree visualization shows probability for **{selected_class}** and sample percentages on each node. Colors indicate probability levels. Hover over nodes for detailed information.")
                            else:
                                st.info("🌳 Tree visualization shows probabilities and percentages clearly displayed on each node. Colors indicate confidence levels.")
                            
                            # Show tree directly in interface with improved button configuration
                            st.success("💡 **Tip:** Click the fullscreen button (⛶) in the top-right corner of the plot for the best viewing experience!")
                            st.plotly_chart(tree_fig, use_container_width=True, config={
                                'displayModeBar': True, 
                                'displaylogo': False,
                                'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                                'toImageButtonOptions': {
                                    'format': 'png',
                                    'filename': 'decision_tree',
                                    'height': 1200,
                                    'width': 1600,
                                    'scale': 2
                                }
                            })
                            
                            # Single download option matching display format
                            st.markdown("### 📥 Download Tree Visualization")
                            
                            # Configure for high-quality PNG export matching display
                            export_fig = tree_fig  # Use same figure
                            try:
                                png_bytes = export_fig.to_image(format="png", width=1200, height=1000, scale=2)
                                
                                st.download_button(
                                    label="📸 Download PNG (High Quality)",
                                    data=png_bytes,
                                    file_name=f"decision_tree_{datetime.now(pytz.timezone('US/Central')).strftime('%Y%m%d_%H%M%S')}.png",
                                    mime="image/png",
                                    help="Download high-resolution PNG showing probabilities and percentages clearly"
                                )
                            except Exception as e:
                                st.warning("⚠️ PNG download requires the kaleido package. Install with: pip install kaleido")
                                
                            # Alternative: HTML download (always available)
                            html_str = tree_fig.to_html(include_plotlyjs='cdn')
                            st.download_button(
                                label="📁 Download Tree (HTML)",
                                data=html_str,
                                file_name=f"decision_tree_{datetime.now(pytz.timezone('US/Central')).strftime('%Y%m%d_%H%M%S')}.html",
                                mime="text/html",
                                help="Download interactive tree as HTML file (always available)"
                            )
                            
                            # Track visualization creation
                            track_feature_usage("visualizations_created")
                            
                            # Display pruning information if available
                            display_pruning_info("Decision Tree")
                            
                            # Text representation
                            with st.expander("📄 Tree Rules (Text Format)"):
                                tree_rules = export_text(model, feature_names=independent_vars, max_depth=max_depth_display)
                                st.text(tree_rules)
                        
                        elif estimation_method == "Random Forest":
                            # For Random Forest, show feature importance and individual trees
                            st.markdown('<h2 class="subheader">🌲 Random Forest Analysis</h2>', unsafe_allow_html=True)
                            
                            # Feature importance plot
                            st.subheader("Feature Importance")
                            importance_fig = create_forest_importance_plot(model, independent_vars)
                            st.plotly_chart(importance_fig, use_container_width=True)
                            
                            # Individual tree visualization
                            st.subheader("Individual Tree Visualization")
                            tree_index = st.slider("Select tree to visualize", 0, len(model.estimators_)-1, 0)
                            max_depth_display = min(5, model.estimators_[tree_index].get_depth())  # Use default depth
                            
                            # Determine if classification or regression
                            if hasattr(model, 'classes_'):
                                class_names = [str(c) for c in model.classes_]
                            else:
                                class_names = None
                            
                            individual_tree_fig = create_interactive_tree_plot(
                                model.estimators_[tree_index], 
                                independent_vars, 
                                class_names=class_names,
                                max_depth=max_depth_display,
                                prob_class_index=prob_class_index
                            )
                            st.success("💡 **Tip:** Click the fullscreen button (⛶) in the top-right corner of the plot for the best viewing experience!")
                            st.plotly_chart(individual_tree_fig, use_container_width=True, config={
                                'displayModeBar': True, 
                                'displaylogo': False,
                                'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                                'toImageButtonOptions': {
                                    'format': 'png',
                                    'filename': 'random_forest_tree',
                                    'height': 1200,
                                    'width': 1600,
                                    'scale': 2
                                }
                            })
                            
                            # Download button for random forest individual tree
                            col1, col2, col3 = st.columns([1, 1, 2])
                            with col1:
                                # Create export version with better spacing
                                export_individual_fig = create_interactive_tree_plot(
                                    model.estimators_[tree_index], 
                                    independent_vars, 
                                    class_names=class_names,
                                    max_depth=max_depth_display,
                                    prob_class_index=prob_class_index
                                )
                                export_individual_fig.update_layout(
                                    width=2000,  # Extra wide for export
                                    height=1500,  # Extra tall for export
                                    margin=dict(l=100, r=100, t=150, b=100)  # More margins
                                )
                                
                                # Download as HTML
                                html_str = export_individual_fig.to_html(include_plotlyjs='cdn')
                                st.download_button(
                                    label="📁 Download Tree (HTML)",
                                    data=html_str,
                                    file_name=f"random_forest_tree_{tree_index}.html",
                                    mime="text/html",
                                    help="Download interactive tree as HTML file"
                                )
                            with col2:
                                # Download as PNG using export figure
                                try:
                                    img_bytes = export_individual_fig.to_image(format="png", width=2000, height=1500)
                                    st.download_button(
                                        label="🖼️ Download Tree (PNG)",
                                        data=img_bytes,
                                        file_name=f"random_forest_tree_{tree_index}.png",
                                        mime="image/png",
                                        help="Download tree as PNG image"
                                    )
                                except Exception as e:
                                    st.caption("⚠️ PNG download requires kaleido package")
                            
                            # Text representation of selected tree
                            with st.expander(f"📄 Tree {tree_index} Rules (Text Format)"):
                                tree_rules = export_text(model.estimators_[tree_index], feature_names=independent_vars, max_depth=max_depth_display)
                                st.text(tree_rules)
                            
                            # Display pruning information if available
                            display_pruning_info("Random Forest")
                    
                    else:
                        # Linear models - show coefficients table
                        st.write("**Regression Coefficients:**")
                        
                        coef_data = []
                        variable_names = (['Intercept'] if include_constant else []) + independent_vars
                        
                        # Handle coefficient concatenation properly for different model types
                        if include_constant:
                            if estimation_method == 'Logistic Regression':
                                # For logistic regression: handle multidimensional arrays properly
                                if hasattr(model, 'intercept_') and model.intercept_.ndim > 0:
                                    intercept_part = model.intercept_.flatten()
                                else:
                                    intercept_part = np.array([float(model.intercept_)])
                                coefficients = np.concatenate([intercept_part, model.coef_.flatten()])
                            else:
                                # For linear models: handle as scalars
                                intercept_val = float(model.intercept_) if hasattr(model, 'intercept_') else 0.0
                                coefficients = np.concatenate([[intercept_val], model.coef_])
                        else:
                            if estimation_method == 'Logistic Regression':
                                coefficients = model.coef_.flatten()
                            else:
                                coefficients = model.coef_
                        
                        for i, var_name in enumerate(variable_names):
                            coef_entry = {
                                'Variable': var_name,
                                'Coefficient': float(coefficients[i])  # Convert to Python float
                            }
                            
                            # Add statistical tests only for OLS
                            if estimation_method == "OLS":
                                coef_entry.update({
                                    'Std Error': float(stats_dict['std_errors'][i]),
                                    't-statistic': float(stats_dict['t_stats'][i]),
                                    'P-value': float(stats_dict['p_values'][i]),
                                    'Significance': '***' if float(stats_dict['p_values'][i]) < 0.01 else 
                                                  '**' if float(stats_dict['p_values'][i]) < 0.05 else 
                                                  '*' if float(stats_dict['p_values'][i]) < 0.1 else ''
                                })
                            else:
                                # For regularized methods, show if coefficient was shrunk to zero
                                coef_entry['Status'] = 'Selected' if abs(float(coefficients[i])) > 1e-10 else 'Excluded'
                            
                            coef_data.append(coef_entry)
                        
                        coef_df = pd.DataFrame(coef_data)
                        st.dataframe(coef_df, use_container_width=True)
                    
                    if estimation_method == "OLS":
                        st.caption("Significance levels: *** p<0.01, ** p<0.05, * p<0.1")
                        # F-statistic
                        st.write(f"**F-statistic:** {float(stats_dict['f_stat']):.4f} (p-value: {float(stats_dict['f_p_value']):.4f})")
                    else:
                        st.caption("Regularized methods don't provide traditional statistical significance tests")
                        # Show cross-validation score if desired
                        try:
                            if estimation_method in ["Lasso", "Ridge", "Elastic Net"]:
                                cv_scores = cross_val_score(model, X_for_plotting, y, cv=5, scoring='r2')
                                st.write(f"**Cross-Validation R² Score:** {float(cv_scores.mean()):.4f} (±{float(cv_scores.std()*2):.4f})")
                        except:
                            pass
                    
                    # Visualization section
                    st.markdown('<h2 class="subheader">📊 Visualization</h2>', unsafe_allow_html=True)
                    
                    # Create tabs for different plots
                    tab1, tab2, tab3, tab4 = st.tabs(["Actual vs Fitted", "Residuals vs Fitted", "Q-Q Plot", "Histogram of Residuals"])
                    
                    with tab1:
                        # Actual vs Fitted values plot
                        fig = px.scatter(
                            x=stats_dict['fitted_values'],
                            y=y,
                            labels={'x': 'Fitted Values', 'y': f'Actual {dependent_var}'},
                            title=f"Actual vs Fitted Values ({estimation_method})"
                        )
                        
                        # Add perfect prediction line (y = x)
                        min_val = min(min(stats_dict['fitted_values']), min(y))
                        max_val = max(max(stats_dict['fitted_values']), max(y))
                        fig.add_trace(go.Scatter(
                            x=[min_val, max_val],
                            y=[min_val, max_val],
                            mode='lines',
                            name='Perfect Prediction (y=x)',
                            line=dict(color='red', width=2, dash='dash')
                        ))
                        
                        st.plotly_chart(fig, use_container_width=True)
                        # Track visualization creation
                        track_feature_usage("visualizations_created")
                        st.caption("Points closer to the red line indicate better predictions. The closer the points to the diagonal line, the better the model fit.")
                    
                    with tab2:
                        # Residuals vs Fitted values
                        fig = px.scatter(
                            x=stats_dict['fitted_values'], 
                            y=stats_dict['residuals'],
                            labels={'x': 'Fitted Values', 'y': 'Residuals'},
                            title="Residuals vs Fitted Values"
                        )
                        fig.add_hline(y=0, line_dash="dash", line_color="red")
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption("This plot helps assess homoscedasticity. Points should be randomly scattered around the horizontal line.")
                    
                    with tab3:
                        # Q-Q plot for normality of residuals
                        from scipy.stats import probplot
                        qq_data = probplot(stats_dict['residuals'], dist="norm")
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=qq_data[0][0], 
                            y=qq_data[0][1],
                            mode='markers',
                            name='Residuals'
                        ))
                        
                        # Add diagonal line
                        line_x = np.array([qq_data[0][0].min(), qq_data[0][0].max()])
                        line_y = qq_data[1][1] + qq_data[1][0] * line_x
                        fig.add_trace(go.Scatter(
                            x=line_x, 
                            y=line_y,
                            mode='lines',
                            name='Normal Distribution',
                            line=dict(color='red', dash='dash')
                        ))
                        
                        fig.update_layout(
                            title="Q-Q Plot of Residuals",
                            xaxis_title="Theoretical Quantiles",
                            yaxis_title="Sample Quantiles"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption("Points should lie close to the diagonal line if residuals are normally distributed.")
                    
                    with tab4:
                        # Histogram of residuals
                        fig = px.histogram(
                            x=stats_dict['residuals'],
                            nbins=20,
                            title="Distribution of Residuals"
                        )
                        fig.update_layout(xaxis_title="Residuals", yaxis_title="Frequency")
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption("Residuals should be approximately normally distributed.")
                    
                    # Model interpretation
                    st.markdown('<h2 class="subheader">💡 Model Interpretation</h2>', unsafe_allow_html=True)
                    
                    if estimation_method in ["Decision Tree", "Random Forest"]:
                        # Tree model interpretation
                        st.write(f"**{estimation_method} Model Summary:**")
                        
                        # Tree-specific insights
                        insights = []
                        
                        if model_type == 'classification':
                            accuracy_pct = stats_dict['accuracy'] * 100
                            insights.append(f"• The model achieves {accuracy_pct:.1f}% accuracy in predicting {dependent_var}")
                        else:
                            r_sq_pct = stats_dict['r_squared'] * 100
                            insights.append(f"• The model explains {r_sq_pct:.1f}% of the variance in {dependent_var}")
                        
                        # Feature importance insights
                        sorted_features = sorted(zip(independent_vars, model.feature_importances_), 
                                               key=lambda x: x[1], reverse=True)
                        
                        insights.append(f"• Most important feature: **{sorted_features[0][0]}** (importance: {sorted_features[0][1]:.3f})")
                        
                        if len(sorted_features) > 1:
                            insights.append(f"• Second most important: **{sorted_features[1][0]}** (importance: {sorted_features[1][1]:.3f})")
                        
                        # Top 3 features by importance
                        top_features = [f"{feat} ({imp:.3f})" for feat, imp in sorted_features[:3]]
                        insights.append(f"• Top 3 features: {', '.join(top_features)}")
                        
                        if estimation_method == "Decision Tree":
                            insights.append(f"• Tree depth: {model.get_depth()} levels")
                            insights.append(f"• Number of leaves: {model.get_n_leaves()}")
                        else:  # Random Forest
                            insights.append(f"• Ensemble of {model.n_estimators} trees")
                            avg_depth = np.mean([tree.get_depth() for tree in model.estimators_])
                            insights.append(f"• Average tree depth: {avg_depth:.1f} levels")
                        
                        for insight in insights:
                            st.write(insight)
                        
                        # Model interpretation guidance
                        st.write("**How to interpret the tree:**")
                        interpretation_guide = [
                            "• Each node shows a decision rule (e.g., 'Feature ≤ threshold')",
                            "• Left branch = condition is True, Right branch = condition is False",
                            "• Leaf nodes show the final prediction",
                            "• Node color intensity indicates prediction confidence",
                            "• Sample count shows how many training examples reached each node"
                        ]
                        
                        for guide in interpretation_guide:
                            st.write(guide)
                    
                    else:
                        # Linear model interpretation
                        interpretation_text = f"""
                        **Model Equation ({estimation_method}):**
                        {dependent_var} = """
                        
                        # Handle both binary and multiclass logistic regression
                        if hasattr(model, 'coef_') and model.coef_.ndim > 1:
                            # Multiclass case - use first class coefficients for display
                            intercept_val = float(model.intercept_[0]) if model.intercept_.ndim > 0 else float(model.intercept_)
                            coef_vals = model.coef_[0]
                        else:
                            # Binary case or linear regression
                            intercept_val = float(model.intercept_) if hasattr(model, 'intercept_') else 0.0
                            coef_vals = model.coef_
                        
                        if include_constant:
                            interpretation_text += f"{intercept_val:.4f}"
                        
                        for i, var in enumerate(independent_vars):
                            coef_val = float(coef_vals[i]) if hasattr(coef_vals, '__getitem__') else float(coef_vals)
                            if include_constant:
                                sign = "+" if coef_val >= 0 else ""
                                interpretation_text += f" {sign} {coef_val:.4f} × {var}"
                            else:
                                if i == 0:
                                    interpretation_text += f"{coef_val:.4f} × {var}"
                                else:
                                    sign = "+" if coef_val >= 0 else ""
                                    interpretation_text += f" {sign} {coef_val:.4f} × {var}"
                        
                        st.write(interpretation_text)
                        
                        st.write("**Key Insights:**")
                        insights = []
                        
                        # R-squared interpretation
                        if model_type == 'regression':
                            r_sq_pct = stats_dict['r_squared'] * 100
                            insights.append(f"• The model explains {r_sq_pct:.1f}% of the variance in {dependent_var}")
                        else:
                            accuracy_pct = stats_dict['accuracy'] * 100
                            insights.append(f"• The model achieves {accuracy_pct:.1f}% accuracy in predicting {dependent_var}")
                        
                        # Method-specific insights
                        if estimation_method == "OLS":
                            # Coefficient interpretations for OLS
                            for i, var in enumerate(independent_vars):
                                # Handle multidimensional coefficient arrays
                                if hasattr(model, 'coef_') and model.coef_.ndim > 1:
                                    coef = float(model.coef_[0][i])  # Use first class for multiclass
                                else:
                                    coef = float(model.coef_[i])  # Binary or regression case
                                
                                if 'p_values' in stats_dict:
                                    p_val = float(stats_dict['p_values'][i + 1])  # +1 because intercept is first
                                    
                                    significance = ""
                                    if p_val < 0.01:
                                        significance = " (highly significant)"
                                    elif p_val < 0.05:
                                        significance = " (significant)"
                                    elif p_val < 0.1:
                                        significance = " (marginally significant)"
                                    else:
                                        significance = " (not significant)"
                                else:
                                    significance = ""
                                
                                direction = "increases" if coef > 0 else "decreases"
                                insights.append(f"• A one-unit increase in {var} is associated with a {abs(coef):.4f} unit {direction} in {dependent_var}{significance}")
                        
                        else:
                            # Regularized methods insights - handle multidimensional arrays
                            selected_vars = []
                            excluded_vars = []
                            
                            for i, var in enumerate(independent_vars):
                                if hasattr(model, 'coef_') and model.coef_.ndim > 1:
                                    coef_val = float(model.coef_[0][i])  # Use first class for multiclass
                                else:
                                    coef_val = float(model.coef_[i])  # Binary or regression case
                                
                                if abs(coef_val) > 1e-10:
                                    selected_vars.append(var)
                                else:
                                    excluded_vars.append(var)
                            
                            if selected_vars:
                                insights.append(f"• {estimation_method} selected {len(selected_vars)} out of {len(independent_vars)} variables: {', '.join(selected_vars)}")
                            if excluded_vars:
                                insights.append(f"• Variables excluded by regularization: {', '.join(excluded_vars)}")
                            
                            for i, var in enumerate(independent_vars):
                                if hasattr(model, 'coef_') and model.coef_.ndim > 1:
                                    coef = float(model.coef_[0][i])  # Use first class for multiclass
                                else:
                                    coef = float(model.coef_[i])  # Binary or regression case
                                
                                if abs(coef) > 1e-10:  # Variable was selected
                                    direction = "increases" if coef > 0 else "decreases"
                                    insights.append(f"• {var}: coefficient = {coef:.4f} (selected by {estimation_method})")
                        
                        for insight in insights:
                            st.write(insight)
            
            else:
                st.sidebar.warning("⚠️ Please select at least one independent variable.")
            
            # Owner Access and Version Information at bottom of sidebar
            st.sidebar.markdown("---")
            
            # Secret access to analytics (only for creator)
            if st.sidebar.checkbox("🔒 Owner Access", value=False, help="For app creator only"):
                owner_password = st.sidebar.text_input("Enter owner password:", type="password")
                if owner_password == "4693943198":  # Change this password as needed
                    st.session_state.show_analytics = True
                    show_analytics_option = True
            
            # Version information and changelog in sidebar
            with st.sidebar.expander("📋 Version Info & Changelog", expanded=False):
                st.markdown("**Current Version:** 2.1.0")
                st.markdown("**Release Date:** September 4, 2025")
                
                # Show recent updates
                st.markdown("**Recent Updates:**")
                st.markdown("""
                • 🌳 **Cost Complexity Pruning** for Decision Trees
                • ⚙️ **Enhanced Regularization Controls** 
                • 🎯 **Improved Binary Classification**
                • 🔒 **Hidden Usage Analytics**
                • 📊 **Better Tree Visualizations**
                """)
                
                # Simplified changelog access
                st.markdown("📄 **[View Complete Changelog](https://github.com/rabbitfxzx2010/econometric-analysis-tool/blob/main/CHANGELOG.md)**")
        
        except Exception as e:
            st.error(f"❌ Error reading the file: {str(e)}")
            st.info("Please make sure your CSV file is properly formatted with column headers.")
    
    else:
        # Instructions when no file is uploaded
        st.info("👆 Please upload a CSV file using the sidebar to get started.")
    
    # Display usage statistics only for the owner
    if st.session_state.get('show_analytics', False):
        st.markdown("---")
        st.markdown("### App Usage Statistics")
        
        # Get persistent usage statistics from file
        usage_file = "app_usage_stats.json"
        total_models_run = 0
        total_sessions = 0
        
        try:
            if os.path.exists(usage_file):
                with open(usage_file, "r") as f:
                    usage_data = json.load(f)
                    total_models_run = usage_data.get("total_models_run", 0)
                    total_sessions = usage_data.get("total_sessions", 0)
        except (json.JSONDecodeError, FileNotFoundError):
            total_models_run = 0
            total_sessions = 0
        
        # Create columns for statistics display
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Total Models Run", 
                value=f"{total_models_run:,}",
                help="Total number of machine learning models executed since app launch"
            )
        
        with col2:
            st.metric(
                label="Total Sessions", 
                value=f"{total_sessions:,}",
                help="Total number of user sessions recorded"
            )
        
        with col3:
            st.metric(
                label="This Session", 
                value=f"{st.session_state.models_run_count}",
                help="Number of models you've run in this session"
            )
        
        # Show current session info
        st.caption(f"Last updated: {datetime.now(pytz.timezone('US/Central')).strftime('%Y-%m-%d %H:%M:%S CST')} | Session ID: {id(st.session_state)}")
    
    if uploaded_file is None:
        st.markdown("""
        ## How to use this tool:
        
        1. **Upload your data**: Use the file uploader in the sidebar to upload a CSV or Excel file
        2. **Select Excel sheet**: If uploading Excel, choose which sheet to analyze
        3. **Filter sample** (optional): Select specific observations using row ranges or conditions
        4. **Explore your data**: Review the dataset overview and column information
        5. **Select variables**: Choose your dependent variable (Y) and independent variables (X)
        6. **Choose estimation method**: Select from OLS, Lasso, Ridge, or Elastic Net
        7. **Set parameters**: Adjust regularization parameters for Lasso/Ridge/Elastic Net
        8. **Run regression**: Click the regression button to perform the analysis
        9. **Interpret results**: Review the coefficients, statistics, and visualizations
        
        ## What you'll get:
        
        - **Comprehensive statistics**: R-squared, adjusted R-squared, RMSE, and more
        - **Method-specific output**: 
          - OLS: Standard errors, t-statistics, p-values, F-statistic
          - Regularized methods: Variable selection results, cross-validation scores
        - **Diagnostic plots**: Scatter plots, residual analysis, and normality tests
        - **Model interpretation**: Clear explanations tailored to the chosen method
        
        ## Sample data format:
        
        Your CSV/Excel should have column headers and numeric data. For example:
        
        ```
        income,education,experience,age
        50000,16,5,28
        65000,18,8,32
        45000,14,3,25
        ...
        ```
        """)

if __name__ == "__main__":
    main()