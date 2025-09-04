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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import StringIO
import openpyxl
from datetime import datetime, date
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
    current_time = datetime.now()
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
    
    if os.path.exists(usage_file):
        try:
            with open(usage_file, "r") as f:
                usage_data = json.load(f)
        except:
            return
        
        # Update feature usage
        if feature_name in usage_data["feature_usage"]:
            usage_data["feature_usage"][feature_name] += 1
        
        # Update daily feature usage
        if today in usage_data["daily_stats"]:
            if feature_name == "model_runs":
                usage_data["daily_stats"][today]["models_run"] += 1
            elif feature_name == "file_uploads":
                usage_data["daily_stats"][today]["files_uploaded"] += 1
        
        # Save updated data
        try:
            with open(usage_file, "w") as f:
                json.dump(usage_data, f, indent=2)
        except:
            pass

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
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
    
    # Add edges with labels and improved nodes for maximum visibility
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
    
    # Create nodes with enhanced visibility and heatmap coloring
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
                
                # Calculate color intensity for heatmap
                if max_prob > min_prob:
                    color_intensity = (predicted_probability - min_prob) / (max_prob - min_prob)
                else:
                    color_intensity = predicted_probability
                
                # Enhanced heatmap colors: light blue to deep red
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
                
                # GUARANTEED VISIBLE text: probability and percentage
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
                
                node_text = f"<b>{predicted_probability:.3f}</b><br><b>{proportion*100:.0f}%</b>"
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
        
        # Create BIGGER rectangular nodes for better visibility
        node_width = 1.8   # Even bigger width
        node_height = 1.2  # Bigger height
        
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
            font=dict(size=24, color=text_color, family='Arial Black'),  # VERY large font
            bgcolor='rgba(255,255,255,0.4)' if text_color == 'white' else 'rgba(0,0,0,0.2)',
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
            marker=dict(size=50, color='rgba(0,0,0,0)'),
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
    
    # Add color scale legend
    if class_names is not None:
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
                    tickfont=dict(size=11, family='Arial Bold')
                )
            ),
            showlegend=False,
            name='Probability Scale'
        ))
    else:
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
    
    # Update layout with MAXIMUM spacing and margins
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

# Enhanced CSS styling for better appearance
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem !important;
    font-weight: 700 !important;
    color: #1f77b4 !important;
    text-align: center !important;
    margin-bottom: 2rem !important;
    padding: 1rem !important;
    border-bottom: 3px solid #1f77b4 !important;
}

.subheader {
    font-size: 1.8rem !important;
    font-weight: 600 !important;
    color: #2c3e50 !important;
    margin-top: 2rem !important;
    margin-bottom: 1rem !important;
    padding-bottom: 0.5rem !important;
    border-bottom: 2px solid #ecf0f1 !important;
}

.metric-container {
    background-color: #f8f9fa !important;
    padding: 1rem !important;
    border-radius: 0.5rem !important;
    border: 1px solid #e9ecef !important;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)


def calculate_regression_stats(X, y, model, method='OLS', fit_intercept=True):
        
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
        if max_depth is not None and positions[n][1] < -max_depth * 2.5:
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
        if max_depth is not None and positions[node][1] < -max_depth * 2.5:
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
        if max_depth is not None and positions[node][1] < -max_depth * 2.5:
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
                
                # Create heatmap-style colors: light blue (low) to deep red/blue (high)
                # Similar to SimilarWeb intelligence heatmap
                if color_intensity < 0.2:
                    # Very light blue/white for very low probabilities
                    node_color = f'rgba(240, 248, 255, 0.95)'  # Alice blue - very light
                    text_color = 'black'
                elif color_intensity < 0.4:
                    # Light blue for low probabilities
                    node_color = f'rgba(173, 216, 230, 0.95)'  # Light blue
                    text_color = 'black'
                elif color_intensity < 0.6:
                    # Medium orange-red for medium probabilities  
                    node_color = f'rgba(255, 140, 105, 0.95)'  # Coral/salmon
                    text_color = 'black'
                elif color_intensity < 0.8:
                    # Deep orange-red for high probabilities
                    node_color = f'rgba(255, 69, 58, 0.95)'   # Red-orange
                    text_color = 'white'
                else:
                    # Deep red for very high probabilities
                    node_color = f'rgba(220, 20, 20, 0.95)'   # Deep red
                    text_color = 'white'
                
                # Node text: probability and percentage on the square
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
                
                # Heatmap-style colors for regression too
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
                
                # Lighter heatmap colors for internal nodes with better contrast
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
                
                # Lighter heatmap colors for internal regression nodes with better contrast
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
        
        # Add node text (probability and percentage) with MAXIMUM visibility
        fig.add_annotation(
            x=x, y=y,
            text=node_text,
            showarrow=False,
            font=dict(size=22, color=text_color, family='Arial Black'),  # Even larger, bolder font
            bgcolor='rgba(255,255,255,0.3)' if text_color == 'white' else 'rgba(0,0,0,0.15)',  # More visible background
            bordercolor=text_color,
            borderwidth=1.5
        )
        
        # Add decision rule below internal nodes
        if not is_leaf:
            fig.add_annotation(
                x=x, y=y - 0.8,
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
    
    # Update layout for horizontal scrolling and bigger squares
    fig.update_layout(
        title=dict(
            text="Decision Tree Probability Heatmap<br><sub>Probabilities and percentages displayed - Colors show confidence levels - Hover for details</sub>",
            font=dict(size=18, color='black'),
            x=0.5,
            xanchor='center'
        ),
        showlegend=False,
        hovermode='closest',
        margin=dict(b=200, l=150, r=250, t=150),  # MUCH larger margins to prevent cutoff
        annotations=[
            dict(
                text="HEATMAP STYLE GUIDE:<br>" +
                     "LIGHT BLUE = Low probability - CORAL/ORANGE = Medium probability - DEEP RED = High probability<br>" +
                     "Numbers show: Top = probability, Bottom = sample percentage<br>" +
                     "Decision rules shown below internal nodes",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.5, y=-0.12,
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
        # Make figure much wider and taller to prevent cutoff and accommodate bigger squares
        width=6000,   # Much wider for better spacing and no overlap
        height=3000,  # Much taller to prevent bottom cutoff
        dragmode='pan'  # Enable horizontal scrolling
    )
    
    return fig


def create_forest_importance_plot(model, feature_names):
        
        if int(children_left[node]) != int(children_right[node]):  # Not a leaf
            # Calculate spacing for children - MUCH wider spacing
            spacing = spacing_factor / (level * 0.7 + 1)  # Reduce spacing decrease
            
            # Left child
            if children_left[node] >= 0:
                left_child = int(children_left[node])
                get_tree_positions(left_child, x - spacing, y - 2, level + 1, positions, spacing_factor)
            
            # Right child
            if children_right[node] >= 0:
                right_child = int(children_right[node])
                get_tree_positions(right_child, x + spacing, y - 2, level + 1, positions, spacing_factor)
        
        return positions
    
    positions = get_tree_positions()
    
    # Create the plot
    fig = go.Figure()
    
    # Add edges first (so they appear behind nodes)
    edge_x = []
    edge_y = []
    
    for node in range(tree.node_count):
        if max_depth is not None and positions[node][1] < -max_depth * 2:
            continue
            
        if children_left[node] >= 0:  # Has left child
            left_child = int(children_left[node])
            if left_child in positions:
                x0, y0 = positions[node]
                x1, y1 = positions[left_child]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
        
        if children_right[node] >= 0:  # Has right child
            right_child = int(children_right[node])
            if right_child in positions:
                x0, y0 = positions[node]
                x1, y1 = positions[right_child]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
    
    # Add edges with thick lines
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(color='rgba(50,50,50,0.6)', width=4),
        hoverinfo='none',
        showlegend=False,
        name='Tree Structure'
    ))
    
    # Calculate value ranges for heatmap coloring
    all_values = []
    all_impurities = []
    all_proportions = []
    
    for n in range(tree.node_count):
        if max_depth is not None and positions[n][1] < -max_depth * 2:
            continue
        samples = int(n_node_samples[n])
        proportion = samples / total_samples
        all_proportions.append(proportion)
        all_impurities.append(float(impurity[n]))
        
        if class_names is None:  # Regression
            all_values.append(float(value[n][0][0]))
    
    if class_names is None:  # Regression
        min_val, max_val = min(all_values), max(all_values)
    min_prop, max_prop = min(all_proportions), max(all_proportions)
    min_imp, max_imp = min(all_impurities), max(all_impurities)
    
    # Create nodes with heatmap coloring and GUARANTEED text visibility
    for node in range(tree.node_count):
        if max_depth is not None and positions[node][1] < -max_depth * 2:
            continue
            
        x, y = positions[node]
        
        # Node information
        samples = int(n_node_samples[node])
        proportion = samples / total_samples
        impurity_val = float(impurity[node])
        
        # Calculate LARGE square size for text visibility
        base_size = 1.2  # Very large base size
        square_size = base_size + (proportion * 0.8)  # Even larger for important nodes
        
        is_leaf = int(children_left[node]) == int(children_right[node])
        
        if is_leaf:  # Leaf node
            if class_names is not None:  # Classification
                predicted_class = int(np.argmax(value[node][0]))
                class_probs = value[node][0] / np.sum(value[node][0])
                predicted_probability = float(class_probs[predicted_class])
                
                # Heatmap color based on probability (high probability = warmer color)
                color_intensity = predicted_probability
                # Warm colors for high probability: red to orange to yellow
                red = int(255)
                green = int(255 * (1 - color_intensity * 0.3))  # Less green for higher probability
                blue = int(100 * (1 - color_intensity))  # Much less blue for higher probability
                bg_color = f'rgba({red}, {green}, {blue}, 0.85)'
                
                # GUARANTEED VISIBLE TEXT with thick black outline
                square_text = f"<b>🎯 LEAF</b><br><b>Prop: {proportion:.2f}</b><br><b>Prob: {predicted_probability:.2f}</b><br><b>Class: {class_names[predicted_class]}</b>"
                
                # Comprehensive hover info
                prob_text = "<br>".join([f"  🔸 {cls}: {prob:.3f}" for cls, prob in zip(class_names, class_probs)])
                hover_text = (f"<b>🎯 FINAL PREDICTION (LEAF NODE)</b><br>"
                            f"📊 Sample Proportion: {proportion:.1%} ({samples:,}/{total_samples:,})<br>"
                            f"🎲 Predicted Class: <b>{class_names[predicted_class]}</b><br>"
                            f"📈 Confidence: {predicted_probability:.3f}<br>"
                            f"� Impurity: {impurity_val:.3f}<br>"
                            f"🔍 All Class Probabilities:<br>{prob_text}")
                            
            else:  # Regression
                predicted_value = float(value[node][0][0])
                
                # Normalize for heatmap coloring
                if max_val > min_val:
                    color_intensity = (predicted_value - min_val) / (max_val - min_val)
                else:
                    color_intensity = 0.5
                
                # Cool to warm color scheme for regression values
                red = int(100 + 155 * color_intensity)
                green = int(150 + 105 * color_intensity)
                blue = int(255 - 155 * color_intensity)
                bg_color = f'rgba({red}, {green}, {blue}, 0.85)'
                
                square_text = f"<b>🎯 LEAF</b><br><b>Value: {predicted_value:.2f}</b><br><b>Prop: {proportion:.2f}</b>"
                
                hover_text = (f"<b>🎯 FINAL PREDICTION (LEAF NODE)</b><br>"
                            f"📊 Sample Proportion: {proportion:.1%} ({samples:,}/{total_samples:,})<br>"
                            f"📈 Predicted Value: <b>{predicted_value:.3f}</b><br>"
                            f"📉 MSE: {impurity_val:.3f}")
                            
        else:  # Internal node (including ROOT)
            feature_name = feature_names[int(feature[node])]
            threshold_val = float(threshold[node])
            
            # Special formatting for ROOT node
            if node == 0:
                node_type = "🌳 ROOT"
            else:
                node_type = "🌿 DECISION"
            
            if class_names is not None:  # Classification
                predicted_class = int(np.argmax(value[node][0]))
                class_probs = value[node][0] / np.sum(value[node][0])
                predicted_probability = float(class_probs[predicted_class])
                
                # Lighter colors for internal nodes with blue tint
                color_intensity = predicted_probability
                red = int(200 + 55 * color_intensity)
                green = int(220 + 35 * color_intensity)
                blue = int(255)
                bg_color = f'rgba({red}, {green}, {blue}, 0.8)'
                
                square_text = f"<b>{node_type}</b><br><b>{feature_name}</b><br><b>≤ {threshold_val:.2f}</b><br><b>Prop: {proportion:.2f}</b>"
                
                prob_text = "<br>".join([f"  🔸 {cls}: {prob:.3f}" for cls, prob in zip(class_names, class_probs)])
                hover_text = (f"<b>{node_type} NODE</b><br>"
                            f"🔀 Decision Rule: <b>{feature_name} ≤ {threshold_val:.3f}</b><br>"
                            f"📊 Sample Proportion: {proportion:.1%} ({samples:,}/{total_samples:,})<br>"
                            f"🎲 Current Best Class: <b>{class_names[predicted_class]}</b><br>"
                            f"📈 Current Probability: {predicted_probability:.3f}<br>"
                            f"� Impurity: {impurity_val:.3f}<br>"
                            f"📋 Class Distribution:<br>{prob_text}<br>"
                            f"⬅️ <b>LEFT (TRUE)</b>: {feature_name} ≤ {threshold_val:.3f}<br>"
                            f"➡️ <b>RIGHT (FALSE)</b>: {feature_name} > {threshold_val:.3f}")
                            
            else:  # Regression
                predicted_value = float(value[node][0][0])
                
                if max_val > min_val:
                    color_intensity = (predicted_value - min_val) / (max_val - min_val)
                else:
                    color_intensity = 0.5
                
                # Green tint for internal regression nodes
                red = int(200 + 55 * color_intensity)
                green = int(255)
                blue = int(200 + 55 * color_intensity)
                bg_color = f'rgba({red}, {green}, {blue}, 0.8)'
                
                square_text = f"<b>{node_type}</b><br><b>{feature_name}</b><br><b>≤ {threshold_val:.2f}</b><br><b>Prop: {proportion:.2f}</b>"
                
                hover_text = (f"<b>{node_type} NODE</b><br>"
                            f"🔀 Decision Rule: <b>{feature_name} ≤ {threshold_val:.3f}</b><br>"
                            f"📊 Sample Proportion: {proportion:.1%} ({samples:,}/{total_samples:,})<br>"
                            f"📊 Current Value: {predicted_value:.3f}<br>"
                            f"� MSE: {impurity_val:.3f}<br>"
                            f"⬅️ <b>LEFT (TRUE)</b>: {feature_name} ≤ {threshold_val:.3f}<br>"
                            f"➡️ <b>RIGHT (FALSE)</b>: {feature_name} > {threshold_val:.3f}")
        
        # Add rectangle with heatmap coloring and thick border
        fig.add_shape(
            type="rect",
            x0=x - square_size/2, y0=y - square_size/2,
            x1=x + square_size/2, y1=y + square_size/2,
            fillcolor=bg_color,
            line=dict(color='black', width=5)  # Very thick border for definition
        )
        
        # Add text with MAXIMUM visibility - white text with black shadow
        fig.add_annotation(
            x=x, y=y,
            text=square_text,
            showarrow=False,
            font=dict(
                size=14,  # Large readable font
                color='black',  # Black text for best contrast
                family='Arial Black'
            ),
            bgcolor='rgba(255,255,255,0.95)',  # Nearly opaque white background
            bordercolor='black',
            borderwidth=3,  # Thick border around text
            borderpad=6    # More padding for better readability
        )
        
        # Add invisible scatter point for hover with larger area
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='markers',
            marker=dict(size=40, color='rgba(0,0,0,0)'),  # Large invisible hover area
            hovertext=hover_text,
            hoverinfo='text',
            hoverlabel=dict(
                bgcolor='rgba(255,255,255,0.98)',
                bordercolor='black',
                font=dict(size=12, color='black', family='Arial')
            ),
            showlegend=False,
            name=f'Node_{node}'
        ))
    
    # Create heatmap legend
    if class_names is not None:
        # Classification heatmap legend
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(
                size=1,
                color=[0, 1],
                colorscale=[[0, 'rgb(255,255,100)'], [1, 'rgb(255,100,100)']],
                showscale=True,
                colorbar=dict(
                    title="Prediction<br>Confidence",
                    titleside="right",
                    thickness=20,
                    len=0.5,
                    x=1.02,
                    tickvals=[0, 0.5, 1],
                    ticktext=['Low', 'Medium', 'High']
                )
            ),
            showlegend=False,
            name='Confidence Scale'
        ))
    else:
        # Regression heatmap legend
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(
                size=1,
                color=[min_val, max_val] if max_val > min_val else [0, 1],
                colorscale=[[0, 'rgb(155,255,255)'], [1, 'rgb(255,155,155)']],
                showscale=True,
                colorbar=dict(
                    title="Predicted<br>Value",
                    titleside="right",
                    thickness=20,
                    len=0.5,
                    x=1.02
                )
            ),
            showlegend=False,
            name='Value Scale'
        ))
    
    # Update layout for maximum clarity
    fig.update_layout(
        title=dict(
            text="🌳 Interactive Decision Tree Heatmap<br><sub>🎯 ROOT clearly visible at top • 🌡️ Colors show prediction confidence/values • 📍 Hover for details</sub>",
            font=dict(size=18, color='black'),
            x=0.5,
            xanchor='center'
        ),
        showlegend=False,
        hovermode='closest',
        margin=dict(b=80,l=80,r=120,t=140),
        annotations=[
            dict(
                text="� <b>How to Read the Heatmap:</b><br>" +
                     "🌳 <b>ROOT</b> = starting point at top • 🌿 <b>DECISION</b> = internal split nodes • 🎯 <b>LEAF</b> = final predictions<br>" +
                     "🌡️ <b>Colors</b> = warmer (red/orange) = higher confidence/values, cooler (blue/green) = lower<br>" +
                     "📊 <b>Prop</b> = proportion of samples • <b>Prob</b> = prediction probability",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.5, y=-0.15,
                xanchor='center', yanchor='top',
                font=dict(color='rgb(60,60,60)', size=12),
                bgcolor='rgba(248,248,248,0.8)',
                bordercolor='gray',
                borderwidth=1,
                borderpad=10
            )
        ],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='rgba(250,250,250,1)',  # Very light background
        paper_bgcolor='white',
        width=2000,  # Much wider for better spacing
        height=1600   # Taller for better tree visibility
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
    
    fig = px.bar(
        feature_importance, 
        x='importance', 
        y='feature',
        orientation='h',
        title='Feature Importance (Random Forest)',
        labels={'importance': 'Importance', 'feature': 'Features'}
    )
    
    fig.update_layout(
        height=max(400, len(feature_names) * 25),
        margin=dict(l=150)
    )
    
    return fig

# Set page configuration
st.set_page_config(
    page_title="Supervised Learning Tool: Regression and Classification",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.subheader {
    font-size: 1.5rem;
    color: #ff7f0e;
    margin-top: 2rem;
    margin-bottom: 1rem;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

def calculate_regression_stats(X, y, model, method='OLS', fit_intercept=True):
    """Calculate comprehensive regression statistics for different methods"""
    # Predictions
    y_pred = model.predict(X)
    
    # Basic statistics
    n = len(y)
    k = X.shape[1]  # number of features
    
    # R-squared
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    # Adjusted R-squared (account for constant term)
    k_adj = k + (1 if fit_intercept else 0)  # Add 1 for intercept if included
    adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k_adj)
    
    # Mean Squared Error and Root Mean Squared Error
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
        model = LogisticRegression(fit_intercept=fit_intercept, random_state=42, max_iter=1000)
    elif method == 'Decision Tree':
        model_type = kwargs.get('model_type', 'regression')
        max_depth = kwargs.get('max_depth', None)
        min_samples_split = kwargs.get('min_samples_split', 2)
        min_samples_leaf = kwargs.get('min_samples_leaf', 1)
        
        if model_type == 'classification':
            model = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=42
            )
        else:
            model = DecisionTreeRegressor(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=42
            )
    elif method == 'Random Forest':
        model_type = kwargs.get('model_type', 'regression')
        n_estimators = kwargs.get('n_estimators', 100)
        max_depth = kwargs.get('max_depth', None)
        min_samples_split = kwargs.get('min_samples_split', 2)
        min_samples_leaf = kwargs.get('min_samples_leaf', 1)
        
        if model_type == 'classification':
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=42
            )
        else:
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=42
            )
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
    
    # Check if this is the analytics page (for app owner)
    page = st.sidebar.selectbox(
        "Navigation",
        ["📊 Main App", "📈 Usage Analytics (Owner)"],
        help="Select Main App for normal use, or Usage Analytics to view app usage statistics"
    )
    
    if page == "📈 Usage Analytics (Owner)":
        # Display usage analytics dashboard
        display_usage_analytics()
        return
    
    # Main header
    st.markdown('<h1 class="main-header">📊 Supervised Learning Tool: Regression and Classification</h1>', unsafe_allow_html=True)
    
    # About section first
    st.markdown("**About:** This webapp is created by Ren Zhang. Please leave your feedback below:")
    
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
                    alpha = st.sidebar.slider(
                        "Regularization Strength (α)",
                        min_value=0.001,
                        max_value=10.0,
                        value=1.0,
                        step=0.001,
                        help="Higher values increase regularization"
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
            else:
                max_depth = None
                min_samples_split = 2
                min_samples_leaf = 1
                n_estimators = 100
            
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
                    
                    # For regularized methods, standardize features
                    if estimation_method in ["Lasso", "Ridge", "Elastic Net"]:
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
                                        n_estimators=n_estimators)
                        # Track model run
                        track_feature_usage("model_runs")
                        
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
                                        n_estimators=n_estimators)
                        # Track model run
                        track_feature_usage("model_runs")
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
                            max_depth_display = st.slider("Maximum depth to display", 1, 10, min(5, model.get_depth()))
                            
                            # Determine if classification or regression
                            if hasattr(model, 'classes_'):
                                class_names = [str(c) for c in model.classes_]
                            else:
                                class_names = None
                            
                            tree_fig = create_interactive_tree_plot(
                                model, 
                                independent_vars, 
                                class_names=class_names,
                                max_depth=max_depth_display
                            )
                            st.info("📊 Tree visualization shows probabilities with blue heatmap colors. Darker blue = higher probability. You can scroll horizontally to see all nodes.")
                            st.plotly_chart(tree_fig, use_container_width=False)  # Don't use container width to allow scrolling
                            
                            # Download buttons for decision tree
                            st.markdown("### 📥 Download Tree Visualization")
                            col1, col2, col3 = st.columns([1, 1, 2])
                            
                            with col1:
                                # Create a copy of the figure for PNG export with full view
                                export_fig = create_interactive_tree_plot(
                                    model, 
                                    independent_vars, 
                                    class_names=class_names,
                                    max_depth=max_depth_display
                                )
                                
                                # Configure for high-quality PNG export
                                export_fig.update_layout(
                                    width=4000,   # Very wide for full tree view
                                    height=1600,  # Tall enough for full tree
                                    font=dict(size=14),  # Larger font for export
                                    margin=dict(l=100, r=300, t=150, b=100),  # Extra margins
                                )
                                
                                # Convert to PNG bytes with maximum quality and size to show full tree
                                png_bytes = export_fig.to_image(format="png", width=8000, height=3500, scale=3)
                                
                                st.download_button(
                                    label="📸 Download PNG (Full View)",
                                    data=png_bytes,
                                    file_name=f"decision_tree_{estimation_method.lower().replace(' ', '_')}.png",
                                    mime="image/png",
                                    help="Download high-resolution PNG showing the complete tree"
                                )
                            
                            with col2:
                                # Create HTML file with full interactive tree
                                html_str = export_fig.to_html(
                                    include_plotlyjs='cdn',
                                    config={
                                        'displayModeBar': True,
                                        'displaylogo': False,
                                        'modeBarButtonsToRemove': ['select2d', 'lasso2d'],
                                        'toImageButtonOptions': {
                                            'format': 'png',
                                            'filename': f'decision_tree_{estimation_method.lower().replace(" ", "_")}',
                                            'height': 1600,
                                            'width': 4000,
                                            'scale': 2
                                        }
                                    }
                                )
                                
                                st.download_button(
                                    label="🌐 Download HTML (Interactive)",
                                    data=html_str.encode('utf-8'),
                                    file_name=f"decision_tree_{estimation_method.lower().replace(' ', '_')}.html",
                                    mime="text/html",
                                    help="Download interactive HTML file that can be opened in any browser"
                                )
                            
                            with col3:
                                st.info("💡 **Download Tips:**\n- PNG: High-quality image for presentations\n- HTML: Interactive version for exploration\n- Both show the complete tree with all probability values")
                                export_fig.update_layout(
                                    width=2000,  # Extra wide for export
                                    height=1500,  # Extra tall for export
                                    margin=dict(l=100, r=100, t=150, b=100)  # More margins
                                )
                                
                                # Download as HTML
                                html_str = export_fig.to_html(include_plotlyjs='cdn')
                                st.download_button(
                                    label="📁 Download Tree (HTML)",
                                    data=html_str,
                                    file_name=f"decision_tree_{estimation_method.lower().replace(' ', '_')}.html",
                                    mime="text/html",
                                    help="Download interactive tree as HTML file"
                                )
                            with col2:
                                # Download as PNG using the same export figure
                                try:
                                    img_bytes = export_fig.to_image(format="png", width=2000, height=1500)
                                    st.download_button(
                                        label="🖼️ Download Tree (PNG)",
                                        data=img_bytes,
                                        file_name=f"decision_tree_{estimation_method.lower().replace(' ', '_')}.png",
                                        mime="image/png",
                                        help="Download tree as PNG image"
                                    )
                                except Exception as e:
                                    st.caption("⚠️ PNG download requires kaleido package")
                            
                            # Track visualization creation
                            track_feature_usage("visualizations_created")                            # Text representation
                            with st.expander("📄 Tree Rules (Text Format)"):
                                tree_rules = export_text(model, feature_names=independent_vars, max_depth=max_depth_display)
                                st.text(tree_rules)
                        
                        elif estimation_method == "Random Forest":
                            # For Random Forest, show feature importance plot and individual tree option
                            st.subheader("Feature Importance Plot")
                            importance_fig = create_forest_importance_plot(model, independent_vars)
                            st.plotly_chart(importance_fig, use_container_width=True)
                            
                            # Download button for feature importance plot
                            col1, col2, col3 = st.columns([1, 1, 2])
                            with col1:
                                # Download as HTML
                                html_str = importance_fig.to_html(include_plotlyjs='cdn')
                                st.download_button(
                                    label="📁 Download Importance (HTML)",
                                    data=html_str,
                                    file_name="random_forest_feature_importance.html",
                                    mime="text/html",
                                    help="Download feature importance plot as HTML file"
                                )
                            with col2:
                                # Download as PNG
                                try:
                                    img_bytes = importance_fig.to_image(format="png", width=800, height=600)
                                    st.download_button(
                                        label="🖼️ Download Importance (PNG)",
                                        data=img_bytes,
                                        file_name="random_forest_feature_importance.png",
                                        mime="image/png",
                                        help="Download feature importance plot as PNG image"
                                    )
                                except Exception as e:
                                    st.caption("⚠️ PNG download requires kaleido package")
                            
                            # Option to view individual trees
                            st.subheader("Individual Tree Visualization")
                            tree_index = st.slider("Select tree to visualize", 0, len(model.estimators_)-1, 0)
                            max_depth_display = st.slider("Maximum depth to display", 1, 10, min(5, model.estimators_[tree_index].get_depth()))
                            
                            # Determine if classification or regression
                            if hasattr(model, 'classes_'):
                                class_names = [str(c) for c in model.classes_]
                            else:
                                class_names = None
                            
                            individual_tree_fig = create_interactive_tree_plot(
                                model.estimators_[tree_index], 
                                independent_vars, 
                                class_names=class_names,
                                max_depth=max_depth_display
                            )
                            st.plotly_chart(individual_tree_fig, use_container_width=False)  # Don't use container width to allow scrolling
                            
                            # Download button for random forest individual tree
                            col1, col2, col3 = st.columns([1, 1, 2])
                            with col1:
                                # Create export version with better spacing
                                export_individual_fig = create_interactive_tree_plot(
                                    model.estimators_[tree_index], 
                                    independent_vars, 
                                    class_names=class_names,
                                    max_depth=max_depth_display
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
        
        except Exception as e:
            st.error(f"❌ Error reading the file: {str(e)}")
            st.info("Please make sure your CSV file is properly formatted with column headers.")
    
    else:
        # Instructions when no file is uploaded
        st.info("👆 Please upload a CSV file using the sidebar to get started.")
    
    # Display usage statistics at the bottom of the front page
    st.markdown("---")
    st.markdown("### App Usage Statistics")
    
    # Initialize session state for model run counter if it doesn't exist
    if 'models_run_count' not in st.session_state:
        st.session_state.models_run_count = 0
    
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
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Session ID: {id(st.session_state)}")
    
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