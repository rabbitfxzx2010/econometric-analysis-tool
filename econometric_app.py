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

# Email feedback function with daily limit
def send_feedback_email(feedback_text):
    """Send feedback via email with daily limit protection"""
    try:
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart
        import os
        from datetime import datetime, date
        
        # Check daily limit
        today = date.today().strftime("%Y-%m-%d")
        count_file = f"email_count_{today}.txt"
        
        # Read current count
        current_count = 0
        if os.path.exists(count_file):
            try:
                with open(count_file, "r") as f:
                    current_count = int(f.read().strip())
            except:
                current_count = 0
        
        # Check if limit reached
        if current_count >= 5:
            return False  # Limit reached
        
        # Save feedback locally as fallback
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
    Create an interactive decision tree visualization using Plotly.
    
    Parameters:
    - model: Trained decision tree model (DecisionTreeRegressor/Classifier)
    - feature_names: List of feature names
    - class_names: List of class names (for classification)
    - max_depth: Maximum depth to display (None for full tree)
    
    Returns:
    - Plotly figure object
    """
    tree = model.tree_
    feature = tree.feature
    threshold = tree.threshold
    children_left = tree.children_left
    children_right = tree.children_right
    value = tree.value
    impurity = tree.impurity
    n_node_samples = tree.n_node_samples
    
    # Calculate positions for nodes
    def get_tree_positions(node=0, x=0, y=0, level=0, positions=None, level_width=None):
        if positions is None:
            positions = {}
        if level_width is None:
            level_width = {}
            
        if max_depth is not None and level >= max_depth:
            return positions
            
        positions[node] = (x, y)
        
        if level not in level_width:
            level_width[level] = 0
        level_width[level] += 1
        
        if children_left[node] != children_right[node]:  # Not a leaf
            # Calculate spacing for children
            spacing = max(1.0 / (level + 1), 0.1)
            
            # Left child
            if children_left[node] >= 0:
                get_tree_positions(children_left[node], x - spacing, y - 1, level + 1, positions, level_width)
            
            # Right child
            if children_right[node] >= 0:
                get_tree_positions(children_right[node], x + spacing, y - 1, level + 1, positions, level_width)
        
        return positions
    
    positions = get_tree_positions()
    
    # Create edges
    edge_x = []
    edge_y = []
    edge_info = []
    
    for node in range(tree.node_count):
        if max_depth is not None and positions[node][1] < -max_depth:
            continue
            
        if children_left[node] >= 0:  # Has left child
            x0, y0 = positions[node]
            x1, y1 = positions[children_left[node]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_info.extend(['True', 'True', None])
        
        if children_right[node] >= 0:  # Has right child
            x0, y0 = positions[node]
            x1, y1 = positions[children_right[node]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_info.extend(['False', 'False', None])
    
    # Create nodes
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_info = []
    
    for node in range(tree.node_count):
        if max_depth is not None and positions[node][1] < -max_depth:
            continue
            
        x, y = positions[node]
        node_x.append(x)
        node_y.append(y)
        
        # Node information
        samples = n_node_samples[node]
        impurity_val = impurity[node]
        
        if children_left[node] == children_right[node]:  # Leaf node
            if class_names is not None:  # Classification
                predicted_class = np.argmax(value[node][0])
                class_probs = value[node][0] / np.sum(value[node][0])
                node_text.append(f"Class: {class_names[predicted_class]}<br>Samples: {samples}")
                node_color.append(predicted_class)
                
                prob_text = "<br>".join([f"{cls}: {prob:.3f}" for cls, prob in zip(class_names, class_probs)])
                node_info.append(f"Predicted Class: {class_names[predicted_class]}<br>"
                               f"Samples: {samples}<br>"
                               f"Impurity: {impurity_val:.3f}<br>"
                               f"Class Probabilities:<br>{prob_text}")
            else:  # Regression
                predicted_value = value[node][0][0]
                node_text.append(f"Value: {predicted_value:.3f}<br>Samples: {samples}")
                node_color.append(predicted_value)
                node_info.append(f"Predicted Value: {predicted_value:.3f}<br>"
                               f"Samples: {samples}<br>"
                               f"MSE: {impurity_val:.3f}")
        else:  # Internal node
            feature_name = feature_names[feature[node]]
            threshold_val = threshold[node]
            
            if class_names is not None:  # Classification
                predicted_class = np.argmax(value[node][0])
                node_text.append(f"{feature_name} ≤ {threshold_val:.3f}<br>Samples: {samples}")
                node_color.append(predicted_class)
                
                class_probs = value[node][0] / np.sum(value[node][0])
                prob_text = "<br>".join([f"{cls}: {prob:.3f}" for cls, prob in zip(class_names, class_probs)])
                node_info.append(f"Split: {feature_name} ≤ {threshold_val:.3f}<br>"
                               f"Samples: {samples}<br>"
                               f"Impurity: {impurity_val:.3f}<br>"
                               f"Class Probabilities:<br>{prob_text}")
            else:  # Regression
                predicted_value = value[node][0][0]
                node_text.append(f"{feature_name} ≤ {threshold_val:.3f}<br>Samples: {samples}")
                node_color.append(predicted_value)
                node_info.append(f"Split: {feature_name} ≤ {threshold_val:.3f}<br>"
                               f"Samples: {samples}<br>"
                               f"MSE: {impurity_val:.3f}<br>"
                               f"Predicted Value: {predicted_value:.3f}")
    
    # Create the plot
    fig = go.Figure()
    
    # Add edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(color='rgb(125,125,125)', width=2),
        hoverinfo='none',
        showlegend=False
    ))
    
    # Add nodes
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        marker=dict(
            size=20,
            color=node_color,
            colorscale='Viridis' if class_names is None else 'Set3',
            line=dict(width=2, color='black'),
            showscale=True,
            colorbar=dict(title="Predicted Value" if class_names is None else "Class")
        ),
        text=node_text,
        textposition="middle center",
        textfont=dict(size=8, color='white'),
        hovertext=node_info,
        hoverinfo='text',
        showlegend=False
    ))
    
    # Update layout
    fig.update_layout(
        title="Interactive Decision Tree Visualization<br><sub>Hover over nodes for details • Left branch = True, Right branch = False</sub>",
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=80),
        annotations=[
            dict(
                text="",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor='left', yanchor='bottom',
                font=dict(color='black', size=12)
            )
        ],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white',
        width=800,
        height=600
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
            coefficients = np.concatenate([[model.intercept_], model.coef_])
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
    y_pred_proba = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None
    
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='binary', zero_division=0)
    recall = recall_score(y, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y, y_pred, average='binary', zero_division=0)
    
    # ROC AUC if probabilities are available
    roc_auc = roc_auc_score(y, y_pred_proba) if y_pred_proba is not None else None
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
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
            numeric_columns = current_df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_columns) < 2:
                st.error("❌ Please upload a dataset with at least 2 numeric columns for regression analysis.")
                return
            
            # Update numeric columns for the final filtered data
            numeric_columns = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
            
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
            
            # Detect if dependent variable is binary (for classification)
            is_binary = len(df_filtered[dependent_var].dropna().unique()) == 2 if dependent_var else False
            
            # Method categories - always include all methods
            method_options = ["OLS", "Logistic Regression", "Decision Tree", "Random Forest", "Lasso", "Ridge", "Elastic Net"]
            
            if is_binary:
                st.sidebar.info("🎯 Binary dependent variable detected - Classification methods recommended")
            
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
                
                max_depth = st.sidebar.selectbox(
                    "Maximum Depth",
                    [None, 3, 5, 10, 15, 20],
                    index=0,
                    help="Maximum depth of the tree (None = unlimited)"
                )
                
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
                    
                    # Prepare data for regression - handle missing values
                    y_raw = df_filtered[dependent_var]
                    X_raw = df_filtered[independent_vars]
                    
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
                    model_type = 'classification' if estimation_method in ['Logistic Regression', 'Decision Tree', 'Random Forest'] and is_binary else 'regression'
                    
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
                            st.metric("Accuracy", f"{stats_dict['accuracy']:.4f}")
                        with col2:
                            st.metric("Precision", f"{stats_dict['precision']:.4f}")
                        with col3:
                            st.metric("Recall", f"{stats_dict['recall']:.4f}")
                        with col4:
                            st.metric("F1-Score", f"{stats_dict['f1_score']:.4f}")
                        
                        if stats_dict['roc_auc'] is not None:
                            st.metric("ROC AUC", f"{stats_dict['roc_auc']:.4f}")
                    else:
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("R-squared", f"{stats_dict['r_squared']:.4f}")
                        with col2:
                            st.metric("Adj. R-squared", f"{stats_dict['adj_r_squared']:.4f}")
                        with col3:
                            st.metric("RMSE", f"{stats_dict['rmse']:.4f}")
                        with col4:
                            st.metric("Observations", stats_dict['n_obs'])
                    
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
                            st.plotly_chart(tree_fig, use_container_width=True)
                            
                            # Text representation
                            with st.expander("📄 Tree Rules (Text Format)"):
                                tree_rules = export_text(model, feature_names=independent_vars, max_depth=max_depth_display)
                                st.text(tree_rules)
                        
                        elif estimation_method == "Random Forest":
                            # For Random Forest, show feature importance plot and individual tree option
                            st.subheader("Feature Importance Plot")
                            importance_fig = create_forest_importance_plot(model, independent_vars)
                            st.plotly_chart(importance_fig, use_container_width=True)
                            
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
                            st.plotly_chart(individual_tree_fig, use_container_width=True)
                            
                            # Text representation of selected tree
                            with st.expander(f"📄 Tree {tree_index} Rules (Text Format)"):
                                tree_rules = export_text(model.estimators_[tree_index], feature_names=independent_vars, max_depth=max_depth_display)
                                st.text(tree_rules)
                    
                    else:
                        # Linear models - show coefficients table
                        st.write("**Regression Coefficients:**")
                        
                        coef_data = []
                        variable_names = (['Intercept'] if include_constant else []) + independent_vars
                        coefficients = (np.concatenate([[model.intercept_], model.coef_]) if include_constant 
                                       else model.coef_)
                        
                        for i, var_name in enumerate(variable_names):
                            coef_entry = {
                                'Variable': var_name,
                                'Coefficient': coefficients[i]
                            }
                            
                            # Add statistical tests only for OLS
                            if estimation_method == "OLS":
                                coef_entry.update({
                                    'Std Error': stats_dict['std_errors'][i],
                                    't-statistic': stats_dict['t_stats'][i],
                                    'P-value': stats_dict['p_values'][i],
                                    'Significance': '***' if stats_dict['p_values'][i] < 0.01 else 
                                                  '**' if stats_dict['p_values'][i] < 0.05 else 
                                                  '*' if stats_dict['p_values'][i] < 0.1 else ''
                                })
                            else:
                                # For regularized methods, show if coefficient was shrunk to zero
                                coef_entry['Status'] = 'Selected' if abs(coefficients[i]) > 1e-10 else 'Excluded'
                            
                            coef_data.append(coef_entry)
                        
                        coef_df = pd.DataFrame(coef_data)
                        st.dataframe(coef_df, use_container_width=True)
                    
                    if estimation_method == "OLS":
                        st.caption("Significance levels: *** p<0.01, ** p<0.05, * p<0.1")
                        # F-statistic
                        st.write(f"**F-statistic:** {stats_dict['f_stat']:.4f} (p-value: {stats_dict['f_p_value']:.4f})")
                    else:
                        st.caption("Regularized methods don't provide traditional statistical significance tests")
                        # Show cross-validation score if desired
                        try:
                            if estimation_method in ["Lasso", "Ridge", "Elastic Net"]:
                                cv_scores = cross_val_score(model, X_for_plotting, y, cv=5, scoring='r2')
                                st.write(f"**Cross-Validation R² Score:** {cv_scores.mean():.4f} (±{cv_scores.std()*2:.4f})")
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
                        
                        if include_constant:
                            interpretation_text += f"{model.intercept_:.4f}"
                        
                        for i, var in enumerate(independent_vars):
                            if include_constant:
                                sign = "+" if model.coef_[i] >= 0 else ""
                                interpretation_text += f" {sign} {model.coef_[i]:.4f} × {var}"
                            else:
                                if i == 0:
                                    interpretation_text += f"{model.coef_[i]:.4f} × {var}"
                                else:
                                    sign = "+" if model.coef_[i] >= 0 else ""
                                    interpretation_text += f" {sign} {model.coef_[i]:.4f} × {var}"
                        
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
                                coef = model.coef_[i]
                                p_val = stats_dict['p_values'][i + 1]  # +1 because intercept is first
                                
                                significance = ""
                                if p_val < 0.01:
                                    significance = " (highly significant)"
                                elif p_val < 0.05:
                                    significance = " (significant)"
                                elif p_val < 0.1:
                                    significance = " (marginally significant)"
                                else:
                                    significance = " (not significant)"
                                
                                direction = "increases" if coef > 0 else "decreases"
                                insights.append(f"• A one-unit increase in {var} is associated with a {abs(coef):.4f} unit {direction} in {dependent_var}{significance}")
                        
                        else:
                            # Regularized methods insights
                            selected_vars = [var for i, var in enumerate(independent_vars) if abs(model.coef_[i]) > 1e-10]
                            excluded_vars = [var for i, var in enumerate(independent_vars) if abs(model.coef_[i]) <= 1e-10]
                            
                            if selected_vars:
                                insights.append(f"• {estimation_method} selected {len(selected_vars)} out of {len(independent_vars)} variables: {', '.join(selected_vars)}")
                            if excluded_vars:
                                insights.append(f"• Variables excluded by regularization: {', '.join(excluded_vars)}")
                            
                            for i, var in enumerate(independent_vars):
                                coef = model.coef_[i]
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