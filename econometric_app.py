import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
import openpyxl

# Set page configuration
st.set_page_config(
    page_title="Econometric Analysis Tool",
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

def calculate_regression_stats(X, y, model, method='OLS'):
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
    
    # Adjusted R-squared
    adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k - 1)
    
    # Mean Squared Error and Root Mean Squared Error
    mse = ss_res / (n - k - 1) if method == 'OLS' else ss_res / n
    rmse = np.sqrt(mse)
    
    # Calculate residuals
    residuals = y - y_pred
    
    # For OLS, calculate standard errors and statistical tests
    if method == 'OLS':
        # Standard errors of coefficients
        X_with_intercept = np.column_stack([np.ones(n), X])
        
        try:
            # Variance-covariance matrix
            var_cov_matrix = mse * np.linalg.inv(X_with_intercept.T @ X_with_intercept)
            std_errors = np.sqrt(np.diag(var_cov_matrix))
            
            # T-statistics
            coefficients = np.concatenate([[model.intercept_], model.coef_])
            t_stats = coefficients / std_errors
            
            # P-values (two-tailed test)
            p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - k - 1))
            
        except np.linalg.LinAlgError:
            # If matrix is singular, return NaN values
            std_errors = np.full(k + 1, np.nan)
            t_stats = np.full(k + 1, np.nan)
            p_values = np.full(k + 1, np.nan)
        
        # F-statistic
        f_stat = (r_squared / k) / ((1 - r_squared) / (n - k - 1))
        f_p_value = 1 - stats.f.cdf(f_stat, k, n - k - 1)
    else:
        # For regularized methods, we don't calculate traditional statistical tests
        std_errors = np.full(k + 1, np.nan)
        t_stats = np.full(k + 1, np.nan)
        p_values = np.full(k + 1, np.nan)
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

def fit_model(X, y, method, alpha=1.0, l1_ratio=0.5):
    """Fit model based on selected method"""
    if method == 'OLS':
        model = LinearRegression()
    elif method == 'Lasso':
        model = Lasso(alpha=alpha, random_state=42)
    elif method == 'Ridge':
        model = Ridge(alpha=alpha, random_state=42)
    elif method == 'Elastic Net':
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    model.fit(X, y)
    return model

def main():
    # Main header
    st.markdown('<h1 class="main-header">📊 Econometric Analysis Tool</h1>', unsafe_allow_html=True)
    st.markdown("Upload your data, select variables, and perform OLS regression analysis with comprehensive statistics and visualizations.")
    
    # Initialize default values for variables used in main area
    show_plots = False
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
            
            # Main content area
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
                
                # Variable Plotting Display Area
                if show_plots and plot_var != 'None' and plot_type:
                    st.markdown('<h3 class="subheader">📈 Variable Visualization</h3>', unsafe_allow_html=True)
                    
                    current_df = df_filtered if 'df_filtered' in locals() else df
            
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
                
                # Plot Display Area (plots will appear here when variables are selected on the left)
                if 'plot_var' in locals() and 'plot_type' in locals() and plot_var != 'None' and plot_type:
                    st.markdown('<h3 class="subheader">� Variable Plot</h3>', unsafe_allow_html=True)
                    
                    if plot_type == "Histogram":
                        fig = px.histogram(
                            current_df, 
                            x=plot_var,
                            title=f"Distribution of {plot_var}",
                            nbins=30
                        )
                        fig.update_layout(
                            xaxis_title=plot_var,
                            yaxis_title="Frequency"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Add basic statistics
                        stats_col1, stats_col2 = st.columns(2)
                        with stats_col1:
                            st.metric("Mean", f"{current_df[plot_var].mean():.2f}")
                            st.metric("Median", f"{current_df[plot_var].median():.2f}")
                        with stats_col2:
                            st.metric("Std Dev", f"{current_df[plot_var].std():.2f}")
                            st.metric("Range", f"{current_df[plot_var].max() - current_df[plot_var].min():.2f}")
                    
                    elif plot_type == "Box Plot":
                        fig = px.box(
                            current_df, 
                            y=plot_var,
                            title=f"Box Plot of {plot_var}"
                        )
                        fig.update_layout(yaxis_title=plot_var)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Outlier information
                        Q1 = current_df[plot_var].quantile(0.25)
                        Q3 = current_df[plot_var].quantile(0.75)
                        IQR = Q3 - Q1
                        outliers = current_df[(current_df[plot_var] < Q1 - 1.5*IQR) | (current_df[plot_var] > Q3 + 1.5*IQR)]
                        st.info(f"📊 Potential outliers detected: {len(outliers)} observations")
                    
                    elif plot_type == "Line Plot (Multiple Variables)":
                        if len(current_df) > 1:
                            # Create a simple index-based line plot
                            fig = px.line(
                                x=current_df.index, 
                                y=current_df[plot_var],
                                title=f"Line Plot of {plot_var} (Index Order)"
                            )
                            fig.update_layout(
                                xaxis_title="Observation Index",
                                yaxis_title=plot_var
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("Need more than one observation for line plot")            # Get numeric columns only from the current dataset
            current_df = df_filtered if 'df_filtered' in locals() else df
            numeric_columns = current_df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_columns) < 2:
                st.error("❌ Please upload a dataset with at least 2 numeric columns for regression analysis.")
                return
            
            # Sample selection section
            st.sidebar.markdown("---")
            st.sidebar.header("🎯 Sample Selection")
            
            # Option to filter observations
            use_sample_filter = st.sidebar.checkbox(
                "Filter Sample Observations",
                help="Check this to select specific observations for analysis"
            )
            
            if use_sample_filter:
                # Let user choose filtering method
                filter_method = st.sidebar.radio(
                    "Select Filtering Method:",
                    ["Row Range", "Condition Filter"],
                    help="Choose how to filter your data"
                )
                
                if filter_method == "Row Range":
                    start_row = st.sidebar.number_input(
                        "Start Row (1-indexed)",
                        min_value=1,
                        max_value=len(df),
                        value=1,
                        help="First row to include in analysis"
                    )
                    end_row = st.sidebar.number_input(
                        "End Row (1-indexed)",
                        min_value=start_row,
                        max_value=len(df),
                        value=len(df),
                        help="Last row to include in analysis"
                    )
                    # Filter dataframe by row range (convert to 0-indexed)
                    df_filtered = df.iloc[start_row-1:end_row].copy()
                    
                else:  # Condition Filter
                    filter_column = st.sidebar.selectbox(
                        "Filter Column",
                        df.columns.tolist(),
                        help="Column to apply filter condition on"
                    )
                    
                    if df[filter_column].dtype in ['object', 'string']:
                        # Categorical filtering
                        unique_values = df[filter_column].unique()
                        selected_values = st.sidebar.multiselect(
                            f"Select values for {filter_column}",
                            unique_values,
                            default=unique_values.tolist(),
                            help="Choose which values to include"
                        )
                        df_filtered = df[df[filter_column].isin(selected_values)].copy()
                    else:
                        # Numeric filtering
                        min_val = float(df[filter_column].min())
                        max_val = float(df[filter_column].max())
                        
                        value_range = st.sidebar.slider(
                            f"Range for {filter_column}",
                            min_value=min_val,
                            max_value=max_val,
                            value=(min_val, max_val),
                            help="Select the range of values to include"
                        )
                        df_filtered = df[
                            (df[filter_column] >= value_range[0]) & 
                            (df[filter_column] <= value_range[1])
                        ].copy()
                
                st.sidebar.info(f"Filtered sample: {len(df_filtered)} rows")
                
            # Missing value handling
            st.sidebar.markdown("---")
            st.sidebar.header("🔧 Missing Value Handling")
            
            missing_method = st.sidebar.selectbox(
                "How to handle missing values?",
                ["Listwise Deletion", "Mean Imputation", "Median Imputation", "Mode Imputation", "KNN Imputation"],
                help="Choose how to handle missing data"
            )
            
            # Data Visualization section
            st.sidebar.markdown("---")
            st.sidebar.header("📊 Data Visualization")
            
            show_plots = st.sidebar.checkbox(
                "Show data plots",
                value=False,
                help="Enable this to show variable visualization plots in the main area"
            )
            
            if show_plots:
                # Select variable to plot
                current_df = df_filtered if 'df_filtered' in locals() else df
                plot_var = st.sidebar.selectbox(
                    "Choose a variable to plot:",
                    ['None'] + current_df.select_dtypes(include=[np.number]).columns.tolist(),
                    help="Select a numeric variable to visualize"
                )
                
                if plot_var != 'None':
                    # Plot type selection
                    plot_type = st.sidebar.selectbox(
                        "Plot type:",
                        ["Histogram", "Box Plot", "Line Plot (Multiple Variables)"],
                        help="Choose how to visualize the variable"
                    )
                else:
                    plot_type = None
            else:
                plot_var = 'None'
                plot_type = None
                
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
            
            # Independent variables selection
            available_independent = [col for col in numeric_columns if col != dependent_var]
            independent_vars = st.sidebar.multiselect(
                "Select Independent Variables (X)",
                available_independent,
                default=available_independent[:1] if available_independent else [],
                help="Choose one or more variables to use as predictors"
            )
            
            # Estimation method selection
            st.sidebar.markdown("---")
            st.sidebar.header("⚙️ Estimation Method")
            
            estimation_method = st.sidebar.selectbox(
                "Choose Estimation Method",
                ["OLS", "Lasso", "Ridge", "Elastic Net"],
                help="Select the regression method to use"
            )
            
            # Method-specific parameters
            if estimation_method in ["Lasso", "Ridge", "Elastic Net"]:
                alpha = st.sidebar.slider(
                    "Regularization Strength (α)",
                    min_value=0.001,
                    max_value=10.0,
                    value=1.0,
                    step=0.001,
                    help="Higher values increase regularization"
                )
                
                if estimation_method == "Elastic Net":
                    l1_ratio = st.sidebar.slider(
                        "L1 Ratio (Lasso vs Ridge)",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.5,
                        step=0.01,
                        help="0 = Ridge, 1 = Lasso, 0.5 = Equal mix"
                    )
                else:
                    l1_ratio = 0.5
            else:
                alpha = 1.0
                l1_ratio = 0.5
            
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
                    
                    # For regularized methods, standardize features
                    if estimation_method in ["Lasso", "Ridge", "Elastic Net"]:
                        scaler = StandardScaler()
                        X_scaled = pd.DataFrame(
                            scaler.fit_transform(X), 
                            columns=X.columns, 
                            index=X.index
                        )
                        # Fit the model on scaled data
                        model = fit_model(X_scaled, y, estimation_method, alpha, l1_ratio)
                        # Calculate stats on scaled data
                        stats_dict = calculate_regression_stats(X_scaled, y, model, estimation_method)
                        X_for_plotting = X_scaled  # Use scaled data for plotting
                    else:
                        # Fit the model on original data
                        model = fit_model(X, y, estimation_method, alpha, l1_ratio)
                        # Calculate stats on original data
                        stats_dict = calculate_regression_stats(X, y, model, estimation_method)
                        X_for_plotting = X  # Use original data for plotting
                    
                    # Display results
                    st.markdown(f'<h2 class="subheader">📈 {estimation_method} Regression Results</h2>', unsafe_allow_html=True)
                    
                    # Method-specific information
                    if estimation_method != "OLS":
                        method_info = {
                            "Lasso": "L1 regularization - promotes sparsity by setting some coefficients to zero",
                            "Ridge": "L2 regularization - shrinks coefficients towards zero but keeps all variables",
                            "Elastic Net": "Combines L1 and L2 regularization for balanced variable selection and shrinkage"
                        }
                        st.info(f"**{estimation_method}**: {method_info[estimation_method]}")
                        if estimation_method in ["Lasso", "Ridge", "Elastic Net"]:
                            st.info(f"**Regularization Parameters**: α = {alpha}" + 
                                   (f", L1 ratio = {l1_ratio}" if estimation_method == "Elastic Net" else ""))
                    
                    # Model summary
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("R-squared", f"{stats_dict['r_squared']:.4f}")
                    with col2:
                        st.metric("Adj. R-squared", f"{stats_dict['adj_r_squared']:.4f}")
                    with col3:
                        st.metric("RMSE", f"{stats_dict['rmse']:.4f}")
                    with col4:
                        st.metric("Observations", stats_dict['n_obs'])
                    
                    # Coefficients table
                    st.write("**Regression Coefficients:**")
                    
                    coef_data = []
                    variable_names = ['Intercept'] + independent_vars
                    coefficients = np.concatenate([[model.intercept_], model.coef_])
                    
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
                    tab1, tab2, tab3, tab4 = st.tabs(["Scatter Plot", "Residuals vs Fitted", "Q-Q Plot", "Histogram of Residuals"])
                    
                    with tab1:
                        # Scatter plot of Y vs first independent variable with regression line
                        first_var = independent_vars[0]
                        
                        fig = px.scatter(
                            x=X[first_var],  # Use original data for plotting
                            y=y,
                            labels={'x': first_var, 'y': dependent_var},
                            title=f"Relationship between {dependent_var} and {first_var} ({estimation_method})"
                        )
                        
                        # Add regression line
                        x_range = np.linspace(X[first_var].min(), X[first_var].max(), 100)
                        if len(independent_vars) == 1:
                            # Simple regression
                            if estimation_method in ["Lasso", "Ridge", "Elastic Net"]:
                                # For regularized methods, we need to scale the x_range
                                x_range_scaled = scaler.transform(x_range.reshape(-1, 1)).flatten()
                                y_line = model.intercept_ + model.coef_[0] * x_range_scaled
                            else:
                                y_line = model.intercept_ + model.coef_[0] * x_range
                        else:
                            # Multiple regression - fix other variables at their means
                            if estimation_method in ["Lasso", "Ridge", "Elastic Net"]:
                                X_line = np.zeros((100, len(independent_vars)))
                                X_line[:, 0] = (x_range - X[first_var].mean()) / X[first_var].std()  # Scale first variable
                                for i in range(1, len(independent_vars)):
                                    X_line[:, i] = 0  # Scaled mean is 0
                                y_line = model.predict(X_line)
                            else:
                                X_line = np.zeros((100, len(independent_vars)))
                                X_line[:, 0] = x_range
                                for i in range(1, len(independent_vars)):
                                    X_line[:, i] = X.iloc[:, i].mean()
                                y_line = model.predict(X_line)
                        
                        fig.add_trace(go.Scatter(
                            x=x_range, 
                            y=y_line,
                            mode='lines',
                            name=f'{estimation_method} Line',
                            line=dict(color='red', width=2)
                        ))
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
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
                    
                    interpretation_text = f"""
                    **Model Equation ({estimation_method}):**
                    {dependent_var} = {model.intercept_:.4f}"""
                    
                    for i, var in enumerate(independent_vars):
                        sign = "+" if model.coef_[i] >= 0 else ""
                        interpretation_text += f" {sign} {model.coef_[i]:.4f} × {var}"
                    
                    st.write(interpretation_text)
                    
                    st.write("**Key Insights:**")
                    insights = []
                    
                    # R-squared interpretation
                    r_sq_pct = stats_dict['r_squared'] * 100
                    insights.append(f"• The model explains {r_sq_pct:.1f}% of the variance in {dependent_var}")
                    
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
        
        ## Estimation Methods:
        
        - **OLS (Ordinary Least Squares)**: Traditional linear regression with statistical significance tests
        - **Lasso**: L1 regularization that can set coefficients to zero (variable selection)
        - **Ridge**: L2 regularization that shrinks coefficients toward zero
        - **Elastic Net**: Combines L1 and L2 regularization for balanced variable selection and shrinkage
        
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