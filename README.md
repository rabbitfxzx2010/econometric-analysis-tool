# Econometric Analysis Tool

A simple web application for performing econometric analysis with OLS regression. Upload your CSV data, select variables, and get comprehensive regression results with visualizations.

## Features

- **Data Upload**: Upload CSV files with your econometric data
- **Data Exploration**: View dataset overview, descriptive statistics, and column information
- **Variable Selection**: Choose dependent and independent variables through an intuitive interface
- **OLS Regression**: Perform ordinary least squares regression with comprehensive statistics
- **Statistical Output**: Get R-squared, adjusted R-squared, F-statistics, t-statistics, p-values, and more
- **Visualizations**: Interactive plots including scatter plots, residual analysis, Q-Q plots, and histograms
- **Model Interpretation**: Clear explanations of regression results

## Installation

1. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit application:
   ```bash
   streamlit run econometric_app.py
   ```

2. Open your web browser and navigate to the provided URL (typically http://localhost:8501)

3. Upload your CSV file using the sidebar file uploader

4. Explore your data and select variables for regression analysis

5. Click "Run OLS Regression" to perform the analysis

6. Review the results, statistics, and visualizations

## Data Format

Your CSV file should:
- Have column headers in the first row
- Contain numeric data for the variables you want to analyze
- Use standard CSV formatting (comma-separated values)

Example:
```csv
income,education,experience,age
50000,16,5,28
65000,18,8,32
45000,14,3,25
```

## Sample Data

A sample dataset (`sample_data.csv`) is included with fictional income, education, experience, age, and gender data for testing the application.

## Statistical Output

The application provides:

- **Coefficients Table**: Variable coefficients, standard errors, t-statistics, p-values, and significance levels
- **Model Fit Statistics**: R-squared, adjusted R-squared, RMSE, F-statistic
- **Diagnostic Plots**: 
  - Scatter plot with regression line
  - Residuals vs fitted values
  - Q-Q plot for normality testing
  - Histogram of residuals
- **Model Interpretation**: Plain English explanations of the results

## Requirements

- Python 3.7+
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- SciPy
- Plotly

## License

This project is open source and available under the MIT License.
