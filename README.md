# 📊 Econometric Analysis Web Application

A comprehensive web-based tool for econometric analysis designed for educational use. Students can upload data, perform various regression analyses, and interpret results through an intuitive interface.

## 🌐 **Live Application**
**Access the tool:** [Coming Soon - Will be deployed on Streamlit Cloud]

## ✨ **Features**

### 📁 **Data Upload & Management**
- **Multi-format support**: CSV and Excel files (.xlsx, .xls)
- **Excel sheet selection**: Choose specific sheets from multi-sheet workbooks
- **Sample observation filtering**: 
  - Row range selection
  - Condition-based filtering (categorical and numeric)
- **Real-time data preview**: View filtered datasets instantly

### 🔧 **Missing Value Handling**
- **Listwise deletion**: Traditional complete case analysis
- **Mean imputation**: Replace missing values with column means
- **Median imputation**: Robust to outliers and skewed distributions
- **Mode imputation**: For categorical variables
- **KNN imputation**: Advanced method using k-nearest neighbors
- **Smart detection**: Automatic identification and reporting of missing values

### ⚙️ **Estimation Methods**
- **OLS (Ordinary Least Squares)**: Full statistical inference with t-tests, F-tests, p-values
- **Lasso Regression**: L1 regularization for variable selection
- **Ridge Regression**: L2 regularization to prevent overfitting  
- **Elastic Net**: Combined L1/L2 regularization with adjustable mixing ratio

### 📊 **Comprehensive Output**
- **Model statistics**: R², Adjusted R², RMSE, F-statistics
- **Coefficient analysis**: Estimates, standard errors, significance tests
- **Interactive visualizations**: Scatter plots, residual analysis, Q-Q plots
- **Method-specific insights**: 
  - OLS: Traditional statistical significance
  - Regularized methods: Variable selection results, cross-validation scores
- **Plain English interpretation**: Clear explanations of results

### 🎯 **Educational Features**
- **No coding required**: Point-and-click interface
- **Method comparison**: Easy switching between estimation techniques
- **Sample datasets**: Built-in examples for learning
- **Comprehensive help**: Tooltips and guidance throughout
- **Real-time feedback**: Immediate results and error handling

## 📚 **Sample Datasets**

Three sample datasets are included for learning and testing:

1. **`sample_data.csv`**: Basic income/education economic data
2. **`sample_data.xlsx`**: Multi-sheet Excel file with Economic_Data and Stock_Data
3. **`sample_data_with_missing.csv`**: Dataset with missing values for testing imputation methods

## 🚀 **For Instructors**

### **Quick Deployment**
This application is designed for easy deployment on Streamlit Cloud:

1. Fork/download this repository
2. Deploy to [Streamlit Cloud](https://share.streamlit.io)
3. Share the URL with students
4. No student setup required!

### **Customization**
- Modify estimation methods in `econometric_app.py`
- Add new sample datasets
- Customize the interface and explanations
- Add institution-specific branding

### **Student Experience**
- **Zero installation**: Works in any web browser
- **Immediate access**: No account creation required
- **Cross-platform**: Works on computers, tablets, phones
- **Always updated**: Students automatically get latest version

## 🛠 **Technical Details**

### **Requirements**
- Python 3.7+
- Streamlit
- pandas, numpy, scipy
- scikit-learn
- matplotlib, seaborn, plotly
- openpyxl (for Excel support)

### **Installation (for local development)**
```bash
pip install -r requirements.txt
streamlit run econometric_app.py
```

### **Architecture**
- **Frontend**: Streamlit web interface
- **Backend**: Python with pandas/scikit-learn
- **Deployment**: Streamlit Cloud (recommended)
- **Data handling**: In-memory processing (no data persistence)

## 📖 **Usage Guide**

### **For Students**
1. **Upload data**: Choose CSV or Excel file
2. **Select sheet**: Pick Excel worksheet (if applicable)
3. **Filter sample**: Optionally select specific observations
4. **Choose variables**: Select dependent and independent variables
5. **Handle missing values**: Pick appropriate method
6. **Select method**: Choose estimation technique
7. **Run analysis**: Click to perform regression
8. **Interpret results**: Review statistics and visualizations

### **For Instructors**
- **Demonstrate methods**: Show different estimation techniques
- **Compare approaches**: Illustrate impact of regularization
- **Discuss missing data**: Explore various handling strategies
- **Interactive teaching**: Real-time analysis during lectures

## 🎓 **Educational Applications**

### **Course Integration**
- **Introductory Econometrics**: Basic OLS concepts
- **Advanced Econometrics**: Regularization and model selection
- **Data Analysis**: Missing value handling and data preprocessing
- **Applied Economics**: Real-world data analysis projects

### **Learning Objectives**
- Understand different regression methods
- Learn impact of missing data handling
- Develop data analysis skills
- Interpret statistical output
- Compare model performance

## 🤝 **Contributing**

### **For Educators**
- Suggest new features or estimation methods
- Provide feedback on user interface
- Share sample datasets
- Report bugs or issues

### **For Developers**
- Fork the repository
- Create feature branches
- Submit pull requests
- Follow Python best practices

## 📞 **Support**

- **Issues**: Report bugs via GitHub Issues
- **Documentation**: See `DEPLOYMENT_GUIDE.md` for hosting options
- **Updates**: Watch repository for new features

## 📄 **License**

This project is open source and available under the MIT License.

---

## 🎯 **Quick Start for Students**

1. **Access the app**: [URL will be provided by instructor]
2. **Upload your data**: Click "Choose a CSV or Excel file"
3. **Follow the sidebar**: Select variables and options
4. **Run regression**: Click the regression button
5. **Interpret results**: Review the output and visualizations

**Need help?** Check the tooltips (ℹ️) throughout the interface or contact your instructor.

---

*Built with ❤️ for economics education using Streamlit and Python*
