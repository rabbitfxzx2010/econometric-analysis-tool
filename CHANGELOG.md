# Econometric Analysis Tool - Version History

## Version 2.1.0 - September 4, 2025

### 🌳 Major Feature: Cost Complexity Pruning for Tree Methods
- **Added Algorithm 8.1 Implementation**: Complete cost complexity pruning algorithm for Decision Trees and Random Forest
- **Cross-Validation Pruning**: Automatic alpha selection using K-fold cross-validation
- **Manual Alpha Setting**: Option for manual cost complexity parameter setting
- **Pruning Visualization**: Interactive plots showing pruning path and cross-validation scores
- **Pruned Tree Display**: Tree visualizations now show the pruned tree structure

### 🔧 Regularization Improvements
- **Expanded Alpha Range**: Users can now input regularization parameters beyond the 0-10 slider range
- **Flexible Parameter Input**: Choice between slider (0.001-10) or number input (any positive value)
- **Improved R² Calculation**: More accurate R² calculation for regularized models using sklearn's score method
- **Better Parameter Control**: Enhanced user interface for regularization parameter selection

### 🎯 Binary Classification Enhancements
- **Conditional Probability Controls**: Probability selection only appears for binary dependent variables
- **Sidebar Integration**: Moved probability controls to sidebar for cleaner interface
- **Smart Detection**: Automatic detection of binary vs. multi-class variables

### 🔒 Privacy and User Experience
- **Hidden Usage Tracking**: Usage statistics now invisible to regular users, only visible to creator
- **Improved UI**: Cleaner interface with better organization of controls
- **Enhanced Tree Visualization**: Better node sizing, hover functionality, and probability display

### 📊 Technical Improvements
- **Algorithm Compliance**: Implementation follows standard Algorithm 8.1 for regression tree building
- **Cross-Validation Integration**: Proper K-fold CV for parameter selection
- **Performance Optimization**: Better handling of large trees and pruning paths
- **Error Handling**: Improved robustness for edge cases in pruning algorithm

---

## Version 2.0.0 - September 2, 2025

### 🌲 Enhanced Decision Tree Visualization
- **Interactive Tree Plots**: Level-based node sizing with progressive reduction
- **Improved Node Display**: Larger nodes with probability and percentage information
- **Color Mapping**: Enhanced color schemes for better visualization
- **Hover Functionality**: Detailed information on hover for each node
- **Smart Sizing**: Automatic adjustment based on tree depth and complexity

### 🎯 Binary Classification Features
- **Class Probability Selection**: Choose between Class 0 and Class 1 probability display
- **Binary Detection**: Automatic detection of binary classification problems
- **User Guidance**: Clear labeling for probability selection options

### 🔧 Technical Fixes
- **Plotly ColorBar**: Fixed invalid 'titleside' property errors
- **Button Configuration**: Improved plot controls and removed problematic options
- **Timezone Support**: Added US Central Time support via pytz
- **Git Synchronization**: Maintained sync between main and development branches

---

## Version 1.0.0 - Initial Release

### 📊 Core Econometric Methods
- **Linear Regression**: OLS with comprehensive statistics
- **Regularized Methods**: Lasso, Ridge, and Elastic Net regression
- **Tree Methods**: Decision Trees and Random Forest
- **Classification**: Logistic regression and tree-based classification

### 📈 Visualization Features
- **Interactive Plots**: Plotly-based visualizations
- **Statistical Displays**: Comprehensive regression statistics
- **Data Exploration**: Missing value analysis and variable selection

### 🎛️ User Interface
- **Streamlit Framework**: Clean, interactive web interface
- **Parameter Controls**: Sidebar controls for method parameters
- **File Upload**: Support for CSV and Excel files
- **Export Options**: Download capabilities for results and visualizations

---

## Development Notes

### Next Planned Features
- Cross-validation for all methods
- Model comparison tools
- Advanced diagnostic plots
- Time series analysis capabilities

### Technical Stack
- **Backend**: Python, scikit-learn, pandas, numpy
- **Frontend**: Streamlit
- **Visualization**: Plotly, matplotlib, seaborn
- **Version Control**: Git with main/development branch strategy

### Contributing
This tool is under active development. For feature requests or bug reports, please contact the development team.

---

*Last Updated: September 4, 2025*
*Version: 2.1.0*
