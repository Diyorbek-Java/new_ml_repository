# 🌪️ Disaster Declaration Analysis System

A comprehensive Machine Learning system for predicting FEMA disaster declaration requirements and assistance program needs based on historical disaster data.

## 📋 Project Overview

This project uses machine learning algorithms to analyze FEMA disaster declaration data and predict whether federal assistance will be required for disaster incidents. The system provides an interactive web interface built with Streamlit for data exploration, model training, evaluation, and real-time predictions.

## ✨ Features

- **📊 Interactive Data Exploration**: Comprehensive visualizations and statistical analysis of disaster patterns
- **🔧 Advanced Preprocessing**: Data cleaning, feature engineering, and transformation pipelines
- **🤖 Multiple ML Models**: Support for Random Forest, Logistic Regression, Decision Trees, SVM, and more
- **📈 Detailed Model Evaluation**: Confusion matrices, ROC curves, feature importance, and cross-validation
- **🔮 Real-time Predictions**: User-friendly interface for predicting disaster declaration requirements
- **📱 Responsive Design**: Beautiful, modern UI with custom styling and animations

## 🗂️ Project Structure

```
ML/
├── app.py                          # Main Streamlit application (home page)
├── pages/
│   ├── 1_📊_Data_Exploration.py   # Data visualization and analysis
│   ├── 2_🔧_Preprocessing.py       # Data preprocessing documentation
│   ├── 3_🤖_Model_Training.py      # Model training interface
│   ├── 4_📈_Model_Evaluation.py    # Model evaluation and metrics
│   └── 5_🔮_Prediction.py          # Prediction interface
├── data/
│   └── disaster_declarations.csv   # FEMA disaster declarations dataset
├── models/
│   ├── disaster_model.pkl          # Trained model (generated)
│   └── ...                         # Other model artifacts
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone or download this repository

2. Navigate to the project directory:
```bash
cd ML
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

### Running the Application

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. The application will open in your default web browser at `http://localhost:8501`

3. Navigate through the pages using the sidebar:
   - **Home**: Project overview and information
   - **📊 Data Exploration**: Explore the dataset
   - **🔧 Preprocessing**: View preprocessing steps
   - **🤖 Model Training**: Train ML models
   - **📈 Model Evaluation**: Evaluate model performance
   - **🔮 Prediction**: Make predictions on new data

## 📊 Dataset

The project uses the **FEMA Disaster Declarations Dataset** from OpenFEMA, which includes:

- **70,000+** disaster declaration records
- **28** features including geographic, temporal, and disaster-specific information
- **Multiple incident types**: Fires, severe storms, hurricanes, floods, earthquakes, etc.
- **Assistance programs**: Individual Assistance (IA), Public Assistance (PA), Hazard Mitigation (HM)

### Key Features:
- `state`: US state where disaster occurred
- `incidentType`: Type of disaster (fire, flood, hurricane, etc.)
- `declarationType`: DR (Major Disaster), EM (Emergency), FM (Fire Management)
- `fyDeclared`: Fiscal year of declaration
- `ihProgramDeclared`: Individual & Households program flag
- `iaProgramDeclared`: Individual Assistance program flag
- `paProgramDeclared`: Public Assistance program flag
- `hmProgramDeclared`: Hazard Mitigation program flag
- `region`: FEMA administrative region (1-10)
- `tribalRequest`: Whether request is from tribal government

## 🤖 Machine Learning Models

The system supports multiple machine learning algorithms:

| Model | Description | Key Strengths |
|-------|-------------|---------------|
| **Random Forest** | Ensemble of decision trees | High accuracy, feature importance, handles non-linearity |
| **Logistic Regression** | Linear classification | Fast, interpretable, probabilistic outputs |
| **Decision Tree** | Tree-based classifier | Interpretable, handles categorical data well |
| **SVM** | Support Vector Machine | Effective in high dimensions, robust |
| **Naive Bayes** | Probabilistic classifier | Fast training, works well with small datasets |

### Model Performance

The best-performing model typically achieves:
- **Accuracy**: ~87-92%
- **Precision**: ~85-90%
- **Recall**: ~82-88%
- **F1-Score**: ~85-89%
- **AUC-ROC**: ~0.90-0.95

## 🔧 Data Preprocessing Pipeline

1. **Data Loading**: Load raw FEMA disaster data
2. **Missing Value Handling**: Impute or remove missing data
3. **Feature Selection**: Select relevant features for prediction
4. **Categorical Encoding**: Label encoding for high-cardinality features
5. **Feature Engineering**: Create new features from existing data
6. **Feature Scaling**: Standardize numerical features
7. **Train-Test Split**: 80-20 stratified split

## 📈 Model Evaluation Metrics

The system provides comprehensive evaluation:

- **Confusion Matrix**: Visual representation of prediction performance
- **Classification Report**: Precision, recall, F1-score for each class
- **ROC Curve**: True positive rate vs false positive rate
- **AUC-ROC Score**: Area under the ROC curve
- **Precision-Recall Curve**: Trade-off between precision and recall
- **Cross-Validation**: 5-fold CV for robust performance estimation
- **Feature Importance**: Understanding which features drive predictions

## 🔮 Making Predictions

### Manual Prediction

1. Navigate to the **🔮 Prediction** page
2. Fill in the disaster information:
   - Location (State, FEMA Region)
   - Incident details (Type, Declaration Type, Year)
   - Assistance programs requested
   - Tribal request status
3. Click **Predict Declaration Outcome**
4. View prediction results, confidence scores, and recommendations

### Batch Prediction

Upload a CSV file with multiple disaster records for batch processing (feature in development).

## 📸 Screenshots

### Home Page
Beautiful landing page with project overview and feature highlights

### Data Exploration
Interactive charts showing:
- Disaster type distributions
- State-wise analysis
- Temporal trends
- Program activation rates
- Correlation matrices

### Model Training
- Configure hyperparameters
- Select models to train
- Real-time training progress
- Model comparison charts

### Model Evaluation
- Detailed performance metrics
- Confusion matrix heatmap
- ROC and Precision-Recall curves
- Feature importance analysis
- Cross-validation results

### Prediction Interface
- User-friendly input forms
- Real-time prediction results
- Risk assessment visualization
- Actionable recommendations

## 🛠️ Technologies Used

- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning algorithms and tools
- **Plotly**: Interactive visualizations
- **XGBoost**: Gradient boosting (optional)

## 📝 Usage Tips

1. **Start with Data Exploration**: Understand the dataset before training models
2. **Review Preprocessing**: See how data is prepared for modeling
3. **Train Multiple Models**: Compare different algorithms to find the best performer
4. **Evaluate Thoroughly**: Check all metrics, not just accuracy
5. **Make Informed Predictions**: Use the trained model for new disaster scenarios

## 🎯 Use Cases

- **Emergency Management**: Predict resource requirements for disasters
- **Federal Agencies**: Assess declaration needs before formal requests
- **State Governments**: Understand likelihood of federal assistance
- **Research**: Analyze disaster patterns and trends
- **Planning**: Prepare for disaster response based on historical data

## 🔒 Data Privacy & Ethics

- All data is from public FEMA OpenFEMA datasets
- No personal information is collected or stored
- Predictions are for planning purposes only
- Actual disaster declarations require formal assessment by FEMA officials

## 🤝 Contributing

This is an academic project for ML coursework. For suggestions or improvements:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is created for educational purposes as part of ML coursework.

## 👥 Authors

**WIUT ML Coursework Project**

## 🙏 Acknowledgments

- **FEMA OpenFEMA**: For providing the disaster declarations dataset
- **Streamlit**: For the amazing web app framework
- **Scikit-learn**: For comprehensive ML tools
- **Plotly**: For interactive visualizations

## 📞 Support

For questions or issues:
- Review the documentation in each page of the app
- Check the preprocessing and training pages for guidance
- Ensure all dependencies are installed correctly

## 🚀 Future Enhancements

- [ ] Real-time data updates from FEMA API
- [ ] Batch prediction functionality
- [ ] Model explainability with SHAP values
- [ ] Historical prediction tracking
- [ ] Export reports as PDF
- [ ] Integration with GIS mapping
- [ ] Mobile-responsive improvements
- [ ] Additional ML algorithms (Neural Networks, Ensemble methods)

## 📊 Project Status

✅ **Completed**:
- Data exploration and visualization
- Data preprocessing pipeline
- Multiple ML model training
- Comprehensive model evaluation
- Prediction interface
- Documentation

🔄 **In Progress**:
- Batch prediction from CSV
- Advanced feature engineering

## 🎓 Academic Information

**Course**: Machine Learning
**Institution**: Westminster International University in Tashkent (WIUT)
**Project Type**: Coursework
**Year**: 2024

---

<div align="center">
    <p><strong>🌪️ Disaster Declaration Analysis System</strong></p>
    <p>Powered by Machine Learning | Built with Streamlit</p>
    <p>Data Source: FEMA OpenFEMA Dataset</p>
</div>
