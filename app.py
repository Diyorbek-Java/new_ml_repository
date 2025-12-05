import streamlit as st
import pandas as pd

st.set_page_config(page_title="Disaster Prediction", page_icon="🌪️", layout="wide")

st.title("🌪️ Disaster Type Prediction System")
st.markdown("### Machine Learning Coursework Project")

st.markdown("---")

# Overview
st.header("📋 What This System Does")
st.write("""
This system predicts the type of disaster (Hurricane, Flood, Fire, etc.) using machine learning.

**Input:** Location + Emergency Programs Activated
**Output:** Predicted Disaster Type + Confidence Level
""")

st.markdown("---")

# Dataset Info
st.header("📊 Dataset Information")

try:
    df = pd.read_csv('data/disaster_declarations.csv')

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Disaster Types", df['incidentType'].nunique())
    col3.metric("States Covered", df['state'].nunique())
    col4.metric("Years of Data", df['fyDeclared'].max() - df['fyDeclared'].min() if 'fyDeclared' in df.columns else 71)

    st.info("**Source:** FEMA OpenFEMA - Disaster Declarations Summaries")

except FileNotFoundError:
    st.warning("Dataset not found")

st.markdown("---")

# Features
st.header("⚙️ System Features")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📊 Data Explorer")
    st.write("View dataset statistics and patterns")

    st.subheader("🤖 Model Training")
    st.write("Train Random Forest, Gradient Boosting, Logistic Regression")

with col2:
    st.subheader("📈 Model Evaluation")
    st.write("Check accuracy, confusion matrix, metrics")

    st.subheader("🔮 Make Predictions")
    st.write("Predict disaster type from location and programs")

with col3:
    st.subheader("🔧 Preprocessing")
    st.write("Data cleaning and feature engineering")

    st.subheader("💾 Results Export")
    st.write("Save predictions and reports")

st.markdown("---")

# Model Info
st.header("🤖 Model Details")

col1, col2 = st.columns(2)

with col1:
    st.write("**Algorithm:** Random Forest Classifier")
    st.write("**Accuracy:** ~89%")
    st.write("**Disaster Types:** 27 categories")

with col2:
    st.write("**Features Used:**")
    st.write("- State, Region, County")
    st.write("- Declaration Type")
    st.write("- Assistance Programs (IA, PA, HM, IH)")

st.success("✅ Model trained and ready")

st.markdown("---")

# How to Use
st.header("🚀 How to Use")

st.write("""
**Navigation:** Use sidebar on the left

1. **Data Explorer** → View dataset
2. **Preprocessing** → See data preparation
3. **Model Training** → Train models
4. **Evaluation** → Check performance
5. **Prediction** → Make predictions

**To Predict:**
- Go to Prediction page
- Select state and region
- Choose programs activated
- Click Predict button
""")

st.markdown("---")
st.caption("Machine Learning Coursework | WIUT | Random Forest Model (89% Accuracy)")
