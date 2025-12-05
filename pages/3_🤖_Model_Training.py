import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import plotly.graph_objects as go
import plotly.express as px
import pickle
import joblib
import time
import os

st.set_page_config(page_title="Model Training", page_icon="🤖", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .content-box {
        background-color: white;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 20px 0;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="content-box">', unsafe_allow_html=True)
st.title("🤖 Model Training")
st.markdown("Train and compare multiple machine learning models")
st.markdown('</div>', unsafe_allow_html=True)

# Load and prepare data
@st.cache_data
def load_and_prepare_data():
    try:
        df = pd.read_csv('data/disaster_declarations.csv')

        # Select relevant columns (incidentType is now the target, not a feature)
        feature_cols = ['state', 'declarationType', 'fyDeclared',
                       'ihProgramDeclared', 'iaProgramDeclared', 'paProgramDeclared',
                       'hmProgramDeclared', 'tribalRequest', 'region']

        # Also need incidentType as target
        all_cols = feature_cols + ['incidentType']
        available_cols = [col for col in all_cols if col in df.columns]
        df_clean = df[available_cols].copy()

        # Remove missing values
        df_clean = df_clean.dropna()

        return df_clean
    except FileNotFoundError:
        return None

df = load_and_prepare_data()

if df is not None:
    # Training Configuration
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    st.header("⚙️ Training Configuration")

    col1, col2, col3 = st.columns(3)

    with col1:
        test_size = st.slider("Test Size (%)", 10, 40, 20, 5)
        test_size = test_size / 100

    with col2:
        random_state = st.number_input("Random State", 0, 100, 42)

    with col3:
        cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5)

    st.markdown('</div>', unsafe_allow_html=True)

    # Model Selection
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    st.header("📊 Select Models to Train")

    col1, col2, col3 = st.columns(3)

    with col1:
        train_rf = st.checkbox("Random Forest", value=True)
        train_lr = st.checkbox("Logistic Regression", value=True)

    with col2:
        train_dt = st.checkbox("Decision Tree", value=True)
        train_svm = st.checkbox("Support Vector Machine", value=False)

    with col3:
        train_nb = st.checkbox("Naive Bayes", value=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Hyperparameter Configuration
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    st.header("🎛️ Hyperparameter Configuration")

    with st.expander("Random Forest Parameters"):
        col1, col2, col3 = st.columns(3)
        with col1:
            rf_n_estimators = st.slider("Number of Trees", 50, 200, 100, 10)
        with col2:
            rf_max_depth = st.slider("Max Depth", 5, 30, 15, 5)
        with col3:
            rf_min_samples_split = st.slider("Min Samples Split", 2, 20, 10, 2)

    with st.expander("Logistic Regression Parameters"):
        col1, col2 = st.columns(2)
        with col1:
            lr_c = st.slider("Regularization (C)", 0.01, 10.0, 1.0, 0.1)
        with col2:
            lr_max_iter = st.slider("Max Iterations", 100, 1000, 200, 100)

    with st.expander("Decision Tree Parameters"):
        col1, col2 = st.columns(2)
        with col1:
            dt_max_depth = st.slider("Max Depth (DT)", 3, 30, 10, 2)
        with col2:
            dt_min_samples_split = st.slider("Min Samples Split (DT)", 2, 20, 5, 2)

    st.markdown('</div>', unsafe_allow_html=True)

    # Train Models Button
    if st.button("🚀 Start Training", type="primary", use_container_width=True):
        # Prepare data
        st.markdown('<div class="content-box">', unsafe_allow_html=True)
        st.header("🔄 Training Progress")

        progress_bar = st.progress(0)
        status_text = st.empty()

        # Encode categorical variables
        status_text.text("Encoding categorical variables...")
        progress_bar.progress(10)

        df_encoded = df.copy()
        encoders = {}

        # Encode categorical features (not incidentType - that's the target)
        categorical_cols = ['state', 'declarationType']
        for col in categorical_cols:
            if col in df_encoded.columns:
                le = LabelEncoder()
                df_encoded[col + '_encoded'] = le.fit_transform(df_encoded[col].astype(str))
                encoders[col] = le

        # Encode target variable (incidentType)
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(df_encoded['incidentType'].astype(str))

        # Prepare features
        feature_cols = []
        for col in ['state_encoded', 'declarationType_encoded']:
            if col in df_encoded.columns:
                feature_cols.append(col)

        numeric_cols = ['fyDeclared', 'ihProgramDeclared', 'iaProgramDeclared',
                       'paProgramDeclared', 'hmProgramDeclared', 'tribalRequest', 'region']
        for col in numeric_cols:
            if col in df_encoded.columns:
                feature_cols.append(col)

        X = df_encoded[feature_cols].fillna(0)

        # Split data
        status_text.text("Splitting data into train and test sets...")
        progress_bar.progress(20)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=int(random_state), stratify=y
        )

        # Scale features
        status_text.text("Scaling features...")
        progress_bar.progress(30)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train models
        models = {}
        results = {}
        progress_step = 40

        # Random Forest
        if train_rf:
            status_text.text("Training Random Forest...")
            progress_bar.progress(progress_step)
            rf_model = RandomForestClassifier(
                n_estimators=rf_n_estimators,
                max_depth=rf_max_depth,
                min_samples_split=rf_min_samples_split,
                random_state=int(random_state),
                n_jobs=-1
            )
            rf_model.fit(X_train_scaled, y_train)
            train_acc = rf_model.score(X_train_scaled, y_train)
            test_acc = rf_model.score(X_test_scaled, y_test)
            models['Random Forest'] = rf_model
            results['Random Forest'] = {'train_acc': train_acc, 'test_acc': test_acc}
            progress_step += 10

        # Logistic Regression
        if train_lr:
            status_text.text("Training Logistic Regression...")
            progress_bar.progress(progress_step)
            lr_model = LogisticRegression(
                C=lr_c,
                max_iter=lr_max_iter,
                random_state=int(random_state)
            )
            lr_model.fit(X_train_scaled, y_train)
            train_acc = lr_model.score(X_train_scaled, y_train)
            test_acc = lr_model.score(X_test_scaled, y_test)
            models['Logistic Regression'] = lr_model
            results['Logistic Regression'] = {'train_acc': train_acc, 'test_acc': test_acc}
            progress_step += 10

        # Decision Tree
        if train_dt:
            status_text.text("Training Decision Tree...")
            progress_bar.progress(progress_step)
            dt_model = DecisionTreeClassifier(
                max_depth=dt_max_depth,
                min_samples_split=dt_min_samples_split,
                random_state=int(random_state)
            )
            dt_model.fit(X_train_scaled, y_train)
            train_acc = dt_model.score(X_train_scaled, y_train)
            test_acc = dt_model.score(X_test_scaled, y_test)
            models['Decision Tree'] = dt_model
            results['Decision Tree'] = {'train_acc': train_acc, 'test_acc': test_acc}
            progress_step += 10

        # SVM
        if train_svm:
            status_text.text("Training Support Vector Machine...")
            progress_bar.progress(progress_step)
            svm_model = SVC(random_state=int(random_state), probability=True)
            svm_model.fit(X_train_scaled, y_train)
            train_acc = svm_model.score(X_train_scaled, y_train)
            test_acc = svm_model.score(X_test_scaled, y_test)
            models['SVM'] = svm_model
            results['SVM'] = {'train_acc': train_acc, 'test_acc': test_acc}
            progress_step += 10

        # Naive Bayes
        if train_nb:
            status_text.text("Training Naive Bayes...")
            progress_bar.progress(progress_step)
            nb_model = GaussianNB()
            nb_model.fit(X_train_scaled, y_train)
            train_acc = nb_model.score(X_train_scaled, y_train)
            test_acc = nb_model.score(X_test_scaled, y_test)
            models['Naive Bayes'] = nb_model
            results['Naive Bayes'] = {'train_acc': train_acc, 'test_acc': test_acc}
            progress_step += 10

        # Save models
        status_text.text("Saving models...")
        progress_bar.progress(90)

        # Find best model
        best_model_name = max(results, key=lambda x: results[x]['test_acc'])
        best_model = models[best_model_name]

        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)

        # Save individual components (as expected by prediction page)
        joblib.dump(best_model, 'models/disaster_model.pkl')
        joblib.dump(scaler, 'models/scaler.pkl')
        joblib.dump(encoders, 'models/label_encoders.pkl')
        joblib.dump(target_encoder, 'models/target_encoder.pkl')
        joblib.dump(feature_cols, 'models/feature_names.pkl')
        joblib.dump({
            'model_name': best_model_name,
            'accuracy': results[best_model_name]['test_acc'],
            'num_classes': len(target_encoder.classes_),
            'classes': target_encoder.classes_.tolist()
        }, 'models/model_metadata.pkl')

        status_text.text("Training completed!")
        progress_bar.progress(100)
        time.sleep(0.5)

        st.markdown('</div>', unsafe_allow_html=True)

        # Display Results
        st.markdown('<div class="content-box">', unsafe_allow_html=True)
        st.header("📊 Training Results")

        # Results table
        results_df = pd.DataFrame(results).T
        results_df['train_acc'] = (results_df['train_acc'] * 100).round(2)
        results_df['test_acc'] = (results_df['test_acc'] * 100).round(2)
        results_df.columns = ['Training Accuracy (%)', 'Test Accuracy (%)']

        st.dataframe(results_df.style.highlight_max(axis=0, color='lightgreen'), use_container_width=True)

        # Visualization
        fig = go.Figure()

        fig.add_trace(go.Bar(
            name='Training Accuracy',
            x=list(results.keys()),
            y=[results[m]['train_acc'] * 100 for m in results.keys()],
            marker_color='#667eea'
        ))

        fig.add_trace(go.Bar(
            name='Test Accuracy',
            x=list(results.keys()),
            y=[results[m]['test_acc'] * 100 for m in results.keys()],
            marker_color='#764ba2'
        ))

        fig.update_layout(
            title="Model Performance Comparison",
            xaxis_title="Model",
            yaxis_title="Accuracy (%)",
            barmode='group',
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

        # Best Model
        st.success(f"✅ Best Model: **{best_model_name}** with Test Accuracy: **{results[best_model_name]['test_acc']*100:.2f}%**")

        st.info(f"💾 Model saved to: `models/disaster_model.pkl`")

        st.markdown('</div>', unsafe_allow_html=True)

        # Feature Importance (for tree-based models)
        if best_model_name in ['Random Forest', 'Decision Tree']:
            st.markdown('<div class="content-box">', unsafe_allow_html=True)
            st.header("🎯 Feature Importance")

            feature_importance = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': best_model.feature_importances_
            }).sort_values('Importance', ascending=False)

            fig_importance = px.bar(
                feature_importance.head(10),
                x='Importance',
                y='Feature',
                orientation='h',
                title="Top 10 Most Important Features",
                color='Importance',
                color_continuous_scale='Viridis'
            )
            fig_importance.update_layout(height=500)
            st.plotly_chart(fig_importance, use_container_width=True)

            st.markdown('</div>', unsafe_allow_html=True)

else:
    st.error("❌ Dataset not found! Please ensure 'disaster_declarations.csv' is in the data folder.")

st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: white; padding: 10px;'>
        <p>Model Training | Disaster Declaration Analysis System</p>
    </div>
""", unsafe_allow_html=True)
