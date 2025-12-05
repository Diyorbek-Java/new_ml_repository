import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve
)
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Model Evaluation", page_icon="📈", layout="wide")

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
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="content-box">', unsafe_allow_html=True)
st.title("📈 Model Evaluation")
st.markdown("Comprehensive evaluation of trained machine learning models")
st.markdown('</div>', unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    try:
        with open('models/disaster_model.pkl', 'rb') as f:
            model_package = pickle.load(f)
        return model_package
    except FileNotFoundError:
        return None

# Load data
@st.cache_data
def load_and_prepare_data():
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder

        df = pd.read_csv('data/disaster_declarations.csv')

        feature_cols = ['state', 'declarationType', 'fyDeclared', 'incidentType',
                       'ihProgramDeclared', 'iaProgramDeclared', 'paProgramDeclared',
                       'hmProgramDeclared', 'tribalRequest', 'region']

        available_cols = [col for col in feature_cols if col in df.columns]
        df_clean = df[available_cols].copy()

        program_cols = [col for col in ['ihProgramDeclared', 'iaProgramDeclared',
                                         'paProgramDeclared', 'hmProgramDeclared'] if col in df_clean.columns]

        if program_cols:
            df_clean['needs_assistance'] = (df_clean[program_cols].sum(axis=1) > 0).astype(int)
        else:
            df_clean['needs_assistance'] = (df_clean['declarationType'] == 'DR').astype(int)

        df_clean = df_clean.dropna()

        # Encode
        df_encoded = df_clean.copy()
        categorical_cols = ['state', 'declarationType', 'incidentType']
        for col in categorical_cols:
            if col in df_encoded.columns:
                le = LabelEncoder()
                df_encoded[col + '_encoded'] = le.fit_transform(df_encoded[col].astype(str))

        feature_cols_encoded = []
        for col in ['state_encoded', 'declarationType_encoded', 'incidentType_encoded']:
            if col in df_encoded.columns:
                feature_cols_encoded.append(col)

        numeric_cols = ['fyDeclared', 'ihProgramDeclared', 'iaProgramDeclared',
                       'paProgramDeclared', 'hmProgramDeclared', 'tribalRequest', 'region']
        for col in numeric_cols:
            if col in df_encoded.columns:
                feature_cols_encoded.append(col)

        X = df_encoded[feature_cols_encoded].fillna(0)
        y = df_encoded['needs_assistance']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        return X_train, X_test, y_train, y_test

    except FileNotFoundError:
        return None, None, None, None

model_package = load_model()
X_train, X_test, y_train, y_test = load_and_prepare_data()

if model_package is not None and X_test is not None:
    model = model_package['model']
    scaler = model_package['scaler']

    # Scale test data
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = scaler.transform(X_train)

    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)

    # Model Overview
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    st.header("🤖 Model Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <h3 style="margin: 0;">{type(model).__name__}</h3>
                <p style="margin: 5px 0 0 0;">Model Type</p>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <h3 style="margin: 0;">{len(X_test):,}</h3>
                <p style="margin: 5px 0 0 0;">Test Samples</p>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
            <div class="metric-card">
                <h3 style="margin: 0;">{X_test.shape[1]}</h3>
                <p style="margin: 5px 0 0 0;">Features</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Performance Metrics
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    st.header("📊 Performance Metrics")

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Accuracy", f"{accuracy:.4f}", f"{accuracy*100:.2f}%")
    with col2:
        st.metric("Precision", f"{precision:.4f}", f"{precision*100:.2f}%")
    with col3:
        st.metric("Recall", f"{recall:.4f}", f"{recall*100:.2f}%")
    with col4:
        st.metric("F1-Score", f"{f1:.4f}", f"{f1*100:.2f}%")

    # Metrics explanation
    with st.expander("📖 Metrics Explanation"):
        st.markdown("""
        - **Accuracy**: Overall correctness of the model - (TP + TN) / Total
        - **Precision**: Of all positive predictions, how many were correct - TP / (TP + FP)
        - **Recall (Sensitivity)**: Of all actual positives, how many were found - TP / (TP + FN)
        - **F1-Score**: Harmonic mean of Precision and Recall - 2 × (Precision × Recall) / (Precision + Recall)

        Where: TP = True Positives, TN = True Negatives, FP = False Positives, FN = False Negatives
        """)

    st.markdown('</div>', unsafe_allow_html=True)

    # Confusion Matrix
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    st.header("🎯 Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)

    col1, col2 = st.columns([2, 1])

    with col1:
        # Create confusion matrix heatmap
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicted Label", y="True Label", color="Count"),
            x=['No Assistance', 'Needs Assistance'],
            y=['No Assistance', 'Needs Assistance'],
            title="Confusion Matrix",
            color_continuous_scale='Blues',
            text_auto=True
        )
        fig_cm.update_layout(height=500)
        st.plotly_chart(fig_cm, use_container_width=True)

    with col2:
        st.markdown("### Matrix Breakdown")

        total = cm.sum()
        tn, fp, fn, tp = cm.ravel()

        st.markdown(f"""
        **True Negatives (TN):** {tn:,}
        - Correctly predicted NO assistance
        - Percentage: {(tn/total)*100:.2f}%

        **False Positives (FP):** {fp:,}
        - Incorrectly predicted assistance
        - Percentage: {(fp/total)*100:.2f}%

        **False Negatives (FN):** {fn:,}
        - Missed assistance cases
        - Percentage: {(fn/total)*100:.2f}%

        **True Positives (TP):** {tp:,}
        - Correctly predicted assistance
        - Percentage: {(tp/total)*100:.2f}%
        """)

    st.markdown('</div>', unsafe_allow_html=True)

    # Classification Report
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    st.header("📋 Classification Report")

    report = classification_report(y_test, y_pred, target_names=['No Assistance', 'Needs Assistance'], output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    st.dataframe(report_df.style.background_gradient(cmap='YlOrRd', subset=['f1-score']), use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ROC Curve
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    st.header("📉 ROC Curve")

    col1, col2 = st.columns(2)

    with col1:
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)

        fig_roc = go.Figure()

        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr,
            name=f'ROC Curve (AUC = {roc_auc:.4f})',
            mode='lines',
            line=dict(color='#667eea', width=3)
        ))

        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            name='Random Classifier',
            mode='lines',
            line=dict(color='red', dash='dash', width=2)
        ))

        fig_roc.update_layout(
            title=f'ROC Curve (AUC = {roc_auc:.4f})',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=500,
            showlegend=True
        )

        st.plotly_chart(fig_roc, use_container_width=True)

    with col2:
        # Precision-Recall Curve
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba[:, 1])

        fig_pr = go.Figure()

        fig_pr.add_trace(go.Scatter(
            x=recall_curve, y=precision_curve,
            name='Precision-Recall Curve',
            mode='lines',
            line=dict(color='#764ba2', width=3),
            fill='tozeroy'
        ))

        fig_pr.update_layout(
            title='Precision-Recall Curve',
            xaxis_title='Recall',
            yaxis_title='Precision',
            height=500
        )

        st.plotly_chart(fig_pr, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Feature Importance
    if hasattr(model, 'feature_importances_'):
        st.markdown('<div class="content-box">', unsafe_allow_html=True)
        st.header("🎯 Feature Importance")

        feature_cols = model_package['feature_cols']
        feature_importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)

        col1, col2 = st.columns([2, 1])

        with col1:
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

        with col2:
            st.subheader("Importance Scores")
            st.dataframe(
                feature_importance.head(10).style.background_gradient(cmap='Greens', subset=['Importance']),
                use_container_width=True,
                height=500
            )

        st.markdown('</div>', unsafe_allow_html=True)

    # Cross-Validation
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    st.header("🔄 Cross-Validation Results")

    with st.spinner("Performing 5-fold cross-validation..."):
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Mean CV Score", f"{cv_scores.mean():.4f}", f"{cv_scores.mean()*100:.2f}%")
    with col2:
        st.metric("Std Deviation", f"{cv_scores.std():.4f}")
    with col3:
        st.metric("95% Confidence", f"±{1.96*cv_scores.std():.4f}")

    # CV Scores visualization
    fig_cv = go.Figure()

    fig_cv.add_trace(go.Bar(
        x=[f'Fold {i+1}' for i in range(len(cv_scores))],
        y=cv_scores,
        marker_color='#667eea',
        text=[f'{score:.4f}' for score in cv_scores],
        textposition='auto'
    ))

    fig_cv.add_hline(
        y=cv_scores.mean(),
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {cv_scores.mean():.4f}"
    )

    fig_cv.update_layout(
        title="Cross-Validation Scores by Fold",
        xaxis_title="Fold",
        yaxis_title="Accuracy Score",
        height=400
    )

    st.plotly_chart(fig_cv, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Model Comparison (if you have other models)
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    st.header("⚖️ Model Performance Summary")

    summary_data = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'CV Mean'],
        'Score': [
            f"{accuracy:.4f}",
            f"{precision:.4f}",
            f"{recall:.4f}",
            f"{f1:.4f}",
            f"{roc_auc:.4f}",
            f"{cv_scores.mean():.4f}"
        ],
        'Percentage': [
            f"{accuracy*100:.2f}%",
            f"{precision*100:.2f}%",
            f"{recall*100:.2f}%",
            f"{f1*100:.2f}%",
            f"{roc_auc*100:.2f}%",
            f"{cv_scores.mean()*100:.2f}%"
        ]
    }

    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # Performance gauge
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=accuracy * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Overall Accuracy", 'font': {'size': 24}},
        delta={'reference': 80, 'increasing': {'color': "green"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#667eea"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#ffcccb'},
                {'range': [50, 75], 'color': '#ffeb9c'},
                {'range': [75, 90], 'color': '#c6efce'},
                {'range': [90, 100], 'color': '#92d050'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 95
            }
        }
    ))

    fig_gauge.update_layout(height=400)
    st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

elif model_package is None:
    st.warning("⚠️ No trained model found! Please train a model first in the Model Training page.")
else:
    st.error("❌ Dataset not found! Please ensure 'disaster_declarations.csv' is in the data folder.")

st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: white; padding: 10px;'>
        <p>Model Evaluation | Disaster Declaration Analysis System</p>
    </div>
""", unsafe_allow_html=True)
