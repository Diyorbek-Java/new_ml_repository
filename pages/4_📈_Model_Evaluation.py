import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve
)
from sklearn.preprocessing import label_binarize
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
        return {
            'model': joblib.load('models/disaster_model.pkl'),
            'scaler': joblib.load('models/scaler.pkl'),
            'encoders': joblib.load('models/label_encoders.pkl'),
            'target_encoder': joblib.load('models/target_encoder.pkl'),
            'feature_cols': joblib.load('models/feature_names.pkl'),
            'metadata': joblib.load('models/model_metadata.pkl')
        }
    except FileNotFoundError:
        return None

# Load data
@st.cache_data
def load_and_prepare_data():
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder

        df = pd.read_csv('data/disaster_declarations.csv')

        # Feature columns (incidentType is now the target, not a feature)
        feature_cols = ['state', 'declarationType', 'fyDeclared',
                       'ihProgramDeclared', 'iaProgramDeclared', 'paProgramDeclared',
                       'hmProgramDeclared', 'tribalRequest', 'region']

        # Also need incidentType as target
        all_cols = feature_cols + ['incidentType']
        available_cols = [col for col in all_cols if col in df.columns]
        df_clean = df[available_cols].copy()

        df_clean = df_clean.dropna()

        # Encode
        df_encoded = df_clean.copy()

        # Encode categorical features (not incidentType - that's the target)
        categorical_cols = ['state', 'declarationType']
        for col in categorical_cols:
            if col in df_encoded.columns:
                le = LabelEncoder()
                df_encoded[col + '_encoded'] = le.fit_transform(df_encoded[col].astype(str))

        # Encode target variable (incidentType)
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(df_encoded['incidentType'].astype(str))

        feature_cols_encoded = []
        for col in ['state_encoded', 'declarationType_encoded']:
            if col in df_encoded.columns:
                feature_cols_encoded.append(col)

        numeric_cols = ['fyDeclared', 'ihProgramDeclared', 'iaProgramDeclared',
                       'paProgramDeclared', 'hmProgramDeclared', 'tribalRequest', 'region']
        for col in numeric_cols:
            if col in df_encoded.columns:
                feature_cols_encoded.append(col)

        X = df_encoded[feature_cols_encoded].fillna(0)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        return X_train, X_test, y_train, y_test, target_encoder

    except FileNotFoundError:
        return None, None, None, None, None

model_package = load_model()
X_train, X_test, y_train, y_test, target_encoder = load_and_prepare_data()

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
    class_names = target_encoder.classes_

    col1, col2 = st.columns([2, 1])

    with col1:
        # Create confusion matrix heatmap
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicted Label", y="True Label", color="Count"),
            x=class_names,
            y=class_names,
            title="Confusion Matrix - Incident Type Predictions",
            color_continuous_scale='Blues',
            text_auto=True
        )
        fig_cm.update_layout(height=600)
        st.plotly_chart(fig_cm, use_container_width=True)

    with col2:
        st.markdown("### Matrix Summary")

        total = cm.sum()
        correct = np.trace(cm)
        incorrect = total - correct

        st.markdown(f"""
        **Total Predictions:** {total:,}

        **Correct Predictions:** {correct:,}
        - Percentage: {(correct/total)*100:.2f}%

        **Incorrect Predictions:** {incorrect:,}
        - Percentage: {(incorrect/total)*100:.2f}%

        **Number of Classes:** {len(class_names)}
        """)

        # Show most confused classes
        st.markdown("### Most Confused")
        cm_copy = cm.copy()
        np.fill_diagonal(cm_copy, 0)
        if cm_copy.sum() > 0:
            max_idx = np.unravel_index(cm_copy.argmax(), cm_copy.shape)
            st.markdown(f"**{class_names[max_idx[0]]}** → **{class_names[max_idx[1]]}**")
            st.markdown(f"Confused {cm_copy[max_idx]:,} times")

    st.markdown('</div>', unsafe_allow_html=True)

    # Classification Report
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    st.header("📋 Classification Report")

    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    st.dataframe(report_df.style.background_gradient(cmap='YlOrRd', subset=['f1-score']), use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ROC Curve (Multi-class)
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    st.header("📉 ROC Curve - Multi-class")

    # Binarize the output for multi-class ROC
    n_classes = len(class_names)
    y_test_bin = label_binarize(y_test, classes=range(n_classes))

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_pred_proba.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    col1, col2 = st.columns(2)

    with col1:
        # Plot ROC curves for top 5 classes by frequency
        fig_roc = go.Figure()

        # Add micro-average
        fig_roc.add_trace(go.Scatter(
            x=fpr["micro"], y=tpr["micro"],
            name=f'Micro-average (AUC = {roc_auc["micro"]:.3f})',
            mode='lines',
            line=dict(color='deeppink', width=3, dash='dash')
        ))

        # Add ROC curves for each class (show top 5 by AUC)
        top_classes = sorted(range(n_classes), key=lambda i: roc_auc[i], reverse=True)[:5]
        colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#43e97b']

        for idx, i in enumerate(top_classes):
            fig_roc.add_trace(go.Scatter(
                x=fpr[i], y=tpr[i],
                name=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})',
                mode='lines',
                line=dict(color=colors[idx], width=2)
            ))

        # Add random classifier line
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            name='Random',
            mode='lines',
            line=dict(color='gray', dash='dash', width=1)
        ))

        fig_roc.update_layout(
            title='ROC Curves (Top 5 Classes by AUC)',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=500,
            showlegend=True,
            legend=dict(x=0.6, y=0.1)
        )

        st.plotly_chart(fig_roc, use_container_width=True)

    with col2:
        # Show AUC scores for all classes
        st.markdown("### AUC Scores by Class")

        auc_data = pd.DataFrame({
            'Incident Type': [class_names[i] for i in range(n_classes)],
            'AUC Score': [roc_auc[i] for i in range(n_classes)]
        }).sort_values('AUC Score', ascending=False)

        st.dataframe(
            auc_data.style.background_gradient(cmap='RdYlGn', subset=['AUC Score']),
            use_container_width=True,
            height=500
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # Feature Importance
    if hasattr(model, 'feature_importances_'):
        st.markdown('<div class="content-box">', unsafe_allow_html=True)
        st.header("🎯 Feature Importance")

        feature_cols = model_package['feature_cols']
        feature_importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False).head(15)

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
        'Metric': ['Accuracy', 'Precision (Weighted)', 'Recall (Weighted)', 'F1-Score (Weighted)', 'AUC-ROC (Micro)', 'CV Mean'],
        'Score': [
            f"{accuracy:.4f}",
            f"{precision:.4f}",
            f"{recall:.4f}",
            f"{f1:.4f}",
            f"{roc_auc['micro']:.4f}",
            f"{cv_scores.mean():.4f}"
        ],
        'Percentage': [
            f"{accuracy*100:.2f}%",
            f"{precision*100:.2f}%",
            f"{recall*100:.2f}%",
            f"{f1*100:.2f}%",
            f"{roc_auc['micro']*100:.2f}%",
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
