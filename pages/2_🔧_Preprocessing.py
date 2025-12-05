import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Data Preprocessing", page_icon="🔧", layout="wide")

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
    .step-box {
        background-color: #f8fafc;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 15px 0;
    }
    .code-box {
        background-color: #1e293b;
        color: #e2e8f0;
        padding: 15px;
        border-radius: 8px;
        font-family: 'Courier New', monospace;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="content-box">', unsafe_allow_html=True)
st.title("🔧 Data Preprocessing")
st.markdown("Learn about the data cleaning, transformation, and feature engineering steps")
st.markdown('</div>', unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/disaster_declarations.csv')
        return df
    except FileNotFoundError:
        return None

df = load_data()

if df is not None:
    # Preprocessing Overview
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    st.header("📋 Preprocessing Pipeline Overview")

    st.markdown("""
    The data preprocessing pipeline consists of several critical steps to prepare the raw data
    for machine learning models. Each step is designed to handle specific data quality issues
    and enhance the predictive power of our features.
    """)

    # Pipeline visualization
    pipeline_steps = [
        "1️⃣ Data Loading",
        "2️⃣ Missing Value Handling",
        "3️⃣ Feature Selection",
        "4️⃣ Encoding Categorical Variables",
        "5️⃣ Feature Engineering",
        "6️⃣ Feature Scaling",
        "7️⃣ Train-Test Split"
    ]

    col1, col2, col3, col4 = st.columns(4)
    cols = [col1, col2, col3, col4]

    for idx, step in enumerate(pipeline_steps):
        with cols[idx % 4]:
            st.markdown(f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            color: white; padding: 15px; border-radius: 10px;
                            text-align: center; margin: 10px 0;">
                    <strong>{step}</strong>
                </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Step 1: Data Loading
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    st.header("1️⃣ Data Loading")

    st.markdown("""
    <div class="step-box">
    <h4>Objective</h4>
    <p>Load the raw FEMA disaster declarations dataset and perform initial inspection.</p>
    </div>
    """, unsafe_allow_html=True)

    st.code("""
import pandas as pd

# Load the dataset
df = pd.read_csv('data/disaster_declarations.csv')

# Initial inspection
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
    """, language="python")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Original Rows", f"{len(df):,}")
    with col2:
        st.metric("Original Columns", len(df.columns))
    with col3:
        st.metric("Total Data Points", f"{df.size:,}")

    st.markdown('</div>', unsafe_allow_html=True)

    # Step 2: Missing Values
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    st.header("2️⃣ Missing Value Handling")

    st.markdown("""
    <div class="step-box">
    <h4>Objective</h4>
    <p>Identify and handle missing values using appropriate strategies for different feature types.</p>
    </div>
    """, unsafe_allow_html=True)

    # Calculate missing values
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        'Column': missing.index,
        'Missing Count': missing.values,
        'Percentage': missing_pct.values
    })
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Percentage', ascending=False)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Missing Values Summary")
        if len(missing_df) > 0:
            st.dataframe(missing_df.head(10), use_container_width=True)
        else:
            st.success("✅ No missing values found!")

    with col2:
        st.subheader("Missing Value Strategy")
        st.markdown("""
        **Strategies Applied:**

        - **Categorical Features**: Fill with mode or 'Unknown'
        - **Numerical Features**: Fill with median or 0
        - **Date Features**: Forward/backward fill or drop
        - **Critical Features**: Drop rows with missing values
        - **Non-essential Features**: Drop columns if >50% missing
        """)

    st.code("""
# Handle missing values
df['incidentType'].fillna('Unknown', inplace=True)
df['tribalRequest'].fillna(0, inplace=True)

# Drop rows where target variable is missing
df = df.dropna(subset=['paProgramDeclared', 'iaProgramDeclared'])

# Drop columns with >50% missing values
threshold = len(df) * 0.5
df = df.dropna(thresh=threshold, axis=1)
    """, language="python")

    st.markdown('</div>', unsafe_allow_html=True)

    # Step 3: Feature Selection
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    st.header("3️⃣ Feature Selection")

    st.markdown("""
    <div class="step-box">
    <h4>Objective</h4>
    <p>Select the most relevant features for predicting disaster assistance requirements.</p>
    </div>
    """, unsafe_allow_html=True)

    # Define selected features
    selected_features = {
        'Geographic': ['state', 'fipsStateCode', 'region'],
        'Temporal': ['fyDeclared', 'declarationDate', 'incidentBeginDate'],
        'Disaster Info': ['incidentType', 'declarationType', 'tribalRequest'],
        'Programs': ['ihProgramDeclared', 'iaProgramDeclared', 'paProgramDeclared', 'hmProgramDeclared']
    }

    for category, features in selected_features.items():
        st.subheader(f"📌 {category} Features")
        available_features = [f for f in features if f in df.columns]
        st.write(", ".join(available_features))

    st.code("""
# Select relevant features
feature_cols = [
    'state', 'declarationType', 'fyDeclared', 'incidentType',
    'ihProgramDeclared', 'iaProgramDeclared', 'paProgramDeclared',
    'hmProgramDeclared', 'tribalRequest', 'region'
]

df_selected = df[feature_cols].copy()
    """, language="python")

    st.markdown('</div>', unsafe_allow_html=True)

    # Step 4: Encoding
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    st.header("4️⃣ Encoding Categorical Variables")

    st.markdown("""
    <div class="step-box">
    <h4>Objective</h4>
    <p>Convert categorical variables into numerical format suitable for machine learning algorithms.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Label Encoding")
        st.markdown("""
        Used for **ordinal** or **nominal** categorical features with many categories.

        **Applied to:**
        - State (50+ unique values)
        - Incident Type (15+ unique types)
        - Declaration Type (DR, EM, FM)
        """)

        st.code("""
from sklearn.preprocessing import LabelEncoder

le_state = LabelEncoder()
df['state_encoded'] = le_state.fit_transform(df['state'])

le_incident = LabelEncoder()
df['incidentType_encoded'] = le_incident.fit_transform(df['incidentType'])
        """, language="python")

    with col2:
        st.subheader("One-Hot Encoding")
        st.markdown("""
        Used for **nominal** categorical features with few categories.

        **Applied to:**
        - Declaration Type (3 categories)
        - Tribal Request (binary: 0/1)
        """)

        st.code("""
# One-hot encoding for declaration type
df_encoded = pd.get_dummies(
    df,
    columns=['declarationType'],
    prefix='declType'
)
        """, language="python")

    # Show encoding example
    if 'state' in df.columns and 'incidentType' in df.columns:
        st.subheader("📊 Encoding Example")

        example_df = df[['state', 'incidentType', 'declarationType']].head(5).copy()
        example_df['state_encoded'] = pd.Categorical(example_df['state']).codes
        example_df['incidentType_encoded'] = pd.Categorical(example_df['incidentType']).codes

        st.dataframe(example_df, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Step 5: Feature Engineering
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    st.header("5️⃣ Feature Engineering")

    st.markdown("""
    <div class="step-box">
    <h4>Objective</h4>
    <p>Create new features from existing ones to enhance model performance.</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("New Features Created")

    feature_engineering_steps = {
        "📅 Temporal Features": [
            "Year, Month, Quarter from dates",
            "Days between incident start and declaration",
            "Season of incident occurrence"
        ],
        "🎯 Aggregated Features": [
            "Total programs activated per incident",
            "Historical disaster frequency by state",
            "Average response time by region"
        ],
        "🔢 Interaction Features": [
            "Incident type × Declaration type",
            "State × Incident type frequency",
            "Region-specific disaster patterns"
        ],
        "🏷️ Binary Flags": [
            "Is major disaster (DR declaration)",
            "Multiple programs activated (>1)",
            "High-frequency disaster state"
        ]
    }

    for category, features in feature_engineering_steps.items():
        st.markdown(f"**{category}**")
        for feature in features:
            st.markdown(f"  - {feature}")

    st.code("""
# Create new features
df['total_programs'] = (
    df['ihProgramDeclared'] + df['iaProgramDeclared'] +
    df['paProgramDeclared'] + df['hmProgramDeclared']
)

df['is_major_disaster'] = (df['declarationType'] == 'DR').astype(int)

df['incident_year'] = pd.to_datetime(df['declarationDate']).dt.year
df['incident_month'] = pd.to_datetime(df['declarationDate']).dt.month

# State disaster frequency
state_freq = df.groupby('state').size()
df['state_disaster_freq'] = df['state'].map(state_freq)
    """, language="python")

    st.markdown('</div>', unsafe_allow_html=True)

    # Step 6: Feature Scaling
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    st.header("6️⃣ Feature Scaling")

    st.markdown("""
    <div class="step-box">
    <h4>Objective</h4>
    <p>Normalize numerical features to ensure equal contribution to the model.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("StandardScaler")
        st.markdown("""
        Transforms features to have **mean=0** and **std=1**.

        **Formula:** `z = (x - μ) / σ`

        **Best for:** Most ML algorithms, especially neural networks and SVM.
        """)

        st.code("""
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
        """, language="python")

    with col2:
        st.subheader("MinMaxScaler")
        st.markdown("""
        Scales features to a fixed range **[0, 1]**.

        **Formula:** `x_scaled = (x - x_min) / (x_max - x_min)`

        **Best for:** Neural networks, distance-based algorithms.
        """)

        st.code("""
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_train)
        """, language="python")

    # Visualization of scaling effect
    st.subheader("📊 Scaling Effect Visualization")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) > 0:
        sample_col = numeric_cols[0]
        sample_data = df[sample_col].dropna().sample(min(1000, len(df)))

        fig = go.Figure()
        fig.add_trace(go.Histogram(x=sample_data, name='Original', opacity=0.7))

        from sklearn.preprocessing import StandardScaler
        scaled_data = StandardScaler().fit_transform(sample_data.values.reshape(-1, 1))
        fig.add_trace(go.Histogram(x=scaled_data.flatten(), name='Scaled', opacity=0.7))

        fig.update_layout(
            title=f"Distribution: Original vs Scaled ({sample_col})",
            xaxis_title="Value",
            yaxis_title="Frequency",
            barmode='overlay',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Step 7: Train-Test Split
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    st.header("7️⃣ Train-Test Split")

    st.markdown("""
    <div class="step-box">
    <h4>Objective</h4>
    <p>Split the dataset into training and testing sets to evaluate model performance on unseen data.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Split Configuration")
        st.markdown("""
        **Split Ratio:** 80% Train, 20% Test

        **Stratification:** Enabled on target variable

        **Random State:** 42 (for reproducibility)

        **Benefits:**
        - Prevents overfitting
        - Provides unbiased performance estimate
        - Maintains class distribution
        """)

    with col2:
        st.subheader("Split Visualization")

        # Create split visualization
        train_size = int(len(df) * 0.8)
        test_size = len(df) - train_size

        fig = go.Figure(data=[
            go.Bar(
                x=['Training Set', 'Test Set'],
                y=[train_size, test_size],
                text=[f"{train_size:,}<br>({80}%)", f"{test_size:,}<br>({20}%)"],
                textposition='auto',
                marker=dict(color=['#667eea', '#764ba2'])
            )
        ])

        fig.update_layout(
            title="Train-Test Split Distribution",
            yaxis_title="Number of Samples",
            showlegend=False,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    st.code("""
from sklearn.model_selection import train_test_split

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")
    """, language="python")

    st.markdown('</div>', unsafe_allow_html=True)

    # Summary
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    st.header("✅ Preprocessing Summary")

    summary_metrics = {
        "Original Dataset Size": f"{len(df):,} rows",
        "Final Features": "12-15 features",
        "Missing Values Handled": "Yes",
        "Categorical Encoding": "Label + One-Hot",
        "Feature Engineering": "Multiple new features",
        "Feature Scaling": "StandardScaler",
        "Train-Test Split": "80-20 stratified"
    }

    cols = st.columns(3)
    items = list(summary_metrics.items())

    for idx, (key, value) in enumerate(items):
        with cols[idx % 3]:
            st.markdown(f"""
                <div style="background-color: #f8fafc; padding: 15px;
                            border-radius: 10px; text-align: center; margin: 10px 0;
                            border-left: 4px solid #667eea;">
                    <h4 style="margin: 0; color: #1e3a8a;">{value}</h4>
                    <p style="margin: 5px 0 0 0; color: #64748b;">{key}</p>
                </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.error("❌ Dataset not found! Please ensure 'disaster_declarations.csv' is in the data folder.")

st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: white; padding: 10px;'>
        <p>Data Preprocessing | Disaster Declaration Analysis System</p>
    </div>
""", unsafe_allow_html=True)
