import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Data Exploration", page_icon="📊", layout="wide")

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
st.title("📊 Data Exploration & Analysis")
st.markdown("Explore the FEMA Disaster Declarations dataset with interactive visualizations")
st.markdown('</div>', unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/disaster_declarations.csv')
        # Convert date columns
        date_cols = ['declarationDate', 'incidentBeginDate', 'incidentEndDate']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        return df
    except FileNotFoundError:
        st.error("❌ Dataset not found! Please ensure 'disaster_declarations.csv' is in the data folder.")
        return None

df = load_data()

if df is not None:
    # Dataset Overview
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    st.header("📋 Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Features", len(df.columns))
    with col3:
        st.metric("States", df['state'].nunique())
    with col4:
        st.metric("Incident Types", df['incidentType'].nunique())

    # Data preview
    st.subheader("📄 Data Sample")
    st.dataframe(df.head(20), width='stretch')

    # Data types and missing values
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔍 Data Types")
        dtype_df = pd.DataFrame({
            'Column': df.dtypes.index,
            'Data Type': df.dtypes.values
        })
        st.dataframe(dtype_df, width='stretch', height=400)

    with col2:
        st.subheader("⚠️ Missing Values")
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        missing_df = pd.DataFrame({
            'Column': missing.index,
            'Missing Count': missing.values,
            'Percentage': missing_pct.values
        })
        missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)

        if len(missing_df) > 0:
            st.dataframe(missing_df, width='stretch', height=400)
        else:
            st.success("✅ No missing values found!")

    st.markdown('</div>', unsafe_allow_html=True)

    # Statistical Summary
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    st.header("📈 Statistical Summary")

    st.subheader("Numerical Features")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        st.dataframe(df[numeric_cols].describe(), width='stretch')
    else:
        st.info("No numerical columns found")

    st.markdown('</div>', unsafe_allow_html=True)

    # Visualizations
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    st.header("📊 Data Visualizations")

    # Incident Type Distribution
    st.subheader("🌪️ Disaster Types Distribution")
    incident_counts = df['incidentType'].value_counts().head(15)

    fig1 = px.bar(
        x=incident_counts.values,
        y=incident_counts.index,
        orientation='h',
        labels={'x': 'Number of Declarations', 'y': 'Incident Type'},
        title="Top 15 Disaster Types",
        color=incident_counts.values,
        color_continuous_scale='Viridis'
    )
    fig1.update_layout(showlegend=False, height=500)
    st.plotly_chart(fig1, width='stretch')

    # State-wise analysis
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🗺️ Declarations by State")
        state_counts = df['state'].value_counts().head(15)

        fig2 = px.bar(
            x=state_counts.values,
            y=state_counts.index,
            orientation='h',
            labels={'x': 'Number of Declarations', 'y': 'State'},
            title="Top 15 States by Declarations",
            color=state_counts.values,
            color_continuous_scale='Reds'
        )
        fig2.update_layout(showlegend=False, height=500)
        st.plotly_chart(fig2, width='stretch')

    with col2:
        st.subheader("🎯 Declaration Types")
        decl_type_counts = df['declarationType'].value_counts()

        fig3 = px.pie(
            values=decl_type_counts.values,
            names=decl_type_counts.index,
            title="Distribution of Declaration Types",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig3.update_layout(height=500)
        st.plotly_chart(fig3, width='stretch')

    # Temporal Analysis
    if 'fyDeclared' in df.columns:
        st.subheader("📅 Temporal Trends")

        yearly_counts = df.groupby('fyDeclared').size().reset_index(name='count')

        fig4 = px.line(
            yearly_counts,
            x='fyDeclared',
            y='count',
            title="Disaster Declarations Over Time",
            labels={'fyDeclared': 'Fiscal Year', 'count': 'Number of Declarations'},
            markers=True
        )
        fig4.update_traces(line_color='#667eea', line_width=3)
        fig4.update_layout(height=400)
        st.plotly_chart(fig4, width='stretch')

    # Program Analysis
    st.subheader("🏛️ Assistance Programs")

    program_cols = ['ihProgramDeclared', 'iaProgramDeclared', 'paProgramDeclared', 'hmProgramDeclared']
    available_programs = [col for col in program_cols if col in df.columns]

    if available_programs:
        program_names = {
            'ihProgramDeclared': 'Individual & Households',
            'iaProgramDeclared': 'Individual Assistance',
            'paProgramDeclared': 'Public Assistance',
            'hmProgramDeclared': 'Hazard Mitigation'
        }

        program_activation = []
        program_labels = []

        for prog in available_programs:
            activation_rate = (df[prog].sum() / len(df) * 100)
            program_activation.append(activation_rate)
            program_labels.append(program_names[prog])

        fig5 = go.Figure(data=[
            go.Bar(
                x=program_labels,
                y=program_activation,
                marker=dict(
                    color=program_activation,
                    colorscale='Blues',
                    showscale=True,
                    colorbar=dict(title="Activation %")
                ),
                text=[f"{val:.1f}%" for val in program_activation],
                textposition='auto'
            )
        ])

        fig5.update_layout(
            title="Assistance Program Activation Rates",
            xaxis_title="Program Type",
            yaxis_title="Activation Rate (%)",
            height=400
        )
        st.plotly_chart(fig5, width='stretch')

    # Region Analysis
    if 'region' in df.columns:
        st.subheader("🌎 FEMA Regional Analysis")

        region_counts = df['region'].value_counts().sort_index()

        fig6 = px.bar(
            x=region_counts.index,
            y=region_counts.values,
            labels={'x': 'FEMA Region', 'y': 'Number of Declarations'},
            title="Declarations by FEMA Region",
            color=region_counts.values,
            color_continuous_scale='Oranges'
        )
        fig6.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig6, width='stretch')

    # Tribal Requests
    if 'tribalRequest' in df.columns:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🏛️ Tribal vs Non-Tribal Requests")
            tribal_counts = df['tribalRequest'].value_counts()
            tribal_labels = ['Non-Tribal', 'Tribal']

            fig7 = px.pie(
                values=tribal_counts.values,
                names=tribal_labels,
                title="Tribal Request Distribution",
                hole=0.3,
                color_discrete_sequence=['#667eea', '#764ba2']
            )
            st.plotly_chart(fig7, width='stretch')

        with col2:
            # Incident type vs declaration type heatmap
            st.subheader("🔥 Incident vs Declaration Type")
            cross_tab = pd.crosstab(
                df['incidentType'].head(10),
                df['declarationType']
            )

            fig8 = px.imshow(
                cross_tab,
                labels=dict(x="Declaration Type", y="Incident Type", color="Count"),
                title="Incident Type vs Declaration Type Heatmap",
                color_continuous_scale='Viridis',
                aspect='auto'
            )
            st.plotly_chart(fig8, width='stretch')

    st.markdown('</div>', unsafe_allow_html=True)

    # Correlation Analysis
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    st.header("🔗 Correlation Analysis")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()

        fig9 = px.imshow(
            corr_matrix,
            labels=dict(color="Correlation"),
            title="Feature Correlation Matrix",
            color_continuous_scale='RdBu_r',
            aspect='auto',
            text_auto='.2f'
        )
        fig9.update_layout(height=600)
        st.plotly_chart(fig9, width='stretch')
    else:
        st.info("Insufficient numerical columns for correlation analysis")

    st.markdown('</div>', unsafe_allow_html=True)

    # Download Section
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    st.header("💾 Download Data")

    col1, col2 = st.columns(2)

    with col1:
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Full Dataset (CSV)",
            data=csv,
            file_name="disaster_declarations_full.csv",
            mime="text/csv",
        )

    with col2:
        summary_stats = df.describe(include='all')
        summary_csv = summary_stats.to_csv().encode('utf-8')
        st.download_button(
            label="📊 Download Summary Statistics (CSV)",
            data=summary_csv,
            file_name="summary_statistics.csv",
            mime="text/csv",
        )

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: white; padding: 10px;'>
        <p>Data Exploration | Disaster Declaration Analysis System</p>
    </div>
""", unsafe_allow_html=True)
