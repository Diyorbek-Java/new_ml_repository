import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

st.set_page_config(page_title="Predict Disaster", page_icon="🔮", layout="wide")

st.title("🔮 Disaster Type Prediction")
st.markdown("Enter disaster information to predict the type")

st.markdown("---")

# Load model
@st.cache_resource
def load_model():
    try:
        return {
            'model': joblib.load('models/disaster_model.pkl'),
            'scaler': joblib.load('models/scaler.pkl'),
            'encoders': joblib.load('models/label_encoders.pkl'),
            'target_encoder': joblib.load('models/target_encoder.pkl'),
            'features': joblib.load('models/feature_names.pkl'),
            'metadata': joblib.load('models/model_metadata.pkl')
        }
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

models = load_model()

if models is None:
    st.warning("⚠️ Please train the model first!")
    st.stop()

st.success(f"✅ Model Loaded: {models['metadata']['model_name']} | Accuracy: {models['metadata']['accuracy']*100:.1f}%")

st.markdown("---")

# Input Form
st.header("📝 Input Parameters")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Location")

    states = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'ID',
              'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS',
              'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK',
              'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV',
              'WI', 'WY', 'DC', 'PR', 'VI', 'GU', 'AS', 'MP']

    state = st.selectbox("State", states, index=4)
    st.caption("Which U.S. state or territory?")

    region = st.selectbox("FEMA Region (1-10)", list(range(1, 11)), index=8)
    st.caption("FEMA administrative region")

    fips_state = st.number_input("FIPS State Code", min_value=0, max_value=100, value=6)
    fips_county = st.number_input("FIPS County Code", min_value=0, max_value=999, value=37)
    place_code = st.number_input("Place Code", min_value=0, max_value=99999, value=0)

with col2:
    st.subheader("Declaration Details")

    declaration_type = st.radio("Declaration Type", ["DR", "EM", "FM"])
    st.caption("DR=Disaster, EM=Emergency, FM=Fire")

    fiscal_year = st.number_input("Fiscal Year", min_value=2000, max_value=2026, value=2024)

    tribal_request = st.radio("Tribal Request", ["No", "Yes"])

    st.subheader("Programs Activated")

    ih_program = st.checkbox("IH - Individual & Households")
    ia_program = st.checkbox("IA - Individual Assistance")
    pa_program = st.checkbox("PA - Public Assistance", value=True)
    hm_program = st.checkbox("HM - Hazard Mitigation", value=True)

st.markdown("---")

# Predict Button
if st.button("🎯 PREDICT", type="primary", use_container_width=True):
    st.markdown("### 🔄 Processing...")

    # Show inputs
    with st.expander("📋 Input Data"):
        input_summary = pd.DataFrame({
            "Parameter": ["State", "Region", "Declaration Type", "Fiscal Year", "Tribal",
                         "IH Program", "IA Program", "PA Program", "HM Program"],
            "Value": [state, region, declaration_type, fiscal_year, tribal_request,
                     "Yes" if ih_program else "No", "Yes" if ia_program else "No",
                     "Yes" if pa_program else "No", "Yes" if hm_program else "No"]
        })
        st.dataframe(input_summary, use_container_width=True)

    try:
        # Prepare input
        input_data = {}

        # Encode categorical
        if 'state' in models['encoders']:
            try:
                input_data['state'] = models['encoders']['state'].transform([state])[0]
            except:
                input_data['state'] = 0

        if 'declarationType' in models['encoders']:
            try:
                input_data['declarationType'] = models['encoders']['declarationType'].transform([declaration_type])[0]
            except:
                input_data['declarationType'] = 0

        # Numeric features
        input_data['fyDeclared'] = fiscal_year
        input_data['ihProgramDeclared'] = int(ih_program)
        input_data['iaProgramDeclared'] = int(ia_program)
        input_data['paProgramDeclared'] = int(pa_program)
        input_data['hmProgramDeclared'] = int(hm_program)
        input_data['tribalRequest'] = 1 if tribal_request == "Yes" else 0
        input_data['region'] = region
        input_data['fipsStateCode'] = fips_state
        input_data['fipsCountyCode'] = fips_county
        input_data['placeCode'] = place_code

        # Show encoded data
        with st.expander("🔢 Encoded Features"):
            st.json(input_data)

        # Create DataFrame
        input_df = pd.DataFrame([input_data])

        # Add missing features
        for col in models['features']:
            if col not in input_df.columns:
                input_df[col] = 0

        # Reorder
        input_df = input_df[models['features']]

        st.write("**Feature Vector Shape:**", input_df.shape)

        # Scale
        input_scaled = models['scaler'].transform(input_df)
        st.write("**Scaled Data:**", input_scaled)

        # Predict
        model = models['model']
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]

        st.write("**Raw Prediction (Class Index):**", int(prediction))
        st.write("**Prediction Probabilities:**", prediction_proba.tolist())

        # Decode
        predicted_disaster = models['target_encoder'].inverse_transform([prediction])[0]
        confidence = prediction_proba[prediction] * 100

        # Results
        st.markdown("---")
        st.markdown("## 🎯 PREDICTION RESULTS")

        st.success(f"### 🌪️ Predicted Disaster Type: **{predicted_disaster.upper()}**")
        st.info(f"### 📊 Confidence: **{confidence:.2f}%**")

        if confidence > 70:
            st.success("✅ HIGH CONFIDENCE - Model is very sure")
        elif confidence > 50:
            st.warning("⚠️ MODERATE CONFIDENCE - Consider alternatives")
        else:
            st.error("❌ LOW CONFIDENCE - Uncertain prediction")

        # Top 5
        st.markdown("### 📈 Top 5 Most Likely Disaster Types")

        top_5_idx = np.argsort(prediction_proba)[::-1][:5]
        top_5_disasters = models['target_encoder'].inverse_transform(top_5_idx)
        top_5_probs = prediction_proba[top_5_idx] * 100

        results_df = pd.DataFrame({
            "Rank": [1, 2, 3, 4, 5],
            "Disaster Type": top_5_disasters,
            "Probability (%)": [f"{p:.2f}" for p in top_5_probs]
        })

        st.dataframe(results_df, use_container_width=True)

        # Chart
        fig = go.Figure(data=[
            go.Bar(
                x=top_5_probs,
                y=top_5_disasters,
                orientation='h',
                text=[f'{prob:.1f}%' for prob in top_5_probs],
                textposition='outside'
            )
        ])
        fig.update_layout(
            xaxis_title="Probability (%)",
            yaxis_title="Disaster Type",
            height=300,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

        # Actions
        st.markdown("### 💡 Recommended Actions")

        actions = {
            'Severe Storm': ["Deploy utility restoration", "Set up shelters", "Assess damage"],
            'Hurricane': ["Activate evacuation", "Pre-position supplies", "Coastal response"],
            'Flood': ["Deploy water rescue", "Monitor water levels", "Contamination prep"],
            'Fire': ["Fire suppression", "Monitor air quality", "Plan evacuations"],
            'Biological': ["Public health response", "Testing facilities", "Quarantine measures"]
        }

        recs = actions.get(predicted_disaster, ["Deploy emergency teams", "Coordinate response", "Monitor situation"])

        for rec in recs:
            st.write(f"- {rec}")

    except Exception as e:
        st.error(f"❌ Prediction Error: {str(e)}")
        with st.expander("🐛 Debug Info"):
            import traceback
            st.code(traceback.format_exc())

st.markdown("---")
st.caption("🔮 Disaster Type Prediction | Random Forest Model | 89% Accuracy")
