import streamlit as st
import pandas as pd
from utils import predict_disease, get_disease_info, symptom_list

st.set_page_config(
    page_title="Disease Predictor",
    page_icon="🩺",
    layout="centered"
)

@st.cache_data
def load_support_data():
    desc_df = pd.read_csv('data/symptom_Description.csv')
    prec_df = pd.read_csv('data/symptom_precaution.csv')
    return desc_df, prec_df

desc_df, prec_df = load_support_data()

st.title("🩺 Disease Prediction System")
st.markdown("##### AI-Based Symptom Analysis")
st.markdown("**Riphah International University | AI Course Project | 6th Semester BSCS**")
st.markdown("---")
st.markdown("Select all the symptoms you are experiencing. The system will predict the most probable disease.")

selected_symptoms = st.multiselect(
    label="🔍 Select Your Symptoms:",
    options=sorted(symptom_list),
    placeholder="Type or scroll to find symptoms...",
    help="Select all symptoms that apply to you."
)

st.markdown(f"**Symptoms selected:** {len(selected_symptoms)}")

if st.button("🔎 Predict Disease", use_container_width=True, type="primary"):

    if len(selected_symptoms) == 0:
        st.warning("⚠️ Please select at least one symptom before predicting.")

    elif len(selected_symptoms) < 2:
        st.warning("⚠️ Please select at least 2 symptoms for a reliable prediction.")

    else:
        disease, confidence = predict_disease(selected_symptoms)
        description, precautions = get_disease_info(disease, desc_df, prec_df)

        st.markdown("---")
        st.success(f"### 🏥 Predicted Disease: **{disease}**")
        st.metric(label="Prediction Confidence", value=f"{confidence}%")

        with st.expander("📋 About This Disease", expanded=True):
            st.write(description)

        if precautions:
            with st.expander("✅ Recommended Precautions", expanded=True):
                for i, p in enumerate(precautions, 1):
                    st.write(f"**{i}.** {p.capitalize()}")
        else:
            st.info("No specific precautions found for this disease.")

        st.markdown("---")
        st.error("⚕️ **Medical Disclaimer:** This tool is for educational purposes only. Please consult a qualified physician for proper diagnosis.")

st.markdown("---")
st.caption("Built with Python • Scikit-learn • Streamlit | Muhammad Rehan Mehdi, Raheel Nazir, Mohammad Sami")