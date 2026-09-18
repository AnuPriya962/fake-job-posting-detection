import streamlit as st
import os
import json
import pandas as pd
from src.predict import predict_single_posting

st.set_page_config(
    page_title="Fake Job Posting Detector",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Fake Job Posting Detection System")
st.markdown("Automated classification and fraud risk analysis for online job recruitments.")

# Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home / Predict", "Model Performance", "About Project"])

if page == "Home / Predict":
    st.subheader("Analyze Job Posting")
    
    with st.form("job_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            title = st.text_input("Job Title", value="Software Engineer")
            company = st.text_input("Company Profile", value="Fast-growing tech startup...")
            location = st.text_input("Location", value="US, NY, New York")
            department = st.text_input("Department", value="Engineering")
            salary_range = st.text_input("Salary Range", value="100000-120000")
            
        with col2:
            telecommuting = st.selectbox("Telecommuting / Remote", [0, 1], index=1)
            has_company_logo = st.selectbox("Company Logo Present", [1, 0], index=0)
            has_questions = st.selectbox("Has Screening Questions", [1, 0], index=0)
            employment_type = st.selectbox("Employment Type", ["Full-time", "Part-time", "Contract", "Other"])
            required_experience = st.selectbox("Required Experience", ["Entry level", "Mid-Senior level", "Associate", "Executive"])
            required_education = st.selectbox("Required Education", ["Bachelor's Degree", "Master's Degree", "High School or equivalent", "Unspecified"])

        description = st.text_area("Job Description", height=150, value="We are looking for an experienced developer...")
        requirements = st.text_area("Requirements", height=100, value="3+ years of experience with Python and SQL...")
        benefits = st.text_area("Benefits", height=100, value="Health insurance, 401k match, flexible PTO...")
        
        submit_btn = st.form_submit_button("Analyze Posting Risk")

    if submit_btn:
        input_payload = {
            "title": title,
            "company_profile": company,
            "location": location,
            "department": department,
            "salary_range": salary_range,
            "telecommuting": telecommuting,
            "has_company_logo": has_company_logo,
            "has_questions": has_questions,
            "employment_type": employment_type,
            "required_experience": required_experience,
            "required_education": required_education,
            "description": description,
            "requirements": requirements,
            "benefits": benefits
        }
        
        try:
            result = predict_single_posting(input_payload)
            st.divider()
            
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                if result["is_fake"]:
                    st.error(f"⚠️ **Result:** Potential Fake Job Posting Detected")
                else:
                    st.success(f"✅ **Result:** Authentic Job Posting Likely")
                    
            with res_col2:
                st.metric(
                    label="Fraud Probability Score", 
                    value=f"{result['fraud_probability'] * 100:.1f}%",
                    delta=result["risk_level"],
                    delta_color="inverse" if result["is_fake"] else "normal"
                )
                
        except Exception as e:
            st.warning(f"Error executing prediction: {e}")

elif page == "Model Performance":
    st.subheader("Model Evaluation Metrics")
    eval_file = os.path.join("models", "evaluation_results.json")
    
    if os.path.exists(eval_file):
        with open(eval_file, "r") as f:
            metrics = json.load(f)
        st.json(metrics)
    else:
        st.info("Evaluation results will appear here once the training pipeline executes.")

elif page == "About Project":
    st.subheader("About Fake Job Detection Pipeline")
    st.markdown("""
    This application utilizes natural language processing (TF-IDF) combined with tabular feature engineering 
    to evaluate incoming job listings and classify fraudulent recruitments in real time.
    """)
