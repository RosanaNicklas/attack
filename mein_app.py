import streamlit as st
import pickle
import pandas as pd
import numpy as np
from PIL import Image
import base64
import json
from streamlit_lottie import st_lottie
# -------------------------------- ELEGANT CONFIGURATION -------------------------------
st.set_page_config(
    page_title="CardioSafe | Heart Attack Risk Assessment",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for elegant styling
st.markdown("""
    <style>
        :root {
            --primary: #2e4765;
            --secondary: #3a5a78;
            --accent: #e63946;
            --light: #f8f9fa;
            --dark: #212529;
        }
        
        .main {
            background-color: #fafafa;
            padding-top: 0;
        }
        
        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #fafafa 100%);
        }
        
        h1, h2, h3 {
            font-family: 'Playfair Display', serif;
            color: var(--primary);
        }
        
        .elegant-card {
            background: white;
            border-radius: 12px;
            box-shadow: 0 6px 20px rgba(0,0,0,0.08);
            padding: 2rem;
            margin-bottom: 2rem;
            border-left: 4px solid var(--accent);
        }
        
        .feature-box {
            border-radius: 10px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            background: white;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .feature-box:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 24px rgba(0,0,0,0.1);
        }
        
        .stNumberInput, .stSelectbox {
            border-radius: 8px !important;
            border: 1px solid #e0e0e0 !important;
        }
        
        .stButton>button {
            background: linear-gradient(135deg, #2e4765 0%, #3a5a78 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.75rem 1.5rem;
            font-weight: 500;
            transition: all 0.3s ease;
            width: 100%;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(46, 71, 101, 0.2);
        }
        
        .risk-high {
            background: linear-gradient(135deg, #e63946 0%, #ff758c 100%);
            color: white;
            padding: 1rem;
            border-radius: 8px;
            animation: pulse 2s infinite;
        }
        
        .risk-low {
            background: linear-gradient(135deg, #2ecc71 0%, #1abc9c 100%);
            color: white;
            padding: 1rem;
            border-radius: 8px;
        }
        
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(230, 57, 70, 0.4); }
            70% { box-shadow: 0 0 0 10px rgba(230, 57, 70, 0); }
            100% { box-shadow: 0 0 0 0 rgba(230, 57, 70, 0); }
        }
        
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background: var(--primary);
            color: white;
            padding: 1rem;
            text-align: center;
            font-family: 'Montserrat', sans-serif;
            z-index: 1000;
        }
    </style>
    
    <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Montserrat:wght@300;400;500&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css">
""", unsafe_allow_html=True)

# -------------------------------- MODEL LOADING -------------------------------
@st.cache_resource
def load_model():
    try:
        with open('attack_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open("model_features.pkl", "rb") as f:  # Save this during training
            expected_features = pickle.load(f)
        return model, expected_features
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

model, expected_features = load_model()

# -------------------------------- HERO SECTION -------------------------------

@st.cache_data
def load_lottie_file(filepath: str):
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except Exception as e:
        st.warning(f"Animation not loaded: {str(e)}")
        return None

# Load the heart animation
lottie_heart = load_lottie_file("heart_animation.json")

col1, col2 = st.columns([1, 1])
with col1:
    st.markdown("""
    <div style="padding: 2rem 0;">
        <h1 style="font-size: 3rem; margin-bottom: 1rem;">CardioSafe</h1>
        <h2 style="font-family: 'Montserrat'; font-weight: 300; color: var(--secondary); margin-top: 0;">
            Advanced Cardiac Risk Assessment
        </h2>
        <p style="font-family: 'Montserrat'; color: var(--secondary);">
            Our AI-powered platform evaluates your cardiovascular health indicators 
            to assess heart attack risk with clinical accuracy.
        </p>
    </div>
    """, unsafe_allow_html=True)
   
with col2:
    if lottie_heart:
        st_lottie(lottie_heart, height=200, key="heart")


# -------------------------------- INPUT SECTION -------------------------------
with st.container():
    st.markdown("""
    <div class="elegant-card">
        <h2 style="margin-top: 0;"><i class="fas fa-user-md" style="margin-right: 10px;"></i>Patient Assessment</h2>
        <p style="color: var(--secondary);">Please provide accurate health information for precise risk evaluation</p>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        Age = st.number_input('Age', min_value=18, max_value=120, value=45, 
                            help="Patient's age in years")
        Sex = st.selectbox('Gender', ['Male', 'Female'], 
                          help="Biological sex (important for risk assessment)")
        ChestPainType = st.selectbox('Chest Pain Type', 
                         ['Typical Angina', 'Atypical Angina', 
                          'Non-anginal Pain', 'Asymptomatic'],
                         help="Type of chest pain experienced")
        RestingBP = st.number_input('Resting Blood Pressure (mm Hg)', 
                                min_value=80, max_value=250, value=120,
                                help="Measured at rest")
        Cholesterol = st.number_input('Serum Cholesterol (mg/dl)', 
                              min_value=100, max_value=600, value=200,
                              help="Total cholesterol level")
        
    with col2:
        fasting_blood_sugar = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['No', 'Yes'],
                          help="Elevated fasting blood sugar")
        rest_ecg_result = st.selectbox('Resting ECG Results', 
                              ['Normal', 'ST-T Abnormality', 
                               'Probable LV Hypertrophy'],
                              help="Electrocardiogram results at rest")
        max_heart_rate = st.number_input('Maximum Heart Rate Achieved', 
                                 min_value=60, max_value=220, value=150,
                                 help="Peak heart rate during exercise")
        exercise_induced_angina = st.selectbox('Exercise Induced Angina', ['No', 'Yes'],
                           help="Chest pain during exercise")
        st_depression = st.number_input('ST Depression Induced by Exercise', 
                                 min_value=0.0, max_value=6.0, value=0.0, step=0.1,
                                 help="Depression measurement during ECG")
    
    st_slop = st.selectbox('Slope of Peak Exercise ST Segment', 
                      ['Upsloping', 'Flat', 'Downsloping'],
                      help="ECG pattern during peak exercise")
    
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------- PREDICTION LOGIC -------------------------------
if st.button('Assess Cardiac Risk', key='predict'):
    with st.spinner('Analyzing health indicators...'):
        # Process inputs
        input_data = {
            'Age': Age,
            'Sex': 1 if Sex == 'Male' else 0,
            'ChestPainType': ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'].index(ChestPainType),
            'RestingBP': RestingBP,
            'Cholesterol': Cholesterol,
            'fasting_blood_sugar': 1 if fasting_blood_sugar == 'Yes' else 0,
            'rest_ecg_result': ['Normal', 'ST-T Abnormality', 'Probable LV Hypertrophy'].index(rest_ecg_result),
            'max_heart_rate': max_heart_rate,
            'exercise_induced_angina': 1 if exercise_induced_angina == 'Yes' else 0,
            'st_depression': st_depression,
            'st_slop': ['Upsloping', 'Flat', 'Downsloping'].index(st_slop)
        }
        
        # Convert to DataFrame for prediction
        df = pd.DataFrame([input_data])
        
        try:
            prediction = model.predict(df)
            probability = model.predict_proba(df)[0][1]
            
            st.markdown("""
            <div class="elegant-card" style="margin-top: 2rem;">
                <h2 style="margin-top: 0;"><i class="fas fa-heartbeat" style="margin-right: 10px;"></i>Risk Assessment Results</h2>
            """, unsafe_allow_html=True)
            
            if prediction[0] == 1:
                st.markdown(f"""
                <div class="risk-high">
                    <h3 style="color: white; margin: 0;"><i class="fas fa-exclamation-triangle"></i> Elevated Risk Detected</h3>
                    <p style="color: white; margin-bottom: 0;">Probability: {probability*100:.1f}%</p>
                </div>
                <div style="margin-top: 1.5rem;">
                    <h4><i class="fas fa-notes-medical" style="margin-right: 10px;"></i>Clinical Recommendations</h4>
                    <ul>
                        <li>Consult a cardiologist immediately</li>
                        <li>Consider lifestyle modifications (diet, exercise)</li>
                        <li>Monitor blood pressure regularly</li>
                        <li>Schedule follow-up cardiac testing</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="risk-low">
                    <h3 style="color: white; margin: 0;"><i class="fas fa-check-circle"></i> Low Risk Profile</h3>
                    <p style="color: white; margin-bottom: 0;">Probability: {probability*100:.1f}%</p>
                </div>
                <div style="margin-top: 1.5rem;">
                    <h4><i class="fas fa-heart" style="margin-right: 10px;"></i>Preventive Guidance</h4>
                    <ul>
                        <li>Maintain current healthy habits</li>
                        <li>Annual cardiac screening recommended</li>
                        <li>Continue regular physical activity</li>
                        <li>Monitor cholesterol levels</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"An error occurred during analysis: {str(e)}")

# -------------------------------- ELEGANT FOOTER -------------------------------
st.markdown("""
    <div class="footer">
        <div style="display: flex; justify-content: center; align-items: center; gap: 2rem;">
            <span style="font-family: 'Montserrat'; font-weight: 300;">
                <i class="fas fa-shield-alt"></i> Rosana Longares
            </span>
            <span style="font-family: 'Montserrat'; font-weight: 300;">
                <i class="fas fa-certificate"></i> Validated Algorithm
            </span>
        </div>
        <div style="margin-top: 0.5rem; font-family: 'Montserrat'; font-size: 0.9rem;">
            © 2025 CardioSafe AI | Not a substitute for professional medical advice
        </div>
    </div>
""", unsafe_allow_html=True)