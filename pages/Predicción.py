import streamlit as st
import pickle
import json
import pandas as pd
import numpy as np
from streamlit_lottie import st_lottie

# ----------------------------- CONFIGURACIÓN --------------------------------
st.set_page_config(page_title="Heart Attack Prediction", layout="wide")

# CSS para reducir espacios
st.markdown("""
    <style>
        .block-container {padding-top: 1rem; padding-bottom: 1rem;}
        .stButton>button {width: 100%;}
    </style>
""", unsafe_allow_html=True)

# Inicializar historial en session_state
if 'historial' not in st.session_state:
    st.session_state.historial = []

# ----------------------------- CARGA DEL MODELO -----------------------------
def load_lottiefile(filepath: str):
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except Exception as e:
        st.warning(f"No se pudo cargar la animación: {e}")
        return None

lottie_heart = load_lottiefile("heart_animation.json")  # You'll need this file

# ----------------------------- MODEL LOADING --------------------------------
@st.cache_resource
def load_model(path):
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# ----------------------------- INTERFAZ -------------------------------------


c1, c2 = st.columns([1, 3])
with c1:
    if lottie_heart:
        st_lottie(lottie_heart, width=150, height=150, quality="low")
with c2:
    st.title("Predicción de Riesgo de Ataque al Corazón")
    st.markdown("Complete sus datos médicos para evaluar el riesgo")
with st.expander("ℹ️ Explicación de los parámetros del modelo"):
    st.markdown("""
    ### 🩺 Parámetros del modelo de predicción de infarto

    | **Parámetro** | **Significado** | **Normal o saludable** |
    |---------------|------------------|--------------------------|
    | **Age** | Edad del paciente. | Riesgo ↑ después de 45 años (hombres), 55 (mujeres). |
    | **Sex** | Sexo biológico. `0 = Mujer`, `1 = Hombre` | Mayor riesgo en hombres jóvenes. |
    | **Chest Pain Type** | Tipo de dolor torácico. | `0 = Típico` → más asociado a problemas cardíacos. |
    | **RestingBP** | Presión arterial en reposo. | Ideal: `120/80 mm Hg`. Hipertensión > `130`. |
    | **Cholesterol** | Colesterol total en sangre. | < `200 mg/dl` es lo recomendado. |
    | **FastingBS** | ¿Glucosa en ayunas > 120 mg/dl? | < `100 mg/dl` es lo normal. |
    | **RestingECG** | Electrocardiograma en reposo. | `0 = Normal`, otros valores → posibles anomalías. |
    | **MaxHR** | Frecuencia cardíaca máxima. | Cerca de `220 - edad`. |
    | **ExerciseAngina** | Dolor de pecho con ejercicio. | `Yes` indica posible enfermedad coronaria. |
    | **Oldpeak** | Depresión del ST durante esfuerzo. | `0.0` es normal, > `1.0` indica riesgo. |
    | **ST_Slope** | Forma del ST al final del esfuerzo. | `0 = Ascendente` es lo más saludable. |

    *Estos parámetros se utilizan para predecir el riesgo de enfermedad cardíaca a partir de datos clínicos.*
    """, unsafe_allow_html=True)
# ----------------------------- SELECTORES -----------------------------------
# Create input fields with clear mapping to model features
col1, col2 = st.columns(2)

with col1:
    age = st.number_input('Edad', min_value=0, max_value=120, value=50)
    sex = st.selectbox('Sexo', ['Masculino', 'Femenino'])
    cp = st.selectbox('Tipo de Dolor de Pecho', [
        'Típico', 
        'Atípico', 
        'Dolor no anginoso', 
        'Asintomático'
    ], help="0: Típico, 1: Atípico, 2: Dolor no anginoso, 3: Asintomático")
    trtbps = st.number_input('Presión Arterial en Reposo (mm Hg)', min_value=0, max_value=250, value=120)
    chol = st.number_input('Colesterol (mg/dl)', min_value=0, max_value=600, value=200)
    
with col2:
    fbs = st.selectbox('Azúcar en Sangre en Ayunas > 120 mg/dl', ['No', 'Sí'])
    restecg = st.selectbox('Resultados ECG en Reposo', [
        'Normal', 
        'Anormalidad ST-T', 
        'Hipertrofia ventricular'
    ], help="0: Normal, 1: Anormalidad ST-T, 2: Hipertrofia ventricular")
    thalachh = st.number_input('Frecuencia Cardíaca Máxima Alcanzada', min_value=0, max_value=250, value=150)
    exng = st.selectbox('Angina Inducida por Ejercicio', ['No', 'Sí'])
    oldpeak = st.number_input('Depresión del ST inducida por ejercicio', min_value=0.0, max_value=10.0, value=0.0, step=0.1)
    slp = st.selectbox('Pendiente del Segmento ST en Ejercicio Máximo', [
        'Ascendente', 
        'Plano', 
        'Descendente'
    ], help="0: Ascendente, 1: Plano, 2: Descendente")

# ----------------------------- MAPPINGS -------------------------------------
cp_mapping = {
    'Típico': 0,
    'Atípico': 1,
    'Dolor no anginoso': 2,
    'Asintomático': 3
}

restecg_mapping = {
    'Normal': 0,
    'Anormalidad ST-T': 1,
    'Hipertrofia ventricular': 2
}

slp_mapping = {
    'Ascendente': 0,
    'Plano': 1,
    'Descendente': 2
}

# ----------------------------- VALIDACIÓN -----------------------------------
def validate_inputs():
    if age <= 0:
        st.error("La edad debe ser positiva")
        return False
    if trtbps <= 0:
        st.error("La presión arterial debe ser positiva")
        return False
    if chol <= 0:
        st.error("El colesterol debe ser positivo")
        return False
    return True

# ----------------------------- PREDICCIÓN -----------------------------------
if st.button("Evaluar Riesgo"):
    if not validate_inputs():
        st.stop()
    
    try:
        # Convert inputs to model format
        input_data = {
            'age': age,
            'sex': 1 if sex == 'Masculino' else 0,
            'cp': cp_mapping[cp],
            'trtbps': trtbps,
            'chol': chol,
            'fbs': 1 if fbs == 'Sí' else 0,
            'restecg': restecg_mapping[restecg],
            'thalachh': thalachh,
            'exng': 1 if exng == 'Sí' else 0,
            'oldpeak': oldpeak,
            'slp': slp_mapping[slp]
        }
        
        # Convert to array for prediction
        features = np.array([[input_data['age'], input_data['sex'], input_data['cp'], 
                             input_data['trtbps'], input_data['chol'], input_data['fbs'],
                             input_data['restecg'], input_data['thalachh'], input_data['exng'],
                             input_data['oldpeak'], input_data['slp']]])
        
        # Load model (you'll need to have attack_model.pkl)
        model = load_model("attack_model.pkl")
        if model is None:
            st.stop()
        
        with st.spinner("Calculando riesgo..."):
            prediction = model.predict(features)
            probability = model.predict_proba(features)[0][1]  # Probability of heart attack
            
            if prediction[0] == 1:
                st.error(f"⚠️ **Alto riesgo de ataque al corazón** ({probability*100:.1f}% probabilidad)")
                st.markdown("""
                **Recomendaciones:**
                - Consulte a un cardiólogo inmediatamente
                - Realice cambios en su estilo de vida
                - Monitoree sus síntomas regularmente
                """)
            else:
                st.success(f"✅ **Bajo riesgo de ataque al corazón** ({probability*100:.1f}% probabilidad)")
                st.markdown("""
                **Recomendaciones:**
                - Mantenga hábitos saludables
                - Realice chequeos regulares
                - Ejercicio moderado
                """)
            
            # Store in session state for history
            if 'history' not in st.session_state:
                st.session_state.history = []
                
            st.session_state.history.append({
                'Edad': age,
                'Sexo': sex,
                'Riesgo': "Alto" if prediction[0] == 1 else "Bajo",
                'Probabilidad': f"{probability*100:.1f}%"
            })
            
    except Exception as e:
        st.error(f"Error en la predicción: {str(e)}")

# ----------------------------- HISTORIAL ------------------------------------
if 'history' in st.session_state and st.session_state.history:
    st.subheader("📜 Historial de Evaluaciones")
    df_history = pd.DataFrame(st.session_state.history)
    st.dataframe(df_history, hide_index=True)
    
    # Clear history button
    if st.button("🧹 Limpiar historial"):
        st.session_state.history = []
        st.rerun()