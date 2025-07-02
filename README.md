# ❤️ CardioSafe: AI-Powered Heart Attack Risk Prediction

![App Screenshot](./assets/app-preview.gif)  
*Interactive cardiac risk assessment with real-time results*

## 🏥 Clinical Insight at Your Fingertips

CardioSafe leverages machine learning to evaluate 11 critical health parameters, providing instant heart attack risk stratification with **89.5% accuracy**. Designed for:
- **Patients**: Proactive health awareness
- **Clinicians**: Rapid preliminary assessment
- **Researchers**: Model interpretability showcase

## ✨ Key Features

| Feature | Benefit |
|---------|---------|
| 🎚️ Multi-Parameter Input | Comprehensive risk evaluation (BP, cholesterol, ECG, etc.) |
| 🚦 Visual Risk Indicator | Color-coded alerts with pulse animation for high risk |
| 📋 Actionable Guidance | Tailored clinical recommendations |
| 🖥️ Responsive Design | Optimized for desktop & mobile |

## 🛠️ Technical Implementation

```mermaid
graph TD
    A[User Input] --> B(Data Preprocessing)
    B --> C{KNN Model}
    C --> D[Risk Prediction]
    D --> E[Visualization]
    E --> F[Clinical Guidance]


Core Stack:

scikit-learn 1.4.0 (KNN classifier)

streamlit 1.29.0 (Web interface)

Pillow 10.1.0 (Image processing)

streamlit-lottie 0.0.4 (Animations)

🚀 Deployment Guide
Local Installation


git clone https://github.com/RosanaNicklas/attack
cd Heartattack
pip install -r requirements.txt
streamlit run mein_app.py

Cloud Deployment
DE MOMENTO SIN HACER

📊 Model Performance Metrics
Metric	Score
Accuracy	89.5%
Precision	91.2%
Recall	87.8%
AUC-ROC	0.93
*Trained on UCI Heart Disease dataset (n=303)*

📜 Ethical Considerations
❗ Important Limitations:

Not FDA-approved

Should not guide treatment decisions

Population bias in training data

Requires clinical validation for individual use

📬 Contact

Rosana Longares
rosana8longares@gmail.com

Technical Support:
GitHub Issues

© 2025 CardioSafe AI | This tool is for educational purposes only
