# Install necessary packages before running
# pip install streamlit scikit-learn xgboost imbalanced-learn pandas numpy

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv", header=None)
    df.columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
                  "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
    
    # Replace zeros with NaN for selected columns
    cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df[cols] = df[cols].replace(0, np.nan)
    df.fillna(df.median(), inplace=True)
    
    return df

# Train the model
@st.cache_resource
def train_model(df):
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
    
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    model.fit(X_train, y_train)
    
    return model, scaler

# Streamlit UI
st.set_page_config(page_title="Diabetes Prediction", layout="centered")
st.title("🔬 Diabetes Prediction App")
st.write("Enter the patient's details to predict diabetes status.")

# Input fields
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.slider("Glucose Level", 50, 200, 120)
blood_pressure = st.slider("Blood Pressure", 30, 130, 70)
skin_thickness = st.slider("Skin Thickness", 5, 100, 20)
insulin = st.slider("Insulin Level", 15, 846, 80)
bmi = st.slider("BMI", 10.0, 67.0, 28.0)
dpf = st.slider("Diabetes Pedigree Function", 0.1, 2.5, 0.5)
age = st.slider("Age", 15, 100, 33)

# Prepare data for prediction
user_input = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                        insulin, bmi, dpf, age]])

# Load and train
df = load_data()
model, scaler = train_model(df)

# Predict

if st.button("Predict", key="predict_button_1"):
    user_input_scaled = scaler.transform(user_input)
    prediction = model.predict(user_input_scaled)[0]
    
    if prediction == 1:
        st.error("🔴 The model predicts that the person **has diabetes.**")
        st.markdown("### 📋 Basic Prescription")
        st.write("""
        - 🍽️ **Diet**: Follow a low-carb, high-fiber, low-sugar diet.
        - 💊 **Medication**: Common medications include *Metformin* (consult a doctor before starting any).
        - 🏃‍♂️ **Exercise**: At least 30 minutes a day (walking, cycling, swimming).
        - 🧪 **Monitoring**: Regular blood glucose checks and HbA1c tests.
        - 💧 **Hydration**: Drink plenty of water.
        - ⚕️ **Doctor Visit**: Schedule a visit with an endocrinologist or diabetologist.
        """)

    else:
        st.success("🟢 The model predicts that the person **does NOT have diabetes.**")
        st.markdown("### ✅ Tips to Prevent Diabetes")
        st.write("""
        - 🍎 **Eat healthy**: Avoid sugary snacks and sodas, eat more fruits and vegetables.
        - 🏋️ **Stay active**: Exercise 4–5 days a week (a mix of cardio and strength training).
        - ⚖️ **Maintain healthy weight**: Losing even 5–10% of body weight reduces risk.
        - 🚭 **Avoid smoking** and limit alcohol.
        - 🛌 **Sleep well**: Aim for 7–9 hours of sleep daily.
        - 🧬 **Family history awareness**: Get regular checkups if diabetes runs in your family.
        """)


