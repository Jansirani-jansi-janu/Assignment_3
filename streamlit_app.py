import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("models/best_model.pkl", "rb"))

st.title("🏥 Medical Insurance Cost Predictor")

age = st.slider("Age", 18, 100, 30)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.slider("BMI", 10.0, 50.0, 25.0)
children = st.selectbox("Children", range(0, 6))
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

region_map = {"northeast": 0, "northwest": 1, "southeast": 2, "southwest": 3}

input_data = np.array([[age,
                        1 if sex == "male" else 0,
                        bmi,
                        children,
                        1 if smoker == "yes" else 0,
                        region_map[region]]])

if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    st.success(f"Estimated Insurance Cost: 💰 ${prediction:.2f}")
