import streamlit as st
import pickle
import numpy as np
import base64   # for encoding local image

# ✅ Background function
def add_bg_from_local(image_path):
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ✅ Call function before UI
add_bg_from_local("C:/Users/Admin/Desktop/Jansi/Project/Medical Insurance Cost Prediction/bg.jpg")

# Load trained model
model = pickle.load(open("models/best_model.pkl", "rb"))

# UI Starts
st.markdown(
    "<h1 style='color:darkblue; text-align:center; font-weight:bold;'>🏥 Medical Insurance Cost Predictor</h1>",
    unsafe_allow_html=True
)

# Inputs
st.markdown("<h3 style='color:#800000; font-weight:bold;'>Age</h3>", unsafe_allow_html=True)
age = st.slider("", 18, 100, 30)

st.markdown("<h3 style='color:#800000; font-weight:bold;'>Sex</h3>", unsafe_allow_html=True)
sex = st.selectbox("", ["male", "female"])

st.markdown("<h3 style='color:#800000; font-weight:bold;'>BMI</h3>", unsafe_allow_html=True)
bmi = st.slider("", 10.0, 50.0, 25.0)

st.markdown("<h3 style='color:#800000; font-weight:bold;'>Children</h3>", unsafe_allow_html=True)
children = st.selectbox("", range(0, 6))

st.markdown("<h3 style='color:#800000; font-weight:bold;'>Smoker</h3>", unsafe_allow_html=True)
smoker = st.selectbox("", ["yes", "no"])

st.markdown("<h3 style='color:#800000; font-weight:bold;'>Region</h3>", unsafe_allow_html=True)
region = st.selectbox("", ["northeast", "northwest", "southeast", "southwest"])


region_map = {"northeast": 0, "northwest": 1, "southeast": 2, "southwest": 3}

input_data = np.array([[age,
                        1 if sex == "male" else 0,
                        bmi,
                        children,
                        1 if smoker == "yes" else 0,
                        region_map[region]]])

if st.button("Predict"):
    prediction = model.predict(input_data)[0]

    st.markdown(
        f"""
        <div style="background-color:#d4edda;
                    padding:15px;
                    border-radius:10px;
                    border:1px solid #155724;
                    margin-top:10px;">
            <h3 style="color:#155724; font-size:24px; font-weight:bold;">
                Estimated Insurance Cost: 💰 ${prediction:.2f}
            </h3>
        </div>
        """,
        unsafe_allow_html=True
    )

