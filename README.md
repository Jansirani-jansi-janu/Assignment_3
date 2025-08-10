# Assignment_3
This project predicts medical insurance costs using patient data such as age, sex, BMI, number of children, smoking status, and region. It preprocesses data with label encoding, trains an XGBoost regression model, and deploys an interactive Streamlit app, allowing users to input details and get instant cost estimates.

# 🏥 Medical Insurance Cost Prediction

This project predicts medical insurance costs based on patient details such as age, sex, BMI, number of children, smoking status, and region.  
It uses **XGBoost Regression** for training and **Streamlit** for an interactive web interface.

---

## 📂 Project Structure
Medical Insurance Cost Prediction/
│
├── data/
│ └── medical_insurance.csv
│
├── models/
│ └── best_model.pkl
│
├── scripts/
│ ├── preprocessing.py
│ └── train_model.py
│
├── streamlit_app.py
└── README.md


---

## ⚙️ Features
- **Data Preprocessing:** Label encoding for categorical features.
- **Model Training:** XGBoost Regressor for cost prediction.
- **Interactive UI:** Real-time predictions via Streamlit.
- **Custom Inputs:** Age, Sex, BMI, Children, Smoking Status, Region.

---

🛠 Requirements
Python 3.8+
scikit-learn
pandas
numpy
xgboost
streamlit

📊 Example
Example input:
Age: 40
Sex: Female
BMI: 26.5
Children: 1
Smoker: Yes
Region: Southeast
Predicted output:
💰 Estimated Insurance Cost: $25,800.50






