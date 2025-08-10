from preprocessing import load_and_preprocess
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import pickle

# Load data
df = load_and_preprocess("D:/Medical Insurance Cost Prediction/data/medical_insurance.csv")
X = df.drop("charges", axis=1)
y = df["charges"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = XGBRegressor()
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("D:/Medical Insurance Cost Prediction/models/best_model.pkl", "wb"))
print("Model trained and saved.")
