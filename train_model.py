import mlflow
import mlflow.sklearn   # for saving scikit-learn compatible models
from preprocessing import load_and_preprocess
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load data
df = load_and_preprocess("C:/Users/Admin/Desktop/Jansi/Project/Medical Insurance Cost Prediction/dataset/medical_insurance.csv")
X = df.drop("charges", axis=1)
y = df["charges"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start MLflow run
with mlflow.start_run():

    # Train model
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    # Predict & evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Log parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("learning_rate", 0.1)
    mlflow.log_param("max_depth", 5)

    # Log metrics
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)

    # Log model
    mlflow.sklearn.log_model(model, "insurance_model")

print("Training complete. Run 'mlflow ui' to see the results.")
