import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime
import os

# Streamlit setup
st.set_page_config(page_title="Diabetes Predictor", layout="centered")
st.title("🩺 Diabetes Predictor App")
st.markdown("Enter patient details to predict the risk of diabetes.")

# Load dataset
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
               'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    df = pd.read_csv(url, header=None, names=columns)
    for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
        median_val = df[col].median()
        df[col] = df[col].replace(0, median_val)
    return df

df = load_data()

# Model training
X = df.drop("Outcome", axis=1)
y = df["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_scaled, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test_scaled))

# Sidebar input without +/-
st.sidebar.header("Enter Patient Data:")
feature_ranges = {
    'Pregnancies': (0, 30),
    'Glucose': (44, 300),
    'BloodPressure': (30, 200),
    'SkinThickness': (7, 99),
    'Insulin': (14, 846),
    'BMI': (15.0, 67.0),
    'DiabetesPedigreeFunction': (0.05, 2.5),
    'Age': (18, 150)
}

user_input = {}
error_flag = False

for feature, (min_val, max_val) in feature_ranges.items():
    val = st.sidebar.text_input(f"{feature} (range: {min_val}-{max_val})")
    try:
        if val == "":
            error_flag = True
        elif isinstance(min_val, int):
            val = int(val)
        else:
            val = float(val)
        if val < min_val or val > max_val:
            st.sidebar.warning(f"{feature} must be between {min_val} and {max_val}")
            error_flag = True
        else:
            user_input[feature] = val
    except:
        if val != "":
            st.sidebar.warning(f"Enter a valid number for {feature}")
            error_flag = True
        else:
            error_flag = True

# Predict
if st.sidebar.button("Predict"):
    if error_flag or len(user_input) < len(feature_ranges):
        st.warning("⚠️ Please provide valid values for all fields before predicting.")
    else:
        input_df = pd.DataFrame(user_input, index=[0])
        input_scaled = scaler.transform(input_df)
        result = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]

        if result == 1:
            st.error("⚠️ The model predicts a **HIGH risk** of diabetes.")
        else:
            st.success("✅ The model predicts a **LOW risk** of diabetes.")

        st.metric("Model Accuracy", f"{accuracy * 100:.2f}%")

        # Save result
        input_df['Prediction'] = "High Risk" if result == 1 else "Low Risk"
        input_df['Probability'] = round(prob, 3)
        input_df['Timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        result_file = "diabetes_predictions.csv"
        if not os.path.exists(result_file):
            input_df.to_csv(result_file, index=False)
        else:
            input_df.to_csv(result_file, mode='a', header=False, index=False)

        # Show history
        st.markdown("---")
        st.subheader("📋 Last 5 Predictions")
        history = pd.read_csv(result_file)
        st.dataframe(history.tail(5))

# Footer
st.markdown("---")
st.caption("Stay aware. Stay safe. Stay strong ❤️")

# streamlit run diabetes_predictor_streamlit.py