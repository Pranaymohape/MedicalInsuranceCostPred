import joblib
import numpy as np
import streamlit as st

# Load the trained model
model = joblib.load("insurance_model.pkl")

# Streamlit Page Config
st.set_page_config(
    page_title="Medical Insurance Predictor", page_icon="🏥", layout="centered"
)

# Header
st.title("🏥 Medical Insurance Cost Prediction")
st.write("Fill in your personal details to get your estimated insurance cost..")
st.markdown("---")

# User Inputs
col1, col2 = st.columns(2)

with col1:
  age = st.number_input(
      "Your Age:", min_value=18, max_value=100, value=25
  )
  sex = st.selectbox("Gender:", ["Male", "Female"])
  bmi = st.number_input(
      "BMI (Body Mass Index):",
      min_value=10.0,
      max_value=50.0,
      value=22.5,
      step=0.1,
  )

with col2:
  children = st.number_input(
      "Number of children:", min_value=0, max_value=10, value=0
  )
  smoker = st.selectbox("Do you smoke:", ["No", "Yes"])
  region = st.selectbox(
      "Region:", ["Southwest", "Southeast", "Northwest", "Northeast"]
  )

# Converting Text Input into Numbers
sex_val = 1 if sex == "Male" else 0
smoker_val = 1 if smoker == "Yes" else 0
region_dict = {"Southwest": 0, "Southeast": 1, "Northwest": 2, "Northeast": 3}
region_val = region_dict[region]

st.markdown("---")

# Prediction Button
if st.button("Predict Insurance Cost 💵", use_container_width=True):
  input_data = np.array([[age, sex_val, bmi, children, smoker_val, region_val]])
  prediction = model.predict(input_data)[0]

  st.success(f"### Your estimated annual premium: **${prediction:,.2f}**")
  st.info("Note: This amount is mainly based on your age, BMI, and smoking habits.")