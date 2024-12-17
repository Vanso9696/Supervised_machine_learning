
# Streamlit app code starts here
import streamlit as st
import pandas as pd
import pickle

# Load the trained model
model_path = "/content/final_random_forest_model_with_features.pkl"
with open(model_path, "rb") as file:
    model = pickle.load(file)

st.title("Passenger Count Prediction App")

# File uploader
uploaded_file = st.file_uploader("Upload CSV", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:", data)

    # Ensure the correct features are used
    features = data.drop(columns=['Passenger_Count', 'Date'], errors='ignore')

    # Predictions
    predictions = model.predict(features)
    data['Predicted Passenger Count'] = predictions
    st.write("Predictions:", data)
