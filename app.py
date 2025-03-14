import streamlit as st
import joblib
import pandas as pd
import re
import os
import sys
import asyncio

# Fix asyncio loop issue for Python 3.10+
try:
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
except Exception as e:
    print("Could not set event loop policy:", e)

# Load the trained model
model = joblib.load("scam_detection_model.pkl")

# Function to preprocess user input (basic text cleaning)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"\W", " ", text)  # Remove special characters
    return text

# Streamlit UI
st.title("Social Media Scam Detection System")
st.subheader("Enter a message to check if it's a scam or not")

# Text input from user
user_input = st.text_area("Enter the text message here:")

if st.button("Check"):
    if user_input:
        # Preprocess text
        cleaned_text = preprocess_text(user_input)
        # Convert to DataFrame (assuming your model expects a dataframe input)
        data = pd.DataFrame([cleaned_text], columns=["message"])
        # Predict
        prediction = model.predict(data)
        # Display result
        if prediction[0] == 1:
            st.error("🚨 This message is likely a scam!")
        else:
            st.success("✅ This message seems safe!")
    else:
        st.warning("Please enter a message to check.")

# Button to restart Streamlit
if st.button("Restart App 🔄"):
    st.warning("Restarting the app...")
    os.execv(sys.executable, ['python'] + sys.argv)

