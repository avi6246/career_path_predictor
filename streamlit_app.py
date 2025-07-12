import streamlit as st
import pickle
import numpy as np

# Load model
try:
    model, target_names = pickle.load(open('career_model.pkl', 'rb'))
except:
    st.error("⚠️ Model file not found. Please run model.py first to train and save the model.")
    st.stop()

# App Layout
st.set_page_config(page_title="Career Path Predictor", page_icon="🎯", layout="centered")

st.markdown(
    "<h1 style='text-align: center; color: #4B8BBE;'>🎯 Career Path Predictor</h1>", unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center; color: #666666;'>Predict the best career path based on your marks and interests</p>",
    unsafe_allow_html=True
)

# Input Form
with st.form("career_form"):
    age = st.slider("📅 Age", 15, 30, 18)
    math = st.slider("➗ Math Marks", 0, 100, 75)
    eng = st.slider("📝 English Marks", 0, 100, 70)
    interest = st.selectbox("🎯 Your Interest Area", ["Engineering", "Medical", "Design", "IT", "Management"])
    submit = st.form_submit_button("🚀 Predict")

interest_map = {
    "Engineering": 0,
    "Medical": 1,
    "Design": 2,
    "IT": 3,
    "Management": 4
}

# Predict Button
if submit:
    input_data = np.array([[age, math, eng, interest_map[interest]]])
    pred = model.predict(input_data)[0]
    st.success(f"✅ Recommended Career Path: **{target_names[pred]}**")
    st.balloons()

