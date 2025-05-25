# import streamlit as st
# import pytesseract
# import cv2
# import numpy as np
# from PIL import Image
# import re
# import joblib
# import openai

# # Set your Tesseract path
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# # Set page config
# st.set_page_config(page_title="Medical OCR Lab Analyzer")
# st.title("📄 Medical Report Analyzer")
# st.markdown("Upload a lab report image to extract and analyze test results.")

# # Upload
# uploaded_file = st.file_uploader("Upload a lab report (JPG, PNG)", type=["jpg", "jpeg", "png"])

# # Load model & encoder
# @st.cache_resource
# def load_model_and_encoder():
#     model = joblib.load("models/trained_model.pkl")
#     encoder = joblib.load("models/label_encoder.pkl")
#     return model, encoder

# model, encoder = load_model_and_encoder()

# # GPT setup (replace key)
# openai.api_key = "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"
# def explain_with_gpt(prompt):
#     try:
#         response = openai.chat.completions.create(
#             model="gpt-3.5-turbo",
#             messages=[{"role": "user", "content": prompt}]
#         )
#         return response.choices[0].message.content
#     except Exception as e:
#         return f"❌ GPT Error: {str(e)}"

# # Predict
# def predict_health_status(inputs):
#     X = np.array(inputs).reshape(1, -1)
#     prediction = model.predict(X)
#     label = encoder.inverse_transform(prediction)[0]
#     return label

# # OCR section
# if uploaded_file is not None:
#     st.image(uploaded_file, caption="Uploaded Report", use_column_width=True)
#     image = Image.open(uploaded_file)
#     image_np = np.array(image)
#     gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
#     extracted_text = pytesseract.image_to_string(gray)

#     st.subheader("📝 Extracted Text")
#     st.text(extracted_text)

#     def parse_lab_results(text):
#         results = []
#         for line in text.split("\n"):
#             line = line.strip()
#             if line == "" or not any(char.isdigit() for char in line):
#                 continue
#             match = re.search(r"(.+?)\s+([\d.]+)\s+(\w+/dL|\w+)", line)
#             if match:
#                 results.append({
#                     "Test": match.group(1).strip(),
#                     "Value": match.group(2).strip(),
#                     "Unit": match.group(3).strip()
#                 })
#         return results

#     parsed_results = parse_lab_results(extracted_text)

#     st.subheader("📊 Structured Lab Results")
#     if parsed_results:
#         for item in parsed_results:
#             st.markdown(f"**{item['Test']}**: {item['Value']} {item['Unit']}")
#             prompt = f"Explain what it means if the patient's {item['Test']} is {item['Value']} {item['Unit']}."
#             explanation = explain_with_gpt(prompt)
#             st.info(explanation)
#     else:
#         st.info("No lab values could be identified.")

#     # ML Prediction (using example values)
#     st.subheader("🤖 ML Prediction (Example)")
#     example_inputs = [45, 190, 120, 99]  # Replace this with real parsed data in future
#     predicted_label = predict_health_status(example_inputs)
#     st.success(f"Predicted Health Status: **{predicted_label}**")


# -----------===============

import streamlit as st
import pytesseract
import cv2
import numpy as np
from PIL import Image
import re
import joblib
import google.generativeai as genai
import os

# ---------------- CONFIG ----------------

# Tesseract Path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Streamlit Config
st.set_page_config(page_title="Medical OCR Lab Analyzer")
st.title("📄 Medical Report Analyzer")
st.markdown("Upload a lab report image to extract and analyze test results.")

# Set Gemini API key
genai.configure(api_key="AIzaSyAPU7HkZLXX8q2EWOr9YRIsCh626CqRJeQ")  # 🔐 Replace with your key

# Load model & encoder
@st.cache_resource
def load_model_and_encoder():
    model = joblib.load("models/trained_model.pkl")
    encoder = joblib.load("models/label_encoder.pkl")
    return model, encoder

model, encoder = load_model_and_encoder()

# ---------------- Functions ----------------

def explain_with_gemini(prompt):
    try:
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ Gemini Error: {str(e)}"

def predict_health_status(inputs):
    X = np.array(inputs).reshape(1, -1)
    prediction = model.predict(X)
    label = encoder.inverse_transform(prediction)[0]
    return label

def parse_lab_results(text):
    results = []
    for line in text.split("\n"):
        line = line.strip()
        if line == "" or not any(char.isdigit() for char in line):
            continue
        match = re.search(r"(.+?)\s+([\d.]+)\s+(\w+/dL|\w+)", line)
        if match:
            results.append({
                "Test": match.group(1).strip(),
                "Value": match.group(2).strip(),
                "Unit": match.group(3).strip()
            })
    return results

# ---------------- Upload OCR + Prediction ----------------

uploaded_file = st.file_uploader("Upload a lab report (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Report", use_column_width=True)
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    extracted_text = pytesseract.image_to_string(gray)

    st.subheader("📝 Extracted Text")
    st.text(extracted_text)

    parsed_results = parse_lab_results(extracted_text)

    st.subheader("📊 Structured Lab Results")
    if parsed_results:
        for item in parsed_results:
            st.markdown(f"**{item['Test']}**: {item['Value']} {item['Unit']}")
            prompt = f"Explain what it means if the patient's {item['Test']} is {item['Value']} {item['Unit']}."
            explanation = explain_with_gemini(prompt)
            st.info(explanation)
    else:
        st.info("No lab values could be identified.")

    # 




    st.subheader("🤖 ML Prediction (Example)")
    example_inputs = [45, 190, 120, 99]  # Replace with parsed values later
    predicted_label = predict_health_status(example_inputs)
    st.success(f"Predicted Health Status: **{predicted_label}**")
