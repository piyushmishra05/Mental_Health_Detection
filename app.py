"""Entry point for launching an IPython kernel.

This is separate from the ipykernel package so we can avoid doing imports until
after removing the cwd from sys.path.
"""
import streamlit as st
import pickle
import re

# Load model & vectorizer
model = pickle.load(open("mental_health_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Title
st.title("🧠 Mental Health Detection App")

# Input
user_input = st.text_area("Enter your text")

# Clean function
def clean_text(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    return text

# Prediction
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        cleaned = clean_text(user_input)

        # 🔥 IMPORTANT FIX
        vec = vectorizer.transform([cleaned])

        prediction = model.predict(vec)

        if prediction[0] == 1:
            st.error("⚠️ Possible Depression Indicator")
        else:
            st.success("✅ No Depression Indicator")
            
            
# import sys

# if __name__ == "__main__":
    # Remove the CWD from sys.path while we load stuff.
    # This is added back by InteractiveShellApp.init_path()
#    if sys.path[0] == "":
#        del sys.path[0]

#    from ipykernel import kernelapp as app

#    app.launch_new_instance()
