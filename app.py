"""Entry point for launching an IPython kernel.

This is separate from the ipykernel package so we can avoid doing imports until
after removing the cwd from sys.path.
"""
import streamlit as st
import pickle
import re

# Page config
st.set_page_config(page_title="Mental Health Detection", page_icon="🧠", layout="centered")

# Load model & vectorizer
model = pickle.load(open("mental_health_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Title
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🧠 Mental Health Detection</h1>", unsafe_allow_html=True)

st.markdown("### Detect potential depression indicators from text using AI")

# Input box
user_input = st.text_area("✍️ Enter your text here:", height=150)

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    return text

# Predict button
if st.button("🔍 Analyze Text"):

    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text to analyze")
    else:
        cleaned = clean_text(user_input)

        # Vectorize
        vec = vectorizer.transform([cleaned])

        # Prediction
        prediction = model.predict(vec)[0]

        # Probability
        prob = model.predict_proba(vec)[0]

        confidence = max(prob) * 100

        st.markdown("---")

        # Result display
        if prediction == 1:
            st.error(f"⚠️ Possible Depression Indicator\n\nConfidence: {confidence:.2f}%")
        else:
            st.success(f"✅ No Depression Indicator\n\nConfidence: {confidence:.2f}%")

        # Progress bar
        st.progress(int(confidence))

        st.markdown("---")

        # Extra info
        st.markdown("### 📊 Model Insights")
        st.write("This prediction is based on NLP analysis of emotional patterns in text.")

 
 
 
 
           
            
# import sys

# if __name__ == "__main__":
    # Remove the CWD from sys.path while we load stuff.
    # This is added back by InteractiveShellApp.init_path()
#    if sys.path[0] == "":
#        del sys.path[0]

#    from ipykernel import kernelapp as app

#    app.launch_new_instance()
