import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# UI
st.set_page_config(page_title="Spam Detector", page_icon="📧")
st.title("📧 Spam Email / SMS Detector")
st.write("Enter a message below to check if it's **Spam** or **Not Spam**.")

user_input = st.text_area("✉️ Your Message", height=150, placeholder="Type or paste a message here...")

if st.button("🔍 Check"):
    if user_input.strip() == "":
        st.warning("Please enter a message first!")
    else:
        input_tfidf = vectorizer.transform([user_input])
        prediction = model.predict(input_tfidf)[0]
        probability = model.predict_proba(input_tfidf)[0]

        if prediction == 1:
            st.error(f"🚨 SPAM detected! (Confidence: {probability[1]*100:.1f}%)")
        else:
            st.success(f"✅ NOT Spam — looks safe! (Confidence: {probability[0]*100:.1f}%)")