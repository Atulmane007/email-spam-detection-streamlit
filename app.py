import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import base64
import time
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix
import seaborn as sns

# -------- NEW IMPORTS FOR MULTI-LANGUAGE SUPPORT --------
from deep_translator import GoogleTranslator
from langdetect import detect

# -------------------------
# Background Image
# -------------------------

def set_bg(image_file):
    try:
        with open(image_file, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()

        page_bg = f"""
        <style>
        .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        }}
        </style>
        """

        st.markdown(page_bg, unsafe_allow_html=True)
    except:
        pass

set_bg("background.jpg")

st.markdown("""
<style>

/* Main content card */
.block-container{
    background: rgba(0, 0, 0, 0.88);
    padding: 40px;
    border-radius: 28px;
    max-width: 760px;
    margin-top: 70px;
    margin-left: auto;
    margin-right: auto;
    box-shadow: 0 10px 40px rgba(0,0,0,0.8);
}

.stApp{
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}

textarea, .stTextArea textarea{
    background-color: rgba(0,0,0,0.95) !important;
    color: white !important;
    border-radius: 16px;
}

.stFileUploader{
    background-color: rgba(0,0,0,0.92);
    padding: 12px;
    border-radius: 16px;
}

.stButton button{
    border-radius: 14px;
}

h1, h2, h3, h4, h5, h6, p, label{
    color: white;
}

</style>
""", unsafe_allow_html=True)

# -------------------------
# Load ML Model
# -------------------------

model = pickle.load(open("model/spam_model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

st.title("📧 Email Spam Detection System")
st.write("Machine Learning based system to detect spam messages.")
st.write("---")
st.write("Try Sample Email Data:")

# -------------------------
# Sidebar
# -------------------------

st.sidebar.title("Project Info")
st.sidebar.write("Model : Multinomial Naive Bayes")
st.sidebar.write("Vectorizer : TF-IDF")
st.sidebar.write("Dataset : SMS Spam Collection")
st.sidebar.write("Accuracy ≈ 96%")
st.sidebar.write("Language Support : English + Indian Languages")

# -------------------------
# Example Buttons
# -------------------------

col1, col2 = st.columns(2)

with col1:
    if st.button("🔴 Spam Example"):
        st.session_state.email = "Congratulations! You have won $5000. Click here to claim your prize now."

with col2:
    if st.button("✅ Normal Example"):
        st.session_state.email = "Hi Atul, let's meet tomorrow to discuss the project."

# -------------------------
# Email Input
# -------------------------

email_text = st.text_area(
    "Enter Email Message",
    value=st.session_state.get("email", "")
)

# -------------------------
# Upload Email File
# -------------------------

file = st.file_uploader("Upload Email (.txt)", type=["txt"])

if file:
    email_text = file.read().decode()

# -------------------------
# Spam Keywords
# -------------------------

spam_keywords = [
    "win", "winner", "free", "prize", "money", "claim",
    "click", "offer", "buy", "urgent", "cash", "reward",
    "lottery", "congratulations", "limited", "deal"
]

# -------------------------
# LANGUAGE NAME MAP
# -------------------------

language_map = {
    "en": "English",
    "hi": "Hindi",
    "mr": "Marathi",
    "gu": "Gujarati",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "ml": "Malayalam",
    "bn": "Bengali",
    "pa": "Punjabi",
    "ur": "Urdu",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "zh-cn": "Chinese",
    "ja": "Japanese",
    "ko": "Korean"
}

# -------------------------
# Prediction Section
# -------------------------

if st.button("Check Email"):

    if email_text.strip() == "":
        st.warning("Please enter an email message")

    else:

        # -------- LANGUAGE DETECTION --------
        try:
            detected_language = detect(email_text)
            language_name = language_map.get(detected_language, detected_language)
            st.info(f"🌐 Detected Language: {language_name}")
        except:
            detected_language = "unknown"

        # -------- TRANSLATE TO ENGLISH --------
        try:
            translated_text = GoogleTranslator(source='auto', target='en').translate(email_text)

            if translated_text != email_text:
                st.write("🔄 Translated Email (for ML Model):")
                st.success(translated_text)
        except:
            translated_text = email_text

        if "http" in email_text or "www" in email_text:
            st.warning("⚠ This email contains a link (possible phishing attempt)")

        with st.spinner("Analyzing email..."):
            time.sleep(5)

        vector = vectorizer.transform([translated_text])
        prediction = model.predict(vector)
        prob = model.predict_proba(vector)

        spam_prob = prob[0][1] * 100
        ham_prob = prob[0][0] * 100

        st.subheader("📊 Prediction Result")

        if prediction[0] == 1:
            st.error("⚠ Spam Email Detected")
        else:
            st.success("✅ Safe Email")

        st.write("Spam Probability:", f"{spam_prob:.2f}%")
        st.progress(int(spam_prob))
        st.write("Ham Probability:", f"{ham_prob:.2f}%")

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=spam_prob,
            title={'text': "Spam Confidence"},
            gauge={'axis': {'range': [0, 100]}}
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.subheader("🥧 Email Analysis Distribution")
        pie_fig = px.pie(
            names=["Spam", "Ham"],
            values=[spam_prob, ham_prob],
            title="Current Email Spam vs Ham Probability"
        )
        st.plotly_chart(pie_fig, use_container_width=True)

        st.subheader("📏 Email Length Insight")
        email_length = len(email_text)
        word_count = len(email_text.split())

        col1, col2 = st.columns(2)
        col1.metric("Characters", email_length)
        col2.metric("Words", word_count)

        if spam_prob < 30:
            level = "🟢 Low"
        elif spam_prob < 60:
            level = "🟡 Medium"
        elif spam_prob < 85:
            level = "🟠 High"
        else:
            level = "🔴 Critical"

        st.write("Spam Risk Level :", level)

        report = f"""
Email: {email_text}

Prediction: {"Spam" if prediction[0]==1 else "Ham"}
Spam Probability: {spam_prob:.2f}%
Risk Level: {level}
"""
        st.download_button("📥 Download Report", report, file_name="report.txt")

        st.session_state.pred = prediction[0]
        st.session_state.email = translated_text

# -------------------------
# AI Explanation
# -------------------------

if "pred" in st.session_state:

    if st.button("🤖 AI Explain Result"):

        st.subheader("🧠 AI Email Analysis")

        email = st.session_state.email.lower()

        detected_spam_words = []
        safe_words = []

        for word in email.split():
            if word in spam_keywords:
                detected_spam_words.append(word)
            else:
                safe_words.append(word)

        highlighted = email
        for word in spam_keywords:
            highlighted = highlighted.replace(
                word, f"<span style='color:red;font-weight:bold'>{word}</span>"
            )

        st.markdown("### 📝 Highlighted Email")
        st.markdown(highlighted, unsafe_allow_html=True)

        if st.session_state.pred == 1:
            st.write("""
The machine learning model classified this message as **Spam**.

Reasons:
• Promotional or reward language detected  
• Spam keyword patterns found  
• Similar patterns exist in training spam dataset
""")
        else:
            st.write("""
The machine learning model classified this message as **Safe (Ham)**.

Reasons:
• Normal conversational language  
• No suspicious promotional patterns  
• Similar to legitimate messages
""")

        st.write("### 🚨 Suspicious Words Detected")

        if detected_spam_words:
            st.error(", ".join(set(detected_spam_words)))
        else:
            st.success("No spam keywords detected")

        st.write("### 🟢 Normal Words")
        st.write(", ".join(safe_words[:20]))

        st.subheader("🧠 Word Type Distribution")
        spam_count = len(detected_spam_words)
        normal_count = len(safe_words)

        bar_fig = px.bar(
            x=["Spam Words", "Normal Words"],
            y=[spam_count, normal_count],
            title="Detected Word Types in Email"
        )
        st.plotly_chart(bar_fig, use_container_width=True)

st.write("---")
st.header("📊 Analysis & Model Insights")
st.caption("Visual insights based on dataset and model performance")

st.subheader("📈 Dataset Distribution")

try:
    data = pd.read_csv("dataset/spam.csv", encoding="latin-1")

    data = data.rename(columns={"v1": "label", "v2": "message"})
    data = data[["label", "message"]].dropna()
    data['message'] = data['message'].str.lower()

    counts = data['label'].value_counts()
    st.bar_chart(counts)

    st.subheader("📊 Model Performance (Sample Data)")

    sample_data = data.sample(min(300, len(data)))

    y_true = sample_data['label'].map({'ham':0, 'spam':1})
    y_pred = model.predict(vectorizer.transform(sample_data['message']))

    cm = confusion_matrix(y_true, y_pred)

    fig_cm, ax = plt.subplots()

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap="Blues",
        xticklabels=["Ham", "Spam"],
        yticklabels=["Ham", "Spam"],
        ax=ax
    )

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    ax.set_title("Confusion Matrix")

    st.pyplot(fig_cm)

    st.subheader("☁ Spam WordCloud")

    spam_words = " ".join(data[data['label'] == "spam"]['message'])
    wordcloud = WordCloud(width=800, height=400, background_color="black").generate(spam_words)

    fig2, ax2 = plt.subplots()
    ax2.imshow(wordcloud)
    ax2.axis("off")
    st.pyplot(fig2)

    st.subheader("☁ Safe Email WordCloud")

    ham_words = " ".join(data[data['label'] == "ham"]['message'])
    wordcloud2 = WordCloud(width=800, height=400, background_color="white").generate(ham_words)

    fig3, ax3 = plt.subplots()
    ax3.imshow(wordcloud2)
    ax3.axis("off")
    st.pyplot(fig3)

except Exception as e:
    st.error(f"Dataset error: {e}")

st.write("---")

st.header("⭐Future Improvements")

st.write("""
• Integration of Email Encryption (AES / RSA) to secure sensitive messages  
• End-to-End encrypted email scanning for secure spam detection  
• Advanced phishing URL detection using security APIs  
• Deep Learning models (LSTM / BERT) for improved spam classification accuracy  
• Secure email authentication using SPF, DKIM, and DMARC protocols  
""")

st.write("---")

st.markdown(
"""
<div style='text-align: center; color: gray; font-size: 14px;'>
🚀 Built with Streamlit & Scikit-Learn | v2.0
</div>
""",
unsafe_allow_html=True
)
