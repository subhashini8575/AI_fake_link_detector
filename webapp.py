import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="AI LinkGuard",
    page_icon="🛡️",
    layout="centered"
)

# ---------------- CUSTOM STYLE ----------------
st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(circle at top left, #243b55 0%, transparent 35%),
        radial-gradient(circle at bottom right, #141e30 0%, transparent 40%),
        linear-gradient(135deg, #070b14, #101827);
    color: white;
}

.block-container {
    max-width: 850px;
    padding-top: 35px;
}

.hero {
    text-align: center;
    padding: 25px 10px 10px 10px;
}

.shield {
    font-size: 70px;
    margin-bottom: 5px;
}

.title {
    font-size: 48px;
    font-weight: 800;
    letter-spacing: 1px;
    margin: 0;
}

.subtitle {
    color: #b8c1d1;
    font-size: 18px;
    margin-top: 10px;
}

.badge {
    display: inline-block;
    margin-top: 18px;
    padding: 8px 18px;
    border-radius: 30px;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    color: #dce5f5;
    font-size: 14px;
}

.card {
    margin-top: 35px;
    padding: 35px;
    border-radius: 25px;
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.14);
    box-shadow: 0 20px 60px rgba(0,0,0,0.35);
}

.card-title {
    text-align: center;
    font-size: 25px;
    font-weight: 700;
    margin-bottom: 8px;
}

.card-subtitle {
    text-align: center;
    color: #9da8ba;
    margin-bottom: 25px;
}

.footer {
    text-align: center;
    color: #7f899b;
    margin-top: 35px;
    line-height: 1.8;
}

</style>
""", unsafe_allow_html=True)


# ---------------- HEADER ----------------

st.markdown("""
<div class="hero">

<div class="shield">🛡️</div>

<div class="title">AI LinkGuard</div>

<div class="subtitle">
🤖 Smart AI-Powered Fake Link Detection
</div>

<div class="badge">
🔐 Stay Safe • Think Before You Click
</div>

</div>
""", unsafe_allow_html=True)


# ---------------- LOAD DATA ----------------

data = pd.read_csv("links.csv")

vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 5)
)

X = vectorizer.fit_transform(data["url"])
y = data["label"]

model = LogisticRegression()
model.fit(X, y)


# ---------------- MAIN CARD ----------------

st.markdown("""
<div class="card">

<div class="card-title">
🔗 Scan Your Website Link
</div>

<div class="card-subtitle">
Paste a URL below and let AI analyse it.
</div>

</div>
""", unsafe_allow_html=True)


# ---------------- INPUT ----------------

url = st.text_input(
    "🌐 Website URL",
    placeholder="https://example.com"
)


# ---------------- BUTTON ----------------

if st.button("🚀  Scan Link", use_container_width=True):

    if url.strip():

        url_data = vectorizer.transform([url])

        prediction = model.predict(url_data)[0]

        st.write("")

        if prediction == 1:

            st.error(
                "🚨 SUSPICIOUS LINK DETECTED!"
            )

            st.warning(
                "⚠️ Be careful! This URL may be unsafe. "
                "Avoid entering passwords, banking details, "
                "or personal information."
            )

        else:

            st.success(
                "✅ THIS LINK LOOKS SAFE!"
            )

            st.info(
                "🔎 The AI model did not detect suspicious "
                "characteristics in this URL."
            )

    else:

        st.warning(
            "⚠️ Please enter a website link first."
        )


# ---------------- FOOTER ----------------

st.markdown("""
<div class="footer">

🧠 Machine Learning Based Security Tool<br>

🔐 AI LinkGuard • Protecting you from suspicious links<br>

✨ Stay Alert • Stay Safe • Browse Smart

</div>
""", unsafe_allow_html=True)