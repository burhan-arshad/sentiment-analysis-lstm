import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import re
import string

st.set_page_config(
    page_title="IMDB Sentiment Analysis",
    layout="centered"
)

MODEL_PATH = "imdb_lstm_model.keras"
TOKENIZER_PATH = "imdb_tokenizer.pkl"
ENCODER_PATH = "imdb_label_encoder.pkl"
MAX_LEN_PATH = "imdb_max_len.pkl"


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_resource
def load_tokenizer():
    with open(TOKENIZER_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_encoder():
    with open(ENCODER_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_max_len():
    with open(MAX_LEN_PATH, "rb") as f:
        return pickle.load(f)


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()

    return text


def predict_sentiment(text, model, tokenizer, encoder, max_len):
    cleaned_text = clean_text(text)

    sequence = tokenizer.texts_to_sequences([cleaned_text])

    padded_sequence = tf.keras.utils.pad_sequences(
        sequence,
        maxlen=max_len,
        padding="post",
        truncating="post"
    )

    probability = float(model.predict(padded_sequence, verbose=0)[0][0])

    predicted_class = int(probability >= 0.5)

    sentiment = encoder.inverse_transform([predicted_class])[0]

    if sentiment == "positive":
        confidence = probability
    else:
        confidence = 1 - probability

    return sentiment, confidence, probability


try:
    model = load_model()
    tokenizer = load_tokenizer()
    encoder = load_encoder()
    max_len = load_max_len()

except Exception as e:
    st.error("Model files could not be loaded.")
    st.exception(e)
    st.stop()


st.title("IMDB Sentiment Analysis")

st.write(
    "Enter a movie review and the LSTM model will predict whether the sentiment is positive or negative."
)

st.subheader("Enter Review")

review = st.text_area(
    "Movie Review",
    height=180,
    placeholder="Write a movie review here..."
)

example_reviews = {
    "Positive Example": "This movie was absolutely amazing. The acting was brilliant, the story was engaging, and I enjoyed every minute of it.",
    "Negative Example": "This movie was terrible. The story was boring, the acting was weak, and I wasted two hours watching it."
}

st.subheader("Testing Examples")

example_choice = st.selectbox(
    "Choose an example",
    ["None"] + list(example_reviews.keys())
)

if example_choice != "None":
    st.text_area(
        "Example Review",
        value=example_reviews[example_choice],
        height=120
    )

    if st.button("Use Example"):
        st.session_state.review = example_reviews[example_choice]

if "review" not in st.session_state:
    st.session_state.review = review

if st.session_state.review:
    review = st.session_state.review


if st.button("Analyze Sentiment", type="primary"):
    if not review.strip():
        st.warning("Please enter a movie review.")

    else:
        sentiment, confidence, probability = predict_sentiment(
            review,
            model,
            tokenizer,
            encoder,
            max_len
        )

        st.divider()

        st.subheader("Prediction")

        if sentiment == "positive":
            st.success("Positive Sentiment")
        else:
            st.error("Negative Sentiment")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Sentiment",
                sentiment.capitalize()
            )

        with col2:
            st.metric(
                "Confidence",
                f"{confidence * 100:.2f}%"
            )

        st.progress(confidence)

        with st.expander("Review Details"):
            st.write("Original Review")
            st.write(review)

            cleaned_review = clean_text(review)

            st.write("Cleaned Review")
            st.write(cleaned_review)

        with st.expander("Model Information"):
            st.write("Model: LSTM")
            st.write("Vocabulary Size: 20,000")
            st.write("Embedding Dimension: 128")
            st.write("LSTM Units: 64")
            st.write(f"Maximum Sequence Length: {max_len}")
            st.write("Output: Binary Sentiment Classification")