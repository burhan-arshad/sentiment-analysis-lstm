# IMDB Sentiment Analysis

A deep learning sentiment analysis project that uses an LSTM neural network to classify IMDB movie reviews as positive or negative.

## Overview

This project processes movie reviews from the IMDB Dataset and uses an LSTM-based neural network to learn sentiment patterns in the text.

The trained model is also deployed through a Streamlit web application where users can enter their own movie reviews and receive a sentiment prediction with confidence.

## Features

* Text preprocessing and cleaning
* Word tokenization
* Vocabulary limited to 20,000 words
* Sequence padding and truncation
* Trainable word embeddings
* LSTM-based sentiment classification
* Binary sentiment prediction
* Model evaluation using accuracy, precision, recall, and F1-score
* Confusion matrix visualization
* Streamlit deployment
* Example reviews for testing

## Dataset

The project uses the IMDB Dataset containing 50,000 movie reviews labeled as either positive or negative.

The dataset is not included in this repository.

## Preprocessing

The reviews are converted to lowercase and cleaned by removing:

* URLs
* HTML tags
* Email addresses
* Numbers
* Punctuation
* Extra whitespace

The cleaned reviews are tokenized and converted into integer sequences.

The vocabulary is limited to the 20,000 most frequent words.

Each sequence is padded or truncated to a maximum length of 200 tokens.

## Model Architecture

The model consists of:

```text
Embedding
    ↓
LSTM(64)
    ↓
Dropout(0.5)
    ↓
Dense(1, sigmoid)
```

### Model Configuration

* Vocabulary size: 20,000
* Embedding dimension: 128
* LSTM units: 64
* Dropout: 0.5
* Optimizer: Adam
* Loss function: Binary Crossentropy
* Batch size: 128
* Maximum epochs: 15
* Early stopping: Enabled

## Results

The model achieved approximately:

```text
Test Accuracy: 79.95%

Negative:
Precision: 0.78
Recall:    0.82
F1-score:  0.80

Positive:
Precision: 0.82
Recall:    0.77
F1-score:  0.80
```

The model achieved balanced performance across positive and negative sentiment classes.

## Project Structure

```text
imdb-sentiment-analysis/
│
├── app.py
├── imdb_lstm_model.keras
├── imdb_tokenizer.pkl
├── imdb_label_encoder.pkl
├── imdb_max_len.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

## Saved Files

### imdb_lstm_model.keras

The trained LSTM model used for sentiment prediction.

### imdb_tokenizer.pkl

The fitted tokenizer used to convert text into the same integer representation used during training.

### imdb_label_encoder.pkl

The fitted label encoder used to convert between sentiment labels and numerical classes.

### imdb_max_len.pkl

Stores the maximum sequence length used by the model.

## Streamlit Application

The Streamlit application loads the trained model and preprocessing objects and allows users to enter movie reviews.

The application:

1. Cleans the input review
2. Converts the review into a token sequence
3. Pads the sequence to the required length
4. Sends it to the LSTM model
5. Predicts positive or negative sentiment
6. Displays the prediction confidence

## Running the Project

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

## Example Reviews

### Positive

```text
This movie was absolutely amazing. The acting was brilliant, the story was engaging, and I enjoyed every minute of it.
```

### Negative

```text
This movie was terrible. The story was boring, the acting was weak, and I wasted two hours watching it.
```

## Technologies

* Python
* TensorFlow
* Keras
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Seaborn
* Streamlit

## Limitations

The model is trained on IMDB movie reviews and is designed for binary positive/negative sentiment classification.

An accuracy of approximately 80% means the model will still make incorrect predictions on some reviews, particularly ambiguous or context-heavy reviews.

## License

This project is intended for educational and portfolio purposes.
