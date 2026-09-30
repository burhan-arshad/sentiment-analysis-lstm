# IMDB Sentiment Analysis

A deep learning NLP project that uses an **LSTM neural network** to classify IMDB movie reviews as **positive or negative**. The trained model is deployed through a Streamlit application for real-time sentiment prediction.

## Overview

This project demonstrates an end-to-end sentiment analysis pipeline:

```text
Movie Review
     ↓
Text Cleaning
     ↓
Tokenization
     ↓
Sequence Padding
     ↓
Word Embedding
     ↓
LSTM Neural Network
     ↓
Positive / Negative Prediction
```

The Streamlit application allows users to enter their own movie reviews and receive a sentiment prediction with confidence.

## Features

* Text cleaning and preprocessing
* Word tokenization
* 20,000-word vocabulary
* Sequence padding and truncation
* Trainable word embeddings
* LSTM-based classification
* Binary sentiment prediction
* Accuracy, precision, recall, and F1-score evaluation
* Confusion matrix visualization
* Streamlit web application
* Example reviews for testing

## Dataset

The project uses the **IMDB Dataset**, containing **50,000 movie reviews** labeled as positive or negative.

The dataset is not included in this repository.

## Text Preprocessing

Reviews are converted to lowercase and cleaned by removing:

* URLs
* HTML tags
* Email addresses
* Numbers
* Punctuation
* Extra whitespace

The cleaned text is tokenized and converted into integer sequences.

The tokenizer is limited to the **20,000 most frequent words**, and sequences are padded or truncated to a maximum length of **200 tokens**.

## Model Architecture

```text
Input Text
    ↓
Embedding
    ↓
LSTM — 64 units
    ↓
Dropout — 0.5
    ↓
Dense — 1 unit
    ↓
Sigmoid
    ↓
Positive / Negative
```

### Configuration

| Parameter           | Value               |
| ------------------- | ------------------- |
| Vocabulary Size     | 20,000              |
| Embedding Dimension | 128                 |
| LSTM Units          | 64                  |
| Dropout             | 0.5                 |
| Optimizer           | Adam                |
| Loss                | Binary Crossentropy |
| Batch Size          | 128                 |
| Maximum Epochs      | 15                  |
| Early Stopping      | Enabled             |
| Sequence Length     | 200                 |

## Results

The model achieved approximately **79.95% test accuracy**.

| Class    | Precision | Recall | F1-Score |
| -------- | --------: | -----: | -------: |
| Negative |      0.78 |   0.82 |     0.80 |
| Positive |      0.82 |   0.77 |     0.80 |

The results show relatively balanced performance across both sentiment classes.

## Streamlit Application

The application loads the trained model and preprocessing objects to perform predictions on new reviews.

The prediction pipeline is:

1. Clean the input review
2. Tokenize the text
3. Convert text into integer sequences
4. Pad the sequence to 200 tokens
5. Pass the sequence through the LSTM model
6. Predict positive or negative sentiment
7. Display the prediction and confidence

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

## Saved Model Files

* **`imdb_lstm_model.keras`** — trained LSTM sentiment classification model
* **`imdb_tokenizer.pkl`** — fitted tokenizer for converting text into sequences
* **`imdb_label_encoder.pkl`** — label encoder for sentiment classes
* **`imdb_max_len.pkl`** — maximum sequence length used during training

Keeping these preprocessing objects with the model ensures that new reviews are processed consistently with the training data.

## Example

### Positive Review

```text
This movie was absolutely amazing. The acting was brilliant, the story was engaging, and I enjoyed every minute of it.
```

### Negative Review

```text
This movie was terrible. The story was boring, the acting was weak, and I wasted two hours watching it.
```

## Running Locally

Create a virtual environment:

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

The application will open locally in your browser.

## Technologies

* Python
* TensorFlow / Keras
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Seaborn
* Streamlit

## Limitations

The model performs **binary sentiment classification** and was trained specifically on IMDB movie reviews.

With approximately 80% test accuracy, the model will still produce incorrect predictions, especially for reviews containing sarcasm, mixed opinions, ambiguous language, or context that is difficult to infer from the text alone.

## Future Improvements

Possible improvements include:

* Bidirectional LSTM
* GRU-based architecture
* Pre-trained word embeddings
* Attention mechanisms
* Transformer-based sentiment models
* Hyperparameter tuning
* Larger and more diverse datasets

## License

This project is intended for educational and portfolio purposes.


## Author

**Burhan Arshad**
Computer Science Student
