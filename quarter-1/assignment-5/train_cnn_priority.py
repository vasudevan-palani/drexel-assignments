# train_cnn_priority.py
#
# Train a CNN to predict ticket PRIORITY from IT support descriptions
# using Kaggle "IT Support Ticket Data" (Parth Patil).
#
# 1) Put the CSV in the same folder.
# 2) Set DATA_PATH, TEXT_COL, LABEL_COL below to match your CSV.
# 3) Run:  python train_cnn_priority.py
#
# Outputs:
#   - priority_cnn_model.h5
#   - priority_tokenizer.pkl
#   - priority_label_encoder.pkl

import os
import numpy as np
import pandas as pd
from typing import List, Tuple

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.utils import to_categorical

import pickle

# -----------------------------
# CONFIG – CHANGE THESE
# -----------------------------
# Path to the Kaggle CSV from Parth Patil
DATA_PATH = "it_support_ticket_data.csv"   # e.g. "IT Support Ticket Data.csv"

# Column names in that CSV (check with: print(pd.read_csv(DATA_PATH).columns))
TEXT_COL  = "Body"   # e.g. "Description", "Issue_Description", "text"
LABEL_COL = "Priority"      # e.g. "Priority", "priority", "Ticket_Priority"

# Model / training hyperparameters
MAX_NUM_WORDS = 20000
MAX_SEQ_LEN   = 200
EMBED_DIM     = 64
TEST_SIZE     = 0.2
RANDOM_STATE  = 42
EPOCHS        = 20
BATCH_SIZE    = 20


# -----------------------------
# Data loading & preprocessing
# -----------------------------
def load_data(csv_path: str) -> Tuple[List[str], List[str]]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"{csv_path} not found. Make sure the Kaggle CSV is downloaded "
            f"and DATA_PATH is set correctly."
        )

    df = pd.read_csv(csv_path)

    if TEXT_COL not in df.columns:
        raise ValueError(
            f'CSV must contain a text column "{TEXT_COL}". '
            f"Current columns: {df.columns.tolist()}"
        )
    if LABEL_COL not in df.columns:
        raise ValueError(
            f'CSV must contain a label column "{LABEL_COL}". '
            f"Current columns: {df.columns.tolist()}"
        )

    # Drop rows with missing text or label
    df = df.dropna(subset=[TEXT_COL, LABEL_COL])

    texts = df[TEXT_COL].astype(str).tolist()
    labels = df[LABEL_COL].astype(str).tolist()
    return texts, labels


def prepare_tokenizer(texts: List[str]) -> Tokenizer:
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, oov_token="<UNK>")
    tokenizer.fit_on_texts(texts)
    return tokenizer


def texts_to_padded_sequences(tokenizer: Tokenizer, texts: List[str]) -> np.ndarray:
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(
        sequences,
        maxlen=MAX_SEQ_LEN,
        padding="post",
        truncating="post"
    )
    return padded


def encode_labels(labels: List[str]) -> Tuple[np.ndarray, LabelEncoder]:
    """
    Encode priority labels into integers.
    Works for both string labels (Low/Medium/High/Urgent)
    and numeric labels (1/2/3/4).
    """
    encoder = LabelEncoder()
    y_int = encoder.fit_transform(labels)
    return y_int, encoder


# -----------------------------
# CNN model
# -----------------------------
def build_cnn_model(vocab_size: int, num_classes: int) -> tf.keras.Model:
    model = Sequential()
    model.add(
        Embedding(
            input_dim=vocab_size,
            output_dim=EMBED_DIM,
            input_length=MAX_SEQ_LEN,
        )
    )
    model.add(Conv1D(filters=128, kernel_size=3, activation="relu"))
    model.add(GlobalMaxPooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )
    return model


# -----------------------------
# Main training flow
# -----------------------------
def main():
    print("Loading data...")
    texts, labels = load_data(DATA_PATH)

    print(f"Loaded {len(texts)} tickets.")
    print("Preparing tokenizer...")
    tokenizer = prepare_tokenizer(texts)
    X = texts_to_padded_sequences(tokenizer, texts)

    print("Encoding labels (PRIORITY)...")
    y_int, label_encoder = encode_labels(labels)
    num_classes = len(label_encoder.classes_)
    y_cat = to_categorical(y_int, num_classes=num_classes)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_cat,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_int,
    )

    vocab_size = min(MAX_NUM_WORDS, len(tokenizer.word_index) + 1)
    print(f"Vocab size: {vocab_size}, Num classes: {num_classes}")

    print("Building CNN model...")
    model = build_cnn_model(vocab_size, num_classes)

    print("Training CNN (priority classification)...")
    model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
    )

    print("Evaluating on test set...")
    y_test_int = np.argmax(y_test, axis=1)
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred_int = np.argmax(y_pred_probs, axis=1)

    acc = accuracy_score(y_test_int, y_pred_int)
    print("\n=== Priority Classification Metrics ===")
    print(f"Accuracy: {acc:.3f}")
    print("\nClassification report:")
    print(
        classification_report(
            y_test_int,
            y_pred_int,
            target_names=label_encoder.classes_,
        )
    )

    # Save model and artifacts
    model.save("priority_cnn_model.keras")
    with open("priority_tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)
    with open("priority_label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

    print("\nSaved:")
    print("- priority_cnn_model.keras")
    print("- priority_tokenizer.pkl")
    print("- priority_label_encoder.pkl")


if __name__ == "__main__":
    main()
