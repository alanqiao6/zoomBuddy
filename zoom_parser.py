#!/usr/bin/env python3
"""
zoom_parser.py

This module provides functions to:
  - Parse Zoom VTT transcript files and chat logs.
  - Combine and stem the parsed messages.
  - Train a multilabel classifier (using MLkNN) on a labeled dataset.
  - Use the trained classifier to assign categories to messages.
  - Compute semantic relevancy of each message to a lesson plan using SentenceTransformer.
  - Write the combined data to CSV.

File inputs are "I/O‑optional": you can pass either a file path (str) or a file-like object.

The fixed training dataset is expected to be located at "training_data.csv" in the same directory as this module.
"""

import os
import re
import csv
import argparse
import pickle  # For saving/loading the classifier model

# ============================
# PATCH: Fix NearestNeighbors error
# ============================
import sklearn.neighbors
_original_NearestNeighbors = sklearn.neighbors.NearestNeighbors
class PatchedNearestNeighbors(_original_NearestNeighbors):
    def __init__(self, k, **kwargs):
        self.k = k  # Save k for later use
        super().__init__(n_neighbors=k, **kwargs)
sklearn.neighbors.NearestNeighbors = PatchedNearestNeighbors
# ============================

# Import text processing and similarity libraries.
from nltk.stem.porter import PorterStemmer
from sentence_transformers import SentenceTransformer, util

# For the multilabel classifier.
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from skmultilearn.adapt import MLkNN
from sklearn.metrics import hamming_loss, accuracy_score

# Optionally, try to import extract_keywords from model_classifier.
try:
    from model_classifier import extract_keywords
except ImportError:
    extract_keywords = None

# ---------- Helper Function to Support File I/O Optionality ----------

def read_text(input_data):
    """
    Read text from either a file path (str) or a file-like object.
    Returns a string.
    """
    if hasattr(input_data, "read"):
        content = input_data.read()
        if isinstance(content, bytes):
            content = content.decode("utf-8")
        return content
    else:
        with open(input_data, "r", encoding="utf-8") as f:
            return f.read()

# ---------- Parsing Functions ----------

def timestamp_to_seconds(timestamp_str):
    """
    Convert a timestamp string (HH:MM:SS or HH:MM:SS.mmm) to seconds (float).
    """
    parts = timestamp_str.split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid timestamp format: {timestamp_str}")
    hours, minutes = int(parts[0]), int(parts[1])
    sec_parts = parts[2].split(".")
    seconds = int(sec_parts[0])
    milliseconds = int(sec_parts[1]) if len(sec_parts) == 2 else 0
    return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000

def parse_vtt(input_data):
    """
    Parse a VTT file (WebVTT format) and return a list of transcript entries.
    
    Each entry is a dict with keys: type, block_index, timestamp, time, end, speaker, text, raw_message_count.
    input_data can be a file path or file-like object.
    """
    content = read_text(input_data)
    blocks = content.strip().split("\n\n")
    if blocks and blocks[0].strip().startswith("WEBVTT"):
        blocks = blocks[1:]
    transcript = []
    for block in blocks:
        lines = block.strip().splitlines()
        if not lines:
            continue
        if re.match(r"^\d+$", lines[0].strip()):
            block_index = lines[0].strip()
            timestamp_line = lines[1].strip() if len(lines) > 1 else ""
            text_lines = lines[2:] if len(lines) > 2 else []
        else:
            block_index = ""
            timestamp_line = lines[0].strip()
            text_lines = lines[1:] if len(lines) > 1 else []
        start, end = None, None
        timestamp_parts = timestamp_line.split("-->")
        if len(timestamp_parts) == 2:
            start = timestamp_parts[0].strip()
            end = timestamp_parts[1].strip()
        text = " ".join(text_lines).strip()
        # Compute raw_message_count: count non-empty lines in text_lines.
        raw_message_count = sum(1 for line in text_lines if line.strip()) if text_lines else 1
        speaker = ""
        message = text
        if ":" in text:
            possible_speaker, possible_message = text.split(":", 1)
            speaker = possible_speaker.strip()
            message = possible_message.strip()
        transcript.append({
            "type": "transcript",
            "block_index": block_index,
            "timestamp": start,
            "time": timestamp_to_seconds(start) if start else None,
            "end": end,
            "speaker": speaker,
            "text": message,
            "raw_message_count": raw_message_count
        })
    return transcript

def parse_chat_log(input_data):
    """
    Parse a chat log file where each line is formatted as:
      timestamp[TAB]Speaker Name:[TAB]Message text
    input_data can be a file path or file-like object.
    Returns a list of chat entries (dicts) with keys: type, timestamp, time, speaker, message, raw_message_count.
    """
    content = read_text(input_data)
    chat_entries = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        timestamp = parts[0].strip()
        speaker = parts[1].strip()
        if speaker.endswith(":"):
            speaker = speaker[:-1].strip()
        message = parts[2].strip()
        chat_entries.append({
            "type": "chat",
            "timestamp": timestamp,
            "time": timestamp_to_seconds(timestamp),
            "speaker": speaker,
            "message": message,
            "raw_message_count": 1
        })
    return chat_entries

def combine_data(transcript, chat_entries):
    """
    Combine transcript and chat entries into one list sorted by the 'time' key.
    """
    combined = transcript + chat_entries
    combined.sort(key=lambda x: x.get("time", 0))
    return combined

def stem_text(text, stemmer=None):
    """
    Stem the words in the text using NLTK's PorterStemmer.
    """
    if stemmer is None:
        stemmer = PorterStemmer()
    words = re.findall(r'\w+', text.lower())
    stemmed_words = [stemmer.stem(word) for word in words]
    return " ".join(stemmed_words)

# ---------- Multilabel Classification Functions ----------

def generate_model():
    """
    Train a multilabel classifier using the training_data.csv file.
    Expects a CSV with a column "text" and one or more label columns.
   
    Returns a tuple: (vectorizer, classifier, label_names)
    If training_data.csv is missing or misformatted, returns None.
    """
    training_file = os.path.join(os.path.dirname(__file__), "training_data.csv")
    if not os.path.exists(training_file):
        print("Warning: training_data.csv not found. Skipping classifier training.")
        return None
    try:
        initial_df = pd.read_csv(training_file)
    except Exception as e:
        print(f"Error reading training data: {e}")
        return None
   
    if "text" not in initial_df.columns:
        print("Error: Training data must have a 'text' column.")
        return None
   
    X = initial_df["text"]
    label_names = list(initial_df.columns.difference(["text"]))
    if not label_names:
        print("Error: No label columns found in training data.")
        return None
    
    # Fill missing values and convert to binary (0/1)
    initial_df[label_names] = initial_df[label_names].fillna(0)
    y = (initial_df[label_names] > 0).astype(np.int64).values
   
    vectorizer = TfidfVectorizer(max_features=3000, max_df=0.85)
    X_tfidf = vectorizer.fit_transform(X)
    classifier = MLkNN(k=3)
    classifier.fit(X_tfidf, y)
    return vectorizer, classifier, label_names

def get_trained_model():
    """
    Load the trained classifier from a pickle file if it exists.
    Otherwise, train the model using generate_model(), save it to mlknn_model.pkl, and return it.
    """
    pkl_path = os.path.join(os.path.dirname(__file__), "mlknn_model.pkl")
    if os.path.exists(pkl_path):
        try:
            with open(pkl_path, "rb") as f:
                model = pickle.load(f)
            print("Loaded classifier from pickle.")
            return model
        except Exception as e:
            print(f"Error loading pickle file: {e}. Retraining model...")
    model = generate_model()
    if model is not None:
        try:
            with open(pkl_path, "wb") as f:
                pickle.dump(model, f)
            print("Trained model saved to pickle.")
        except Exception as e:
            print(f"Error saving pickle file: {e}")
    return model

def classify_message_ml(message, vectorizer, classifier, label_names):
    """
    Classify a message using the provided multilabel classifier.
    Returns a comma-separated string of predicted label names.
    If no label is predicted, returns "uncategorized".
    """
    X_vectorized = vectorizer.transform([message])
    prediction = classifier.predict(X_vectorized)
    prediction_array = prediction.toarray()[0]
    predicted_labels = [label for label, pred in zip(label_names, prediction_array) if pred == 1]
    if not predicted_labels:
        return "uncategorized"
    return ", ".join(predicted_labels)

# ---------- Semantic Similarity ----------

def compute_semantic_similarity(message, lesson_embedding, model):
    """
    Compute the cosine similarity between the message and the lesson plan.
    Returns a float between 0 and 1.
    """
    message_embedding = model.encode(message, convert_to_tensor=True)
    similarity = util.cos_sim(message_embedding, lesson_embedding)
    return float(similarity)

# ---------- CSV Writing ----------

def write_csv(data, output, classifier_model=None, lesson_embedding=None, semantic_model=None):
    """
    Write the combined data to CSV.
    'output' can be a file path (str) or a file-like object.
   
    If classifier_model is provided (tuple of (vectorizer, classifier, label_names)),
    each message is assigned a category using the multilabel classifier.
    If lesson_embedding and semantic_model are provided, lesson relevancy is computed.
    """
    fieldnames = ["type", "block_index", "timestamp", "time", "end", "speaker", "message", "stemmed_message", "raw_message_count"]
    if classifier_model is not None:
        fieldnames.append("assigned_category")
    if lesson_embedding is not None and semantic_model is not None:
        fieldnames.append("lesson_relevancy")
    rows = []
    stemmer = PorterStemmer()
    for entry in data:
        message = entry.get("text", "") if entry.get("type") == "transcript" else entry.get("message", "")
        stemmed_message = stem_text(message, stemmer)
        row = {
            "type": entry.get("type", ""),
            "block_index": entry.get("block_index", ""),
            "timestamp": entry.get("timestamp", ""),
            "time": entry.get("time", ""),
            "end": entry.get("end", ""),
            "speaker": entry.get("speaker", ""),
            "message": message,
            "stemmed_message": stemmed_message,
            "raw_message_count": entry.get("raw_message_count", 1)
        }
        if classifier_model is not None:
            vectorizer, classifier, label_names = classifier_model
            row["assigned_category"] = classify_message_ml(message, vectorizer, classifier, label_names)
        if lesson_embedding is not None and semantic_model is not None:
            row["lesson_relevancy"] = compute_semantic_similarity(message, lesson_embedding, semantic_model)
        rows.append(row)
    if hasattr(output, "write"):
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    else:
        with open(output, "w", encoding="utf-8", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

# ---------- High-Level Processing ----------

def process_zoom_data(vtt_input, chat_input, output, lesson_plan_input=None):
    """
    Process Zoom data by parsing transcript and chat log, combining entries,
    classifying each message using a trained multilabel classifier,
    optionally processing a lesson plan for semantic similarity,
    and writing CSV output.
   
    Parameters:
      vtt_input, chat_input, lesson_plan_input: file path (str) or file-like object.
      output: file path (str) or file-like object.
   
    Returns the combined list of entries.
    """
    transcript = parse_vtt(vtt_input)
    chat_log = parse_chat_log(chat_input)
    combined = combine_data(transcript, chat_log)
   
    # Use pickled classifier if available; otherwise, train.
    classifier_model = get_trained_model()
   
    lesson_embedding = None
    semantic_model = None
    if lesson_plan_input is not None:
        lesson_content = read_text(lesson_plan_input)
        semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        lesson_embedding = semantic_model.encode(lesson_content, convert_to_tensor=True)
   
    write_csv(combined, output, classifier_model, lesson_embedding, semantic_model)
    return combined

# ---------- CLI (For Testing) ----------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse Zoom transcript and chat log files, classify them using a multilabel model, and output a CSV file."
    )
    parser.add_argument("--vtt", required=True, help="Path to the VTT transcript file")
    parser.add_argument("--chat", required=True, help="Path to the chat log file")
    parser.add_argument("--output", required=True, help="Path to the output CSV file")
    parser.add_argument("--lesson", help="Optional path to the lesson plan text file")
    args = parser.parse_args()
   
    process_zoom_data(args.vtt, args.chat, args.output, args.lesson)
    print(f"Combined CSV output written to {args.output}")
