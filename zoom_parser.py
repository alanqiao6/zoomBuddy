#!/usr/bin/env python3
"""
zoom_parser.py

This module provides functions to:
  - Parse Zoom VTT transcript files and chat logs.
  - Combine and stem the parsed messages.
  - Load a CSV file of n‑grams and their associated categories (with word stemming for consistency).
  - Categorize each message by matching stemmed n‑grams.
  - Compute semantic relevancy of each message to a lesson plan using SentenceTransformer.
  - Write the combined data to CSV.

File inputs are "I/O‑optional": you can pass either a file path (str) or a file-like object.
If no n‑grams file is provided, it defaults to the fixed "ngrams.csv" file in the same directory.
"""

import re
import csv
import argparse
import os
from nltk.stem.porter import PorterStemmer
from sentence_transformers import SentenceTransformer, util

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
    Each entry is a dict with keys: type, block_index, timestamp, time, end, speaker, text.
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
            "text": message
        })
    return transcript

def parse_chat_log(input_data):
    """
    Parse a chat log file where each line is formatted as:
      timestamp[TAB]Speaker Name:[TAB]Message text
    input_data can be a file path or file-like object.
    Returns a list of chat entries (dicts) with keys: type, timestamp, time, speaker, message.
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
            "message": message
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

# ---------- N-grams and Categorization ----------

def load_ngrams(input_data):
    """
    Load n‑grams from a CSV file. Each row should have: id, phrase, ngram type, category.
    input_data can be a file path or file-like object.
    Returns a list of dicts with keys: phrase, ngram_type, category, stemmed_phrase, pattern.
    """
    content = read_text(input_data)
    ngrams = []
    stemmer = PorterStemmer()
    reader = csv.reader(content.splitlines())
    for row in reader:
        if len(row) < 4:
            continue
        _, phrase, ngram_type, category = row[0], row[1], row[2], row[3]
        stemmed_phrase = stem_text(phrase, stemmer)
        pattern = re.compile(r'\b' + re.escape(stemmed_phrase) + r'\b')
        ngrams.append({
            "phrase": phrase,
            "ngram_type": ngram_type,
            "category": category,
            "stemmed_phrase": stemmed_phrase,
            "pattern": pattern
        })
    return ngrams

def categorize_message(message, ngrams_list):
    """
    Categorize a message by checking for occurrence of any stemmed n‑grams.
    Returns the category (or comma‑separated categories in case of a tie) with the highest match count.
    If no n‑gram matches, returns "uncategorized".
    """
    stemmer = PorterStemmer()
    message_stemmed = stem_text(message, stemmer)
    counts = {}
    for ngram in ngrams_list:
        if ngram["pattern"].search(message_stemmed):
            cat = ngram["category"]
            counts[cat] = counts.get(cat, 0) + 1
    if not counts:
        return "uncategorized"
    max_count = max(counts.values())
    matched_categories = [cat for cat, cnt in counts.items() if cnt == max_count]
    return ", ".join(matched_categories)

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

def write_csv(data, output, ngrams_list=None, lesson_embedding=None, model=None):
    """
    Write the combined data to CSV.
    'output' can be a file path (str) or a file-like object.
    """
    fieldnames = ["type", "block_index", "timestamp", "time", "end", "speaker", "message", "stemmed_message"]
    if ngrams_list is not None:
        fieldnames.append("assigned_category")
    if lesson_embedding is not None and model is not None:
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
        }
        if ngrams_list is not None:
            row["assigned_category"] = categorize_message(message, ngrams_list)
        if lesson_embedding is not None and model is not None:
            row["lesson_relevancy"] = compute_semantic_similarity(message, lesson_embedding, model)
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

def process_zoom_data(vtt_input, chat_input, output, ngrams_input=None, lesson_plan_input=None):
    """
    Process Zoom data by parsing transcript and chat log, combining entries,
    optionally processing a lesson plan for semantic similarity,
    and writing CSV output.
    
    Parameters:
      vtt_input, chat_input, lesson_plan_input: file path (str) or file-like object.
      output: file path (str) or file-like object to write CSV.
      ngrams_input: if not provided, a fixed file "ngrams.csv" in the same directory is used.
    
    Returns the combined list of entries.
    """
    transcript = parse_vtt(vtt_input)
    chat_log = parse_chat_log(chat_input)
    combined = combine_data(transcript, chat_log)
    if ngrams_input is None:
        ngrams_input = os.path.join(os.path.dirname(__file__), "ngrams.csv")
    ngrams_list = load_ngrams(ngrams_input)
    lesson_embedding = None
    semantic_model = None
    if lesson_plan_input is not None:
        lesson_content = read_text(lesson_plan_input)
        semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        lesson_embedding = semantic_model.encode(lesson_content, convert_to_tensor=True)
    write_csv(combined, output, ngrams_list, lesson_embedding, semantic_model)
    return combined

# ---------- CLI (For Testing) ----------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse Zoom transcript and chat log files, process them, and output a CSV file."
    )
    parser.add_argument("--vtt", required=True, help="Path to the VTT transcript file")
    parser.add_argument("--chat", required=True, help="Path to the chat log file")
    parser.add_argument("--output", required=True, help="Path to the output CSV file")
    parser.add_argument("--ngrams", help="Optional path to the n‑grams CSV file (if not provided, uses fixed ngrams.csv)")
    parser.add_argument("--lesson", help="Optional path to the lesson plan text file")
    args = parser.parse_args()
    process_zoom_data(args.vtt, args.chat, args.output, args.ngrams, args.lesson)
    print(f"Combined CSV output written to {args.output}")
