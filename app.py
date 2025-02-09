#!/usr/bin/env python3
from flask import Flask, request, render_template, redirect, url_for, send_from_directory, flash
import os
import tempfile
from zoom_parser import process_zoom_data
from aggregate_engagement import load_roster, aggregate_engagement, write_aggregate_to_csv

app = Flask(__name__)
app.secret_key = "supersecretkey"  # Change this for production!
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB file limit

def save_uploaded_file(file_storage):
    """
    Save an uploaded file (Flask FileStorage) to the UPLOAD_FOLDER and return its path.
    """
    filename = file_storage.filename
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file_storage.save(path)
    return path

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload')
def upload():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/process', methods=['POST'])
def process_files():
    # Retrieve required files.
    transcript_file = request.files.get("transcript")
    chat_file = request.files.get("chat")
    roster_file = request.files.get("roster")
    lesson_file = request.files.get("lesson")  # Optional
    
    if not (transcript_file and chat_file and roster_file):
        flash("Transcript, chat log, and roster files are required.", "danger")
        return redirect(url_for('upload'))
    
    # Save files.
    transcript_path = save_uploaded_file(transcript_file)
    chat_path = save_uploaded_file(chat_file)
    roster_path = save_uploaded_file(roster_file)
    lesson_path = save_uploaded_file(lesson_file) if lesson_file else None
    
    # Process Zoom data (the fixed n‑grams file is used automatically).
    parsed_csv_path = os.path.join(app.config['UPLOAD_FOLDER'], "parsed_output.csv")
    process_zoom_data(transcript_path, chat_path, parsed_csv_path, lesson_plan_input=lesson_path)
    
    # Aggregate engagement metrics.
    roster_set = load_roster(roster_path)
    aggregate_data, categories = aggregate_engagement(parsed_csv_path, roster_set)
    
    # Save aggregate CSV for download.
    aggregate_csv_path = os.path.join(app.config['UPLOAD_FOLDER'], "aggregate_output.csv")
    write_aggregate_to_csv(aggregate_data, categories, aggregate_csv_path)
    
    return render_template('result.html', aggregate_data=aggregate_data, categories=categories)

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
