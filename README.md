# Zoom Engagement Analyzer

## Website and Video Demo
Here is the **deployed website**: [ZoomBuddy Website](https://zoombuddy.onrender.com/)

Here is a **video demo**: [ZoomBuddy Video Demo](https://duke.zoom.us/rec/share/prTUr6Ky9y9n91VzPgfjXJxHSAOuLGdrcI6aTXDU-NOgmTlK2V98aph2hU35pTQR.YZhrW6sG8s9opZbO?startTime=1741806209000)

## Overview

This project was built by four Duke undergraduates during HackDuke 2025 to support Ignite, a STEM program aimed at increasing STEM self-efficacy for students in Durham public schools.  

The **Zoom Engagement Analyzer** helps educators measure student engagement and participation during virtual Zoom sessions by analyzing:

- Chat logs
- Transcripts
- Lesson relevance

Using **Natural Language Processing (NLP) and Machine Learning**, the tool categorizes student interactions and quantifies engagement based on specific STEM-related themes. It provides detailed analytics to help educators track participation and lesson alignment.

---

## How It Works (Machine Learning & NLP)

This project applies Natural Language Processing (NLP) and Machine Learning (ML) to analyze Zoom session data.

### 1. Text Processing
- **Tokenization & Stemming** – Prepares text for classification by reducing words to their root form.
- **TF-IDF Vectorization** – Converts text into numerical features for machine learning models.

### 2. Multi-Label Classification (MLkNN)
A **multi-label classifier (MLkNN)** was trained on pre-labeled STEM-related messages to categorize messages into different engagement themes:
- **Community** – Mentions of teamwork, collaboration, and outreach.
- **Environment** – Discussions on sustainability, climate, and pollution.
- **Personal** – Self-reflection, personal experiences, and family-related mentions.
- **Hypothesize** – Scientific curiosity, "What if?" statements, and idea exploration.
- **Tinker** – Experimentation, prototyping, and problem-solving.
- **Identity** – Self-identification as a STEM learner or career aspirations.
- **Disruption** – Off-topic or unproductive messages.

### 3. Lesson Relevancy Scoring
- Uses **Sentence-BERT (SentenceTransformer)** to compute the semantic similarity between messages and the lesson plan.
- Assigns a **relevancy score** to each message, helping teachers gauge student focus.

---

## Project Structure  

### Backend (Flask App)
- **`app.py`** – Main Flask server that handles file uploads and processing.
- **`zoom_parser.py`** – Parses Zoom transcript (VTT) & chat logs, applies ML models.
- **`aggregate_engagement.py`** – Aggregates engagement metrics and generates analytics.

### Machine Learning & NLP
- **`training_data.csv`** – Labeled dataset used to train the ML model.
- **`mlknn_model.pkl`** – Pickled MLkNN classifier for message classification.
- **`ngrams.csv`** – List of key phrases used for categorization.

### Frontend (HTML, JavaScript)
- **`templates/`** – Flask templates for UI.
  - `home.html` – Landing page with demo video and instructions.
  - `index.html` – Upload page for Zoom transcript, chat logs, and lesson plan.
  - `result.html` – Displays interactive analytics, charts, and CSV download.
- **`static/`** – Stores CSS, JavaScript, and videos.

### Data Processing
- **`static/videos/demo.mp4`** – Demo video of the tool in action.
- **`static/aggregate_output.csv`** – Processed engagement report for download.

---

## Features & Data Visualization

### Interactive Dashboard
- **Table View** – Filterable and sortable student engagement table.
- **Class Analytics** – Total messages by category (bar chart).
- **Individual Student Analytics** – Select a student & view their engagement breakdown.
- **Engagement Histogram** – Shows the distribution of messages per student.

### File Upload & Processing
- Upload **Zoom transcript (VTT), chat logs, and student roster**.
- Optionally upload a **lesson plan** for relevancy scoring.

### Exportable Reports
- Download **aggregated engagement data (CSV)** for further analysis.

---

