#!/usr/bin/env python3
"""
aggregate_engagement.py

This module provides functions to:
  - Load a student roster from a file (one student name per line).
  - Aggregate engagement metrics from a parsed CSV file (produced by zoom_parser.py).
  - Write the aggregated metrics to a CSV file.

File inputs are I/O‑optional (you can pass a file path or a file-like object).
"""

import csv

def read_text(input_data):
    """
    Read text from a file path or file-like object.
    Returns the text as a string.
    """
    if hasattr(input_data, "read"):
        content = input_data.read()
        if isinstance(content, bytes):
            content = content.decode("utf-8")
        return content
    else:
        with open(input_data, "r", encoding="utf-8") as f:
            return f.read()

def load_roster(input_data):
    """
    Load the roster of student names (one per line).
    If input_data is already a set, return it.
    Otherwise, convert all names to lower-case for matching.
    """
    if isinstance(input_data, set):
        return {name.lower() for name in input_data}
    content = read_text(input_data)
    students = set()
    for line in content.splitlines():
        name = line.strip().lower()
        if name:
            students.add(name)
    return students

def aggregate_engagement(input_csv, roster_input):
    """
    Aggregate engagement metrics from a parsed CSV file and a roster.
    
    Parameters:
      input_csv: parsed CSV file (file path or file-like object) from zoom_parser.py.
      roster_input: roster file (file path, file-like object, or a set of names).
    
    Returns:
      (student_stats, categories) where:
        - student_stats is a dict mapping student names to metrics.
        - categories is a list of expected engagement categories.
    """
    categories = ["environment", "community", "personal", "wonder", "tinker", "identity", "disruption"]
    roster_set = load_roster(roster_input)
    student_stats = {student: {"total_messages": 0, "lesson_relevancy_sum": 0.0, "relevancy_count": 0} for student in roster_set}
    for student in student_stats:
        for cat in categories:
            student_stats[student][cat] = 0
    content = read_text(input_csv)
    reader = csv.DictReader(content.splitlines())
    for row in reader:
        speaker = row.get("speaker", "").strip().lower()
        if speaker in student_stats:
            student_stats[speaker]["total_messages"] += 1
            relevancy_str = row.get("lesson_relevancy", "").strip()
            if relevancy_str:
                try:
                    relevancy = float(relevancy_str)
                    student_stats[speaker]["lesson_relevancy_sum"] += relevancy
                    student_stats[speaker]["relevancy_count"] += 1
                except ValueError:
                    pass
            assigned = row.get("assigned_category", "").strip().lower()
            if assigned and assigned != "uncategorized":
                cats = [cat.strip() for cat in assigned.split(",")]
                for cat in cats:
                    if cat in categories:
                        student_stats[speaker][cat] += 1
    for student, stats in student_stats.items():
        if stats["relevancy_count"] > 0:
            stats["avg_lesson_relevancy"] = stats["lesson_relevancy_sum"] / stats["relevancy_count"]
        else:
            stats["avg_lesson_relevancy"] = 0.0
    return student_stats, categories

def write_aggregate_to_csv(aggregate_data, categories, output):
    """
    Write the aggregated engagement metrics to a CSV file.
    'output' can be a file path or a file-like object.
    The CSV includes: student, total_messages, avg_lesson_relevancy, and one column per category.
    """
    fieldnames = ["student", "total_messages", "avg_lesson_relevancy"] + categories
    rows = []
    for student, stats in aggregate_data.items():
        row = {
            "student": student,
            "total_messages": stats["total_messages"],
            "avg_lesson_relevancy": stats["avg_lesson_relevancy"]
        }
        for cat in categories:
            row[cat] = stats[cat]
        rows.append(row)
    if hasattr(output, "write"):
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    else:
        with open(output, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Aggregate engagement metrics from a parsed CSV file and a roster file."
    )
    parser.add_argument("--input_csv", required=True, help="Path to the parsed CSV file.")
    parser.add_argument("--roster", required=True, help="Path to the roster file (one student per line).")
    parser.add_argument("--output_csv", required=True, help="Path to output aggregated CSV file.")
    args = parser.parse_args()
    aggregate_data, categories = aggregate_engagement(args.input_csv, args.roster)
    write_aggregate_to_csv(aggregate_data, categories, args.output_csv)
    print(f"Aggregated engagement metrics written to {args.output_csv}")
