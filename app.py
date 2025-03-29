import streamlit as st
import pandas as pd
import re
import random
import matplotlib.pyplot as plt
from transformers import pipeline

# Load AI Model
classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")


# Sample log data
def generate_sample_logs():
    log_levels = ["INFO", "WARNING", "ERROR", "CRITICAL"]
    log_messages = {
        "INFO": ["System boot completed", "User logged in", "Configuration loaded"],
        "WARNING": ["Disk space running low", "High memory usage detected"],
        "ERROR": ["File not found", "Database connection failed", "Permission denied"],
        "CRITICAL": ["System crash detected", "Kernel panic", "Critical security breach"]
    }

    logs = []
    for _ in range(50):
        level = random.choice(log_levels)
        message = random.choice(log_messages[level])
        logs.append([level, message])

    return pd.DataFrame(logs, columns=["Level", "Message"])


# AI Model for Log Analysis
def predict_log_category(log_message):
    result = classifier(log_message)[0]
    label = result['label']
    if "negative" in label.lower():
        return "ERROR"
    return "INFO"


def generate_solution(log_message):
    solutions = {
        "File not found": "Check if the file path is correct.",
        "Disk space running low": "Consider deleting unnecessary files or expanding disk space.",
        "Critical security breach": "Immediately review security logs and isolate affected systems.",
        "Database connection failed": "Verify database credentials and network connectivity."
    }

    for key in solutions:
        if key.lower() in log_message.lower():
            return solutions[key]
    return "No specific solution available. Check system documentation."


# Streamlit UI
st.title("AI-Powered Log Analyzer")
st.write("Upload a log file or use sample data to analyze logs with AI.")

if st.button("Use Sample Logs"):
    df = generate_sample_logs()
    st.session_state['logs'] = df

uploaded_file = st.file_uploader("Upload Log File", type=["txt", "log"])
if uploaded_file is not None:
    lines = uploaded_file.readlines()
    log_data = [re.split(r' - ', line.decode("utf-8").strip(), maxsplit=1) for line in lines]
    df = pd.DataFrame(log_data, columns=["Level", "Message"])
    st.session_state['logs'] = df

if 'logs' in st.session_state:
    df = st.session_state['logs']
    df['Predicted_Level'] = df['Message'].apply(predict_log_category)
    df['AI_Solution'] = df['Message'].apply(generate_solution)
    st.write("### Log Data with AI Predictions")
    st.dataframe(df)

    # Visualization
    st.write("### Log Level Distribution")
    fig, ax = plt.subplots()
    df['Predicted_Level'].value_counts().plot(kind='bar', ax=ax, color=['blue', 'orange', 'red', 'purple'])
    st.pyplot(fig)

    st.write("### Critical Logs")
    critical_logs = df[df['Predicted_Level'] == "CRITICAL"]
    st.dataframe(critical_logs)
