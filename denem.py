import os
import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer, TFAutoModelForSeq2SeqLM

# TensorFlowu GPU yada Cpu modinda calistirmal icin olan kod
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def load_models():



        st.session_state["tokenizer_solution"] = AutoTokenizer.from_pretrained(
            "google/flan-t5-base"
        )
        st.session_state["model_solution"] = TFAutoModelForSeq2SeqLM.from_pretrained(
            "google/flan-t5-base"
        )


def generate_solution(log_message, log_level, source, timestamp):
    print(f"Generating detailed solution for: {log_message} (Level: {log_level}, Source: {source}, Timestamp: {timestamp})")
    tokenizer_solution = st.session_state["tokenizer_solution"]
    model_solution = st.session_state["model_solution"]

    prompt = f"""
    Review the  the log message: '{log_message}'  and provide a step-by-step solution to fix it.Dont give 
    the log message as answer just give some basic solutions
    """

    inputs = tokenizer_solution(
        prompt, return_tensors="tf", padding=True, truncation=True, max_length=300
    )
    outputs = model_solution.generate(
        **inputs, max_length=5000, temperature=0.9, top_p=0.95, repetition_penalty=2.0
    )
    solution = tokenizer_solution.decode(outputs[0], skip_special_tokens=True).strip()

    print("Generated Solution:", solution)
    return solution

def parse_log_line(line):
    log_pattern = r"^(\d{2}/\d{2} \d{2}:\d{2}:\d{2})\s+(\w+)\s*:([^:]*):(.*)$"
    match = re.search(log_pattern, line)

    if match:
        timestamp, level, source, message = match.groups()
        return [level.strip(), message.strip(), source.strip(), timestamp]
    else:
        return None

# Streamlit UI
st.title("AI-Powered Log Analyzer")
st.write("Upload a log file to analyze your logs with FLAN-T5.")

load_models()

uploaded_file = st.file_uploader("Upload Log File", type=["txt", "log"])
if uploaded_file is not None:
    with st.spinner("Processing uploaded file..."):
        lines = uploaded_file.readlines()
        log_data = []
        for line in lines:
            line = line.decode("utf-8").strip()
            parsed_line = parse_log_line(line)
            if parsed_line:
                log_data.append(parsed_line)
        df = pd.DataFrame(log_data, columns=["Level", "Message", "Source", "Timestamp"])
        st.session_state["logs"] = df
    st.success("File processed!")

if "logs" in st.session_state:
    df = st.session_state["logs"].copy()
    with st.spinner("Making AI predictions..."):
        print("Starting AI Predictions...")
        df["AI_Solution"] = df.apply(
            lambda row: generate_solution(row["Message"], row["Level"], row["Source"], row["Timestamp"])
            if row["Level"] != "INFO"
            else "No solution needed for INFO level",
            axis=1,
        )
        print("Finished solution generation.")

    st.write("### Log Data with AI Predictions")
    st.dataframe(df)

    st.write("### Log Level Distribution")
    fig, ax = plt.subplots()
    df["Level"].value_counts().plot(
        kind="bar", ax=ax, color=["blue", "orange", "red", "purple", "green", "brown"]
    )
    st.pyplot(fig)

    st.write("### Critical Logs")
    critical_logs = df[df["Level"] == "CRITICAL"]
    if not critical_logs.empty:
        st.dataframe(critical_logs)
    else:
        st.write("No critical logs found")

    st.write("### Warnings")
    warning_logs = df[df["Level"] == "WARNING"]
    if not warning_logs.empty:
        st.dataframe(warning_logs)
    else:
        st.write("No warning logs found")