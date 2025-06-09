import streamlit as st
import numpy as np
import pickle
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model, tokenizer, and config
@st.cache_resource
def load_artifacts():
    model = load_model("D:\\Data Science\\NLP\\Day7_Duplicate_question_pair\\bilstm_duplicate_model.h5")
    with open("D:\\Data Science\\NLP\\Day7_Duplicate_question_pair\\tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_artifacts()

max_len = 30

st.title("Duplicate Question Detector (BiLSTM)")

# Persisting input
q1 = st.text_input("Enter Question 1", value=st.session_state.get("q1", ""))
q2 = st.text_input("Enter Question 2", value=st.session_state.get("q2", ""))

# Prediction logic
if st.button("Predict"):
    # Save input so it doesn’t disappear
    st.session_state.q1 = q1
    st.session_state.q2 = q2

    # Preprocess
    seq1 = tokenizer.texts_to_sequences([q1])
    seq2 = tokenizer.texts_to_sequences([q2])
    pad1 = pad_sequences(seq1, maxlen=max_len, padding='post')
    pad2 = pad_sequences(seq2, maxlen=max_len, padding='post')
    combined = np.hstack((pad1, pad2))

    # Predict
    pred = model.predict(combined)[0][0]
    label = "Duplicate" if pred > 0.5 else "Not Duplicate"

    st.markdown(f"**Prediction Score:** `{pred:.4f}`")
    st.success(label)