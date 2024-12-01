import streamlit as st
import numpy as np
import pandas as pd
import sklearn
import sklearn.model_selection
import sklearn.linear_model
from sklearn.feature_extraction.text import CountVectorizer
from anchor import anchor_text
import spacy
import time

# Load Spacy model
nlp = spacy.load('en_core_web_sm')

# Functions from your code
def clean(data: str):
    return data.strip()

def numberify(sent):
    match sent:
        case "negative":
            return 0
        case "positive":
            return 1
        case _:
            return -1

def load_data(fname="Tweets.csv"):
    df = pd.read_csv(fname)
    df = df[df['sentiment'] != "neutral"]
    return df['selected_text'], df['sentiment'].apply(numberify)

# Load and prepare data
data, labels = load_data()
train, test, train_labels, test_labels = sklearn.model_selection.train_test_split(data, labels, test_size=.2, random_state=42)
train, val, train_labels, val_labels = sklearn.model_selection.train_test_split(train, train_labels, test_size=.1, random_state=42)
train_labels = np.array(train_labels)
val_labels = np.array(val_labels)

vectorizer = CountVectorizer(min_df=1, max_features=10000)
vectorizer.fit(train)
train_vectors = vectorizer.transform(train)
val_vectors = vectorizer.transform(val)

# Train the model
c = sklearn.linear_model.LogisticRegression()
c.fit(train_vectors, train_labels)

# Prediction function
def predict_lr(texts):
    return c.predict(vectorizer.transform(texts))

# Initialize explainer
explainer = anchor_text.AnchorText(nlp, ['negative', 'positive'], use_unk_distribution=True)

# Streamlit UI
st.title("Sentiment Analysis and Explainability")

st.subheader("Input a sentence to analyze sentiment")
text_input = st.text_area("Enter your text here:")

if text_input:
    pred_class = explainer.class_names[predict_lr([text_input])[0]]
    st.write(f"Prediction: **{pred_class}**")

    st.subheader("Anchor Explanation")
    exp = explainer.explain_instance(text_input, predict_lr, threshold=0.95)
    st.write(f"**Anchor**: {' AND '.join(exp.names())}")
    st.write(f"**Precision**: {exp.precision():.2f}")

    st.subheader(f"Examples where anchor applies and model predicts **{pred_class}**:")
    same_pred_examples = exp.examples(only_same_prediction=True)
    for ex in same_pred_examples:
        st.write(f"- {ex[0]}")

    alternative_class = explainer.class_names[1 - predict_lr([text_input])[0]]
    st.subheader(f"Examples where anchor applies and model predicts **{alternative_class}**:")
    diff_pred_examples = exp.examples(partial_index=0, only_different_prediction=True)
    for ex in diff_pred_examples:
        st.write(f"- {ex[0]}")
