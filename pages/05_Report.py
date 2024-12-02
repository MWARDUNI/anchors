import streamlit as st

# Markdown content as a string
markdown_report = """
# Report: Anchors for Explainable AI in Sentiment Analysis

## Introduction
This project leverages the Anchors Explainable AI (XAI) methodology to provide local explanations for sentiment classification predictions on Twitter data. The application is implemented using Python and Streamlit, combining logistic regression with interpretable explanations provided by the `anchor` library. The system is designed to identify and present human-interpretable rules that justify the model's sentiment predictions for specific input texts.

---

## Overview of the Model
The sentiment analysis system processes tweets classified into "positive" and "negative" sentiments. It employs the following key steps:

### Data Preparation:
- Tweets are preprocessed to exclude neutral sentiments.
- Labels are converted to numeric values (`1` for positive, `0` for negative).
- The dataset is split into training, validation, and testing subsets.

### Feature Engineering:
- A `CountVectorizer` transforms textual data into a bag-of-words representation.
- Features are limited to a maximum of 10,000 to optimize model performance.

### Model Training:
- A logistic regression model is trained on the vectorized training data to predict sentiment.

### Prediction Function:
- The `predict_lr` function uses the trained model to infer sentiment labels for new input texts.

---

## Anchor-based Explainability
The Anchors technique, implemented via the `anchor_text.AnchorText` class, provides localized, human-readable explanations. These explanations take the form of "anchors," or sets of textual patterns (words or phrases) that are sufficient to drive the model toward a specific prediction.

### Anchor Mechanism
- **Precision**: The precision of an anchor represents how often the model predicts the same sentiment when the anchor applies to different inputs.
- **Examples**: The explainer also generates instances where the anchor successfully influences the prediction and instances where the prediction changes, enabling comparative analysis.

---

## Streamlit Application
The Streamlit interface enables users to input text, obtain sentiment predictions, and view anchor-based explanations. The key components of the UI include:

### User Input:
- A text box allows users to enter sentences for sentiment analysis.

### Prediction:
- The logistic regression model classifies the input text into positive or negative sentiment.

### Anchor Explanation:
- A detailed breakdown of the anchor is provided, including:
  - **Anchor rules**: Words or phrases that form the basis of the prediction.
  - **Precision**: The reliability of the anchor for the prediction.

### Example Insights:
- Examples where the anchor supports the current prediction are displayed.
- Counterexamples are also provided, illustrating cases where the anchor applies but leads to a different prediction.

---

## Example Output
For an input like *"This product is amazing!"*, the application might output:

- **Prediction**: Positive
- **Anchor**: ["amazing"]
- **Precision**: 0.98
- **Examples (Positive Prediction)**:
  - "Amazing quality and performance!"
  - "The design is amazing."
- **Examples (Negative Prediction)**:
  - None (indicating high precision).

---

## Applications and Benefits
### Transparency:
- Anchors provide a clear, interpretable justification for model decisions.

### Trust:
- Users gain confidence in the model through transparent and robust explanations.

### Model Debugging:
- Developers can use anchors and associated examples to identify model weaknesses and biases.

### Educational Value:
- Anchors serve as a teaching tool, helping users understand the relationship between linguistic patterns and sentiment.

---

## Challenges and Limitations
### Threshold Dependency:
- The threshold for anchor precision impacts explanation quality and may require tuning for specific applications.

### Text Complexity:
- Highly nuanced or context-dependent phrases may reduce anchor clarity.

### Scalability:
- Generating anchors and examples for large-scale text can be computationally expensive.

---

## Future Work
- **Enhanced Linguistic Features**: Incorporating advanced preprocessing (e.g., semantic embeddings) could refine anchor generation.
- **Multi-class Support**: Extending the framework to handle more granular sentiment categories, including "neutral."
- **Integration with Larger Models**: Applying anchors to deep learning models like transformers (e.g., BERT).

---

## Conclusion
Anchors represent a powerful and intuitive approach to XAI for sentiment analysis, enabling human-centric explanations of machine predictions. This project highlights the synergy between effective machine learning models and explainability, paving the way for greater adoption and understanding of AI in sentiment analysis and beyond.
"""

# Display the report in Streamlit
st.markdown(markdown_report)
