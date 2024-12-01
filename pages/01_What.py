import streamlit as st

# page 01
st.title("What Are Anchors")

st.write(f"Anchors are decision rules or conditions that are sufficient to ensure that a model's prediction does not change for similar inputs. These rules identify the key features of the input that most strongly influence the model’s prediction. They are designed to:")
st.write(f"     - Be interpretable to humans (e.g., If feature X is greater than value Y, the model will predict Z).")
st.write(f"     - Work locally, meaning they explain a single prediction, not the overall behavior of the model.")
