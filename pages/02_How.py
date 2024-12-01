import streamlit as st

# page 2
st.subheader(f"How Do Anchors Work?")

st.write("1. **Input Instance**: Begin with a specific instance for which you want an explanation.")

st.write("2. **Generate Candidate Rules**:")
st.write(f"     - Create candidate rules (anchors) using a combination of feature conditions.")
st.write(f"     - e.g. in our sentiment analysis model, anchor might be `The word **great** is in the text.`")

st.write(f"3. **Perturb the Input**:")
st.write(f"     -  Create similar inputs by perturbing the instance (e.g., by adding or removing features).")
st.write(f"     - This is done by slightly changing the text or modifying the numerical features.")

st.write(f"4. **Evaluate Rule Percision**:")
st.write(f"     - Check how often the model`s prediction stays the same when the rule applies to the perturbed instance.")
st.write(f"     - Rules that have **high precision**, > 95% are considered **anchors**, as they reliably explain the prediction.")