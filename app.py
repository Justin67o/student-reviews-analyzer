from transformers import pipeline
import streamlit as st

classifier = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

st.title("Student Reviews Sentiment Analyzer")
st.write("Enter a student review and see if it's positive, neutral, or negative")

user_input = st.text_area("Type review here")

label_map = {
    "LABEL_0": "Negative",
    "Label_1": "Neutral",
    "Label_2": "Positive"
}

if st.button("Analyze Sentiment"):
    if user_input.strip() != "":
        print(user_input)
        result = classifier(user_input)[0]
        label = label_map[result['label']]
        score = result['score']
        st.write(f"**Sentiment:** {label}")
        st.write(f"**Confidence** {score:.2f}")
    else:
        st.write("Please enter some text to analyze.")