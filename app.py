from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import streamlit as st


model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
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
        result = classifier(user_input)[0]
        label = label_map[result['label']]
        score = result['score']
        st.write(f"**Sentiment:** {label}")
        st.write(f"**Confidence** {score:.2f}")
    else:
        st.write("Please enter some text to analyze.")