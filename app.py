from transformers import pipeline
import gradio as gr


model_name = "cardiffnlp/twitter-roberta-base-sentiment"
classifier = pipeline("sentiment-analysis", model=model_name)

label_map = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive"
}
def analyze_sentiment(text):
    result = classifier(text)[0]
    label = label_map[result['label']]
    score = result['score']
    return f"{label} ({score:.2f})"


iface = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(lines=2, placeholder="Enter text here..."),
    outputs="text",
    title="Student Reviews Sentiment Analyzer",
    description="Classifies text into positive, negative, or neutral sentiment."
)


if __name__ == "__main__":
    iface.launch(share=True) 
