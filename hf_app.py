import gradio as gr
from frontend.gradio_app import demo  # or your main Gradio object name

# Hugging Face runs: python app.py
# So we launch the app here.
if __name__ == "__main__":
    demo.launch()
