import gradio as gr
from text_summarizer import TextSummarizer

# Initialize summarizer globally so model is loaded once
summarizer = TextSummarizer()

def summarize_text(text, method, summary_ratio, max_length, min_length):
    if not text or not text.strip():
        return "Please enter some text.", "Please enter some text."
        
    method_val = method.lower().split()[0]
    if method_val == "both":
        method_val = "both"
    elif method_val == "extractive":
        method_val = "extractive"
    else:
        method_val = "abstractive"
        
    results = summarizer.summarize(
        text,
        method=method_val,
        summary_ratio=summary_ratio,
        max_length=int(max_length),
        min_length=int(min_length)
    )
    
    ext_sum = results.get("extractive", "Not selected.")
    abs_sum = results.get("abstractive", "Not selected.")
    return ext_sum, abs_sum

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎯 AI Text Summarizer")
    gr.Markdown("A comprehensive text summarization tool implementing both **extractive** and **abstractive** techniques.")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_text = gr.Textbox(lines=12, label="Input Text", placeholder="Paste your text here...")
            method = gr.Radio(["Both", "Extractive Only", "Abstractive Only"], value="Both", label="Summarization Method")
            
            with gr.Accordion("Advanced Settings", open=False):
                ext_ratio = gr.Slider(0.1, 0.9, value=0.3, step=0.1, label="Extractive Summary Ratio")
                abs_max = gr.Slider(50, 500, value=150, step=10, label="Abstractive Max Length")
                abs_min = gr.Slider(10, 200, value=50, step=10, label="Abstractive Min Length")
                
            submit_btn = gr.Button("Summarize", variant="primary")
            
        with gr.Column(scale=1):
            ext_output = gr.Textbox(lines=7, label="Extractive Summary")
            abs_output = gr.Textbox(lines=7, label="Abstractive Summary")
            
    submit_btn.click(
        fn=summarize_text,
        inputs=[input_text, method, ext_ratio, abs_max, abs_min],
        outputs=[ext_output, abs_output]
    )

if __name__ == "__main__":
    demo.launch()
