import gradio as gr
from unsloth import FastLanguageModel
import torch
import warnings
import os
warnings.filterwarnings("ignore", category=FutureWarning) #warnigns are not displayed anymore

# load tuned model
MODEL_NAME = os.environ.get("MODEL_PATH", "unsloth/Llama-3.2-1B-Instruct-bnb-4bit-lora_private_abstracts")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = 1024,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model) #optimizes the internal kernels for 2x faster inference


def generate_abstract(title, keywords):
    prompt = (
        f"Expand the following research title into a sophisticated academic abstract.\n\n"
        f"Title: {title}\n\n"
        f"Mandatory Keywords to include: {keywords}"
    )

    messages = [
        {"role": "system",
         "content": "You are a professional academic researcher. You write abstracts that strictly incorporate all provided keywords."},
        {"role": "user", "content": prompt},
    ]

    inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(
        "cuda")

    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=512,
        temperature=0.6,
        repetition_penalty=1.2
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True).split("assistant")[-1].strip()

# Gradio UI
with gr.Blocks(title="Academic Abstract Generator") as demo:
    gr.Markdown("# AI Research Abstract Generator")
    gr.Markdown("**Built with Llama**")
    with gr.Row():
        with gr.Column():
            title_input = gr.Textbox(label="Research Title", placeholder="Enter your paper title...")
            keywords_input = gr.Textbox(label="Mandatory Keywords", placeholder="Keyword 1, Keyword 2...")
            submit_btn = gr.Button("Generate Abstract", variant="primary")
        with gr.Column():
            output_text = gr.TextArea(label="Generated Abstract", interactive=False, lines=12)

    submit_btn.click(fn=generate_abstract, inputs=[title_input, keywords_input], outputs=output_text)

    gr.Markdown(
        "Fine-tuned from Llama 3.2 1B Instruct. "
        "Llama 3.2 is licensed under the "
        "[Llama 3.2 Community License](https://www.llama.com/llama3_2/license/), "
        "Copyright © Meta Platforms, Inc. All Rights Reserved. "
        "Outputs are generated text and may contain invented claims; "
        "they are not real research."
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)