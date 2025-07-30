import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import gradio as gr
import torch
import gradio as gr
from transformers import T5Tokenizer, T5ForConditionalGeneration


# Load tokenizer and model
model_name = "t5-base"  # or your LoRA checkpoint directory
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define T5 QA function
def t5_answer_question(question, context):
    input_text = f"question: {question} context: {context}"
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512  # To avoid long context issues
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    output_ids = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=64
    )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Gradio interface
gr.Interface(
    fn=t5_answer_question,
    inputs=[
        gr.Textbox(lines=2, label="Question", placeholder="Ask a question..."),
        gr.Textbox(lines=8, label="Context", placeholder="Paste your car info here...")
    ],
    outputs=gr.Textbox(label="Answer"),
    title="LoRA QA: Cars24 (T5)",
    description="Ask questions about cars based on context (fine-tuned T5 with LoRA)."
).launch()

