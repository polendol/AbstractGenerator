from unsloth import FastLanguageModel
import torch
import warnings
warnings.filterwarnings("ignore", category=FutureWarning) #warnigns are not displayed anymore

# load tuned model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit-lora_abstracts",
    max_seq_length = 1024,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model) #updates and makes the model more professional

# Prompts
test_title = "Cognitive load and reading comprehension in pre-school children"
prompt = f"Expand the following research title into a sophisticated academic abstract.Title: {test_title}"
messages = [
    {"role": "system", "content": "You are a professional academic researcher."},
    {"role": "user", "content": prompt}, #no answer here, this is not training
]

# generate
inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda") #add generation prompt tells you stopped typing and you are waiting for the answer

print("\n--- GENERATED ABSTRACT ---\n")
outputs = model.generate(
    input_ids=inputs,
    max_new_tokens=512, #longer abstracts
    temperature=0.6, #1 gets quite crazy
    repetition_penalty=1.2 #between 1.1. and 1.2 to keep it calm, higher will cause bad grammar and weird synonyms
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True).split("assistant")[-1].strip())