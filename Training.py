from unsloth import FastLanguageModel, get_chat_template
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, AutoTokenizer

# Tokenizer and model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    max_seq_length = 1024,
    load_in_4bit = True,
)
#ensure the tokenizer is correct
tokenizer = get_chat_template(tokenizer,
                              chat_template = "llama-3")

# LoRA adapters layer
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, #ranks
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0)

# load the peS2o dataset, a subset only
dataset = load_dataset("gfissore/arxiv-abstracts-2021", split="train", streaming=True) #this is mandatory to upload just a subset, not the whole dataset
#print(next(iter(dataset)).keys()) useful to make sure what are the names of the columns you need

# format the dataset for the needs of the training
def formatting_prompts_func(examples):
    instructions = "Expand the following research title into a sophisticated academic abstract."
    titles = examples["title"]
    abstracts = examples["abstract"]
    texts = []

    for title, abstract in zip(titles, abstracts):
        messages = [
            {"role": "system", "content": "You are a professional academic researcher."},
            {"role": "user", "content": f"{instructions}\n\nTitle: {title.strip()}"},
            {"role": "assistant", "content": abstract.strip()}
        ]
        # This converts the roles into the actual Llama 3 string format
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False) #only text for now, no tokenization
        texts.append(text)
    return {"text": texts}

# take what you need and map the format to the first 5,000 samples
train_subset = dataset.take(5000)
train_dataset = train_subset.map(formatting_prompts_func, batched=True)

# Trainer
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    dataset_text_field = "text",
    max_seq_length = 1024,
    args = TrainingArguments(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 8,
        warmup_steps = 10,
        max_steps = 300, # Set higher (e.g., 500) for better results
        learning_rate = 2e-4,
        fp16 = False,
        bf16 = True,
        logging_steps = 1,
        optim = "paged_adamw_8bit", #optimized for laptop GPU
        weight_decay = 0.01,
        output_dir = "outputs",
        gradient_checkpointing = True,))

# Training
trainer.train()
#{'train_runtime': '877.8', 'train_samples_per_second': '2.734', 'train_steps_per_second': '0.342', 'train_loss': '2.254', 'epoch': '1'}
#Loss 2.0 – 2.5: This is the "Learning Zone." The model is starting to understand the style of ArXiv abstracts without just memorizing them word-for-word.
#Loss > 4.0: The model is confused or the data is messy.
#Loss < 1.0: For a 1B model on complex text, this usually means it's starting to memorize specific papers rather than learning "how to write.

# New model and tokenizer saved
model.save_pretrained("llama-3.2-1b-abstracts-lora")
tokenizer.save_pretrained("llama-3.2-1b-abstracts-lora")