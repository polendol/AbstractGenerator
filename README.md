# AI Abstract Generator

## Description
AI Abstract Generator aims to facilitate abstract generation for researchers who have to produce several variations of 
the same abstract (e.g. the same study presented in different events). 

Given the specificity of academic abstracts, 1B parameter, 4-bit quantized version of Meta's Llama 3.2 Instruct model 
was fine-tuned using LoRa PEFT with gfissore/arxiv-abstracts-2021 abstracts dataset which maps academic papers titles to
its abstracts. Unsloth was used to fine-tune the model on a local GPU. Training loss reached 2.254, consistent with 
fine-tuning a small 4-bit quantized model on a diverse academic dataset. Output quality was validated qualitatively. 

In order to personalize the model, the fine-tuned model was once again trained using the same methodology, but this time
with a local .jsonl database of researcher's previously submitted titles, abstract, and corresponding keywords. Keywords
are an innovation in the PEFT as compared to the original model fine-tuning and are necessary for the inclusion of this 
element in the final program. Therefore, the final model is specifically fine-tuned to create personalized abstracts 
imitating the user's previous work. This model is available through AbstractGenerator.py. 

AbstractGenerator.py uses gradio interface where user can introduce the title of the paper, as well as the keywords that
must be included in the abstract. It is designed to work uniquely with abstracts in English.

*Test_FT_model.py file is a test file for the first fine-tuning of the model. It is not to be used as abstract generator. 


## Docker

## Requirements
1. Docker & Docker Compose: Installed and running (Docker Desktop for Windows/macOS or Docker Engine on Linux).
2. NVIDIA GPU: An NVIDIA graphics card with up-to-date drivers.
3. NVIDIA Container Toolkit: Required on Linux/WSL2 so Docker can pass GPU access to the container (--gpus all).
4. Hardware: Minimum 16 GB RAM (32 GB recommended), 6 GB+ VRAM on your NVIDIA GPU, at least 15 GB free disk space for the CUDA base image, PyTorch, and model checkpoints.

## How it works

1. Multi-Stage Build:
Builder (nvidia/cuda:12.6.3-devel-ubuntu24.04): Compiles dependencies and sets up the Python virtual environment with build tools (gcc, build-essential, git).
Runtime (nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04): Copies only the compiled virtual environment, leaving behind gigabytes of unnecessary C++ compilers to keep the final image as lean as possible.
2. Security: Runs as a non-root user (appuser with UID 1001).
3. Hugging Face Cache: Uses /home/appuser/.cache/huggingface (HF_HOME) to store downloaded models persistently across runs when mounted.
4. Gradio Web Interface: Listens on port 7860 bound to 0.0.0.
5. Step-by-step: to build the container, run `docker build -t abstract-generator .` to run the container: `docker run --gpus all -p 7860:7860 -v hf_cache:/home/appuser/.cache/huggingface abstract-generator`. Then open `http://localhost:7860`
6. See How it works section in Local section below for specifics on how the program and output details

## Local
## Requirements
- Python 3.8+
- gradio
- torch
- unsloth

## Models
- "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"
- Model: `unsloth/Llama-3.2-1B-Instruct-bnb-4bit-lora_private_abstracts`

## Installation
pip install -r requirements.txt

## Hardware
CUDA GPU is required

## Usage
python AbstractGenerator.py


Enter the academic paper title
Enter the keywords that must be included in the abstract

## Output
Generates an academic abstract related to the introduced titles which includes all keywords. 

## How It Works
1. Fine-tuned model is loaded and optimized for a faster inference
2. Takes a paper title and keywords as input
3. Builds a prompt asking the AI to write an academic abstract
4. Sends it to the model as a conversation (system role + user role)
5. The model generates the abstract and returns it as text
6. Creates a simple UI with gradio
7. Launches the app locally at http://127.0.0.1:7860

## File Structure

- `outputs/` — generated during Unsloth fine-tuning (checkpoints, logs)
- `unsloth/`, `unsloth_compiled_cache/` — Unsloth internal cache directories
- `AbstractGenerator.py` — main program, run this to launch the app
- `Training.py` — first fine-tuning stage on gfissore/arxiv-abstracts-2021 dataset
- `TrainingMyDataset.py` — second fine-tuning stage on private abstracts
- `Test_FT_model.py` — test script for validating the first fine-tuned model (not for end use)
- `abstracts.jsonl` — private abstract database used for personalization
- `requirements.txt` — Python dependencies
- `README.md` — this file
- `Dockerfile`
- `.dockerignore`


## Notes
- the new abstract size is limited to 512 tokens to imitate the standard length of an academic abstract
- the temperature si set to 0.6 to allow for some creativity and avoid repetition, as academic texts tend to be highly formulaic
- repetition penalty applied between 1.1. and 1.2 to avoid bad grammar and strange synonyms