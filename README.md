# Fine-Tuning Qwen2.5-1.5B with LoRA and QLoRA
This project demonstrates how to fine-tune Qwen2.5-1.5B-Instruct on a conversational cybersecurity dataset using the LoRA and QLoRA parameter-efficient fine-tuning methods.

## Overview
Fine-tuning allows us to adapt a model for specific tasks or domains, such as changing its tone (formal, friendly, etc.) or turning it into a domain expert (e.g., data analyst, meteorologist).
In this project, I fine-tune the Qwen2.5-1.5B-Instruct model using the LoRA and QLoRA parameter-efficient fine-tuning methods,
The training pipeline is built with Unsloth to reduce VRAM usage and speed up training while keeping the code modular and easy to extend.
A cleaned conversational cybersecurity dataset containing 2,394 samples is used for training and evaluation.

## Models and Dataset 
- Qwen 2.5 1.5B
- Cybersecurity dataset
- 2394 cleaned Conversational/Chat samples 

## Key Features
- Parameter-Efficient Fine-Tuning (LoRA, QLoRA)
- Unsloth
- 4-bit quantization with bitsandbytes
- Google Colab (Tesla T4)
- Python

## Training Configuration
- Model: Qwen2.5-1.5B-Instruct

- Dataset size: 2,394

- Train / Validation split: 95% / 5%

- Epochs: 3

- Learning rate: 2e-4

- Batch size: 2

- Gradient accumulation: 4

- Optimizer: AdamW

- Framework: TRL + Unsloth

## How to Run
- First:
```
git clone <your-repository-url>

cd FINE-TUNE-QWEN

python -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt
```

- Then execute the cells sequentially.

- Alternatively, you can open the notebook in Google Colab and run it with a Tesla T4 GPU, which was the environment used during development.

## Project's structure 
```
FINE-TUNE-QWEN/
    │
    ├── notebooks/
    │       └── fine_tune_qwen_qlora.ipynb            
    │
    ├── README.md
    │  
    ├── requirements.txt
    │
    ├── .gitignore  
    │
    ├── cybersecurity_train_clean.jsonl
    │     
    │       
    └── outputs/                  # Saved fine-tuned adapters and checkpoints (created after fine-tuning)
```

## Result
After fine-tuning, the model achieved the following results:
```
Metric                     Base   Fine-tuned      Δ  
-----------------------------------------------------
Cross-entropy Loss       3.7407       0.2726   +92.7%
Perplexity              42.1262       1.3134   +96.9%
```
✅ The fine-tuned model achieved substantially lower validation loss and perplexity than the base model, indicating a much better ability to model conversational cybersecurity data.

## Limitations and Future work
- Adding additional PEFT methods (LoftQ, AdaLoRA, DoRA)
- Fine-tune larger language models
- Train on a larger and more diverse cybersecurity dataset
