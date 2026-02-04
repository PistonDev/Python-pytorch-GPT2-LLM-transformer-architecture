# GPT-2 Transformer Architecture (From Scratch)

This repository contains a **GPT-2–style Transformer model implementation built from scratch in Python**.

The project focuses on understanding and implementing the **core architecture of GPT-2**, including:
- Tokenization
- Transformer blocks
- Self-attention
- Model training and text generation

⚠️ This is **not a classification model**.

---

## Project Structure
├── main.py
├── gpt_download3.py
├── requirements.txt
├── verdict.txt
└── README.md

---

## Dataset Handling

You can use **any plain text dataset** for training the GPT-2 model.

### Important Notes
- `verdict.txt` is included **only as a sample dataset for training**
- You are free to replace it with any other text data
  
### Dataset Path

The dataset path is **defined inside the code** and points to the **same working directory**.

If you change the dataset location:
- Update the path inside the Python files
- Ensure the folder structure matches what the code expects

> This design allows flexibility while keeping the repository lightweight.

---

## Model Description

- Architecture inspired by **GPT-2**
- Decoder-only Transformer
- Uses self-attention and positional embeddings
- Designed for **language modeling and text generation**
- Supports training on custom text corpora

---

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd <repo-name>
```

2. Install dependencies:
  pip install -r requirements.txt

## Running the Model
- run main.py
