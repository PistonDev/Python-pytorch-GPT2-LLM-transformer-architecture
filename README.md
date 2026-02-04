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

Learning reference: Vizuara (YouTube) – clear explanations of Transformer architecture, self-attention, and computational complexity.

---

## Computational Notes & Performance Considerations

This project includes detailed in-code comments describing:

What each major function does

How data flows through the Transformer architecture

Why certain operations are computationally expensive

the comments looks kind of messier tbh

Special attention is given to:

Self-attention complexity

Matrix multiplications inside Transformer blocks

Memory usage during training

Hardware Considerations

Training Transformer models is computationally heavy, especially as:

Sequence length increases

Model depth and embedding size grow

For this reason:

GPU acceleration is strongly recommended

Virtual GPUs (such as cloud-based GPUs) can be used for training and experimentation

CPU-only execution may be slow and is mainly suitable for testing or learning purposes

> The code is written to prioritize clarity and architectural understanding over performance optimization.

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
```bash
run main.py
