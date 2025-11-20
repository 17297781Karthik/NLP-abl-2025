# Stylometric Authorship Attribution with Gated Fusion

## Overview
This project implements a **Hybrid Neural Authorship Verification** system. It distinguishes itself from standard text classification tutorials by combining two distinct modalities of text analysis into a single architecture:

1.  **Semantic Content**: Analyzed via a Recurrent Neural Network (LSTM) operating on word embeddings.
2.  **Stylometric Signature**: Analyzed via a Multi-Layer Perceptron (MLP) operating on explicit linguistic features (Type-Token Ratio, punctuation density, sentence length statistics, etc.).

## The Novelty: Gated Fusion Mechanism
Standard ensemble models often just concatenate features. This project introduces a learnable **Gated Fusion Layer** ($\alpha$) that dynamically weighs the importance of "Style" vs "Content" for each input sample.

$$ \text{Prediction} = \sigma( \alpha \cdot \text{ContentLogits} + (1 - \alpha) \cdot \text{StyleLogits} ) $$

This allows the model to decide whether a specific text segment is better attributed by *what* was said (Content) or *how* it was said (Style).

## Project Structure
- `data_loader.py`: Fetches public domain books from Project Gutenberg and processes them into chunks.
- `features.py`: Manual implementation of stylometric feature extractors.
- `model.py`: PyTorch implementation of the `HybridAuthorNet` and `GatedFusion`.
- `train.py`: Custom training loop (no `Trainer` abstractions).
- `main.py`: Entry point to run the experiment.

## Requirements
- Python 3.8+
- PyTorch
- NumPy
- Requests
