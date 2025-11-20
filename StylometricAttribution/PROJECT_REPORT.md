# Project Report: Hybrid Stylometric-Neural Authorship Attribution

## 1. Abstract (Problem Statement)
The core problem addressed in this project is **Authorship Attribution**: the task of identifying the author of a given text segment based on its linguistic properties. While humans often recognize an author's "voice" intuitively, quantifying this computationally remains a challenge.

The specific challenge addressed here is the **"Short-Text Attribution Problem"**. In academic and forensic contexts, we often do not have entire books to analyze, but rather short excerpts, essays, or anonymous posts. Traditional methods often struggle with short texts because they lack sufficient statistical mass, while modern deep learning methods can overfit to semantic content (topic) rather than true stylistic identity.

This project aims to build a robust classifier that can distinguish between authors (specifically Jane Austen and Arthur Conan Doyle) by leveraging both **what** they say (content) and **how** they say it (style).

## 2. Existing Solutions
Historically, solutions to this problem have fallen into two distinct camps:

### A. Traditional Stylometry (The "Linguist's Approach")
This approach relies on hand-crafted statistical features.
*   **Method**: Extracting features like Type-Token Ratio (TTR), average sentence length, function word frequency (e.g., usage of "the", "and", "of"), and punctuation density.
*   **Pros**: Highly interpretable and robust to topic changes.
*   **Cons**: Fails to capture complex semantic dependencies or sequential patterns in the text. It ignores the "meaning" of the text entirely.

### B. Deep Neural Networks (The "Engineer's Approach")
This approach treats text as a sequence of vectors.
*   **Method**: Using Recurrent Neural Networks (RNNs, LSTMs) or Transformers (BERT, GPT) to learn representations directly from raw text.
*   **Pros**: State-of-the-art performance on many benchmarks; captures deep semantic meaning.
*   **Cons**: "Black box" nature. These models often learn to classify based on *topic* rather than *style*. For example, if one author writes about "tea" and another about "detectives", the model learns to detect the word "detective" rather than the author's writing style. This is a failure of attribution logic.

## 3. Proposed Solution (Methodology)
To solve the limitations of both approaches, we implemented a **Dual-Stream Hybrid Architecture**. This system does not rely on a single view of the data but processes it through two parallel pathways that are eventually fused.

### The Architecture
1.  **Stream A: The Semantic Branch (Content)**
    *   **Input**: Raw token sequences (word IDs).
    *   **Model**: An **LSTM (Long Short-Term Memory)** network.
    *   **Role**: This branch reads the text sequentially to understand the context, narrative flow, and semantic meaning. It captures *what* is being said.

2.  **Stream B: The Stylometric Branch (Style)**
    *   **Input**: A vector of explicit linguistic features (Sentence Length, TTR, Punctuation Density, Function Word Frequencies).
    *   **Model**: A **Multi-Layer Perceptron (MLP)**.
    *   **Role**: This branch looks at the statistical "fingerprint" of the text. It captures *how* it is being said, ignoring the actual meaning of the words.

3.  **The Fusion Layer**
    *   The two streams produce two distinct latent representations: $h_{content}$ and $h_{style}$.
    *   These are fed into a custom **Gated Fusion Mechanism** (detailed below) to produce a final classification.

## 4. Novelty
The primary novelty of this project, satisfying the academic requirement for innovation beyond standard textbook implementations, is the **Learnable Gated Fusion Mechanism**.

In standard multi-modal deep learning, features are often simply concatenated ($h_{final} = [h_{content}, h_{style}]$). This assumes both inputs are always equally important.

Our approach introduces a dynamic gate $\alpha$ (alpha):

$$ \alpha = \sigma(W \cdot [h_{content}, h_{style}] + b) $$
$$ h_{final} = \alpha \cdot h_{content} + (1 - \alpha) \cdot h_{style} $$

### Why is this Novel?
1.  **Dynamic Adaptation**: The model learns to self-regulate. For some sentences (e.g., "Elementary, my dear Watson"), the semantic content is a dead giveaway. The gate $\alpha$ will shift towards 1.0 (Content). For others (e.g., a complex, winding sentence structure typical of Austen), the semantic content might be generic, but the *structure* is unique. The gate $\alpha$ will shift towards 0.0 (Style).
2.  **Interpretability**: Unlike a standard black-box neural net, our model outputs the $\alpha$ value during inference. This allows us to peer inside the decision-making process and understand *why* a decision was made (e.g., "The model relied 80% on punctuation patterns for this prediction").

This architecture bridges the gap between classical linguistics and modern deep learning, offering a more scientifically grounded approach to authorship attribution.
