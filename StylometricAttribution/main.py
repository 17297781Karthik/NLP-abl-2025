import torch
from torch.utils.data import DataLoader, random_split
from data_loader import GutenbergLoader, AuthorDataset, build_vocab, simple_tokenizer
from model import HybridAuthorNet
from train import train_model
from features import StyleExtractor

def main():
    # 1. Data Preparation
    print("--- 1. Fetching Data ---")
    loader = GutenbergLoader()
    texts, labels = loader.get_dataset()
    
    # 2. Preprocessing
    print("--- 2. Building Vocab ---")
    vocab = build_vocab(texts, vocab_size=5000)
    print(f"Vocabulary size: {len(vocab)}")
    
    dataset = AuthorDataset(texts, labels, lambda t, m: simple_tokenizer(t, vocab, m))
    
    # Split Train/Val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # 3. Model Initialization
    print("--- 3. Initializing Hybrid Model ---")
    style_extractor = StyleExtractor()
    style_dim = style_extractor.get_feature_dim()
    
    model = HybridAuthorNet(
        vocab_size=len(vocab) + 2, # +2 for PAD, UNK
        embed_dim=64,
        hidden_dim=128,
        style_dim=style_dim
    )
    
    # 4. Training
    print("--- 4. Starting Training ---")
    train_model(model, train_loader, val_loader, epochs=5)
    
    # 5. Save Artifacts
    print("--- 5. Saving Model & Vocab ---")
    torch.save(model.state_dict(), 'author_model.pth')
    import json
    with open('vocab.json', 'w') as f:
        json.dump(vocab, f)
    print("Saved 'author_model.pth' and 'vocab.json'")
    
    print("\n--- Done! ---")
    print("The 'Avg Gate Alpha' in the logs shows the novelty in action.")
    print("It indicates how much the model relied on Semantic Content vs Stylometric Features.")

if __name__ == "__main__":
    main()
