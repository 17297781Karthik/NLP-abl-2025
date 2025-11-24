import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from data_loader import GutenbergLoader, AuthorDataset, build_vocab, simple_tokenizer
from train import train_model
from features import StyleExtractor
from model import HybridAuthorNet

# --- Baseline Models ---

class StyleBaseline(nn.Module):
    """
    Baseline: Traditional Stylometry (Style).
    Ignores semantic content (LSTM).
    """
    def __init__(self, style_dim, hidden_dim):
        super().__init__()
        self.style_mlp = nn.Sequential(
            nn.Linear(style_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input_ids, style_features):
        logits = self.style_mlp(style_features)
        return self.sigmoid(logits), torch.tensor(0.0)


# --- Experiment Runner ---

def run_experiment():
    print("--- Setting up Data ---")
    loader = GutenbergLoader()
    texts, labels = loader.get_dataset()
    vocab = build_vocab(texts, vocab_size=5000)
    dataset = AuthorDataset(texts, labels, lambda t, m: simple_tokenizer(t, vocab, m))
    
    subset_size = len(dataset)
    dataset, _ = random_split(dataset, [subset_size, len(dataset) - subset_size])
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    style_extractor = StyleExtractor()
    style_dim = style_extractor.get_feature_dim()
    
    # --- Define Models ---
    models = {
        'Traditional Stylometry': StyleBaseline(style_dim, hidden_dim=64),
        'Proposed Hybrid Model': HybridAuthorNet(len(vocab)+2, embed_dim=64, hidden_dim=64, style_dim=style_dim)
    }
    
    results = {}
    
    # --- Train Loop ---
    for name, model in models.items():
        print(f"\nTraining {name}...")
        history = train_model(model, train_loader, val_loader, epochs=5)
        results[name] = history['val_acc']
        
    # --- Plotting ---
    print("\nPlotting results...")
    plt.figure(figsize=(10, 6))
    
    for name, accs in results.items():
        plt.plot(range(1, 6), accs, marker='o', label=name)
        
    plt.title('Comparison of Authorship Attribution Models')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('experimental_results.png')
    print("Saved plot to 'experimental_results.png'")

if __name__ == "__main__":
    run_experiment()
