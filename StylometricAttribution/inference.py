import torch
import json
from model import HybridAuthorNet
from features import StyleExtractor
from data_loader import simple_tokenizer

def load_artifacts():
    print("Loading vocab and model...")
    with open('vocab.json', 'r') as f:
        vocab = json.load(f)
        
    style_extractor = StyleExtractor()
    style_dim = style_extractor.get_feature_dim()
    
    model = HybridAuthorNet(
        vocab_size=len(vocab) + 2,
        embed_dim=64,
        hidden_dim=128,
        style_dim=style_dim
    )
    
    model.load_state_dict(torch.load('author_model.pth'))
    model.eval()
    
    return model, vocab, style_extractor

def predict(text, model, vocab, style_extractor):
    # 1. Prepare Content Input
    token_ids = simple_tokenizer(text, vocab, max_len=200)
    input_ids = torch.tensor([token_ids], dtype=torch.long)
    
    # 2. Prepare Style Input
    style_feats = style_extractor.extract_features(text)
    style_feats = style_feats.unsqueeze(0) # Batch dim
    
    # 3. Inference
    with torch.no_grad():
        prob, alpha = model(input_ids, style_feats)
        
    return prob.item(), alpha.item()

def main():
    try:
        model, vocab, style_extractor = load_artifacts()
    except FileNotFoundError:
        print("Error: Model or Vocab not found. Please run 'main.py' first to train the model.")
        return

    print("\n--- Stylometric Authorship Attribution Inference ---")
    print("Type a sentence to check if it sounds more like Jane Austen or Conan Doyle.")
    print("Enter 'q' to quit.\n")
    
    while True:
        text = input("Enter text: ")
        if text.lower() == 'q':
            break
            
        prob, alpha = predict(text, model, vocab, style_extractor)
        
        # 0 = Austen, 1 = Doyle
        author = "Jane Austen" if prob < 0.5 else "Conan Doyle"
        confidence = (1 - prob) if prob < 0.5 else prob
        
        print(f"\nPrediction: {author} ({confidence:.2%} confidence)")
        print(f"Gate Alpha: {alpha:.4f} (Weight given to Content vs Style)")
        
        if alpha > 0.6:
            print("Reasoning: The model relied mostly on WORDS (Content).")
        elif alpha < 0.4:
            print("Reasoning: The model relied mostly on PUNCTUATION/STATS (Style).")
        else:
            print("Reasoning: The model used a balanced mix of both.")
        print("-" * 50)

if __name__ == "__main__":
    main()
