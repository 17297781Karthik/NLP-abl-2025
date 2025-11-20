import torch
import torch.nn as nn

class GatedFusion(nn.Module):
    """
    Novelty Component:
    Dynamically weighs the contribution of Style vs Content features.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.gate_fc = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, content_emb, style_emb):
        # Concatenate both representations to decide the weight
        combined = torch.cat((content_emb, style_emb), dim=1)
        
        # Compute alpha (0 to 1)
        alpha = self.sigmoid(self.gate_fc(combined))
        
        # Weighted sum
        fused = alpha * content_emb + (1 - alpha) * style_emb
        return fused, alpha

class HybridAuthorNet(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, style_dim):
        super().__init__()
        
        # --- Content Branch (LSTM) ---
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        
        # --- Style Branch (MLP) ---
        self.style_mlp = nn.Sequential(
            nn.Linear(style_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # --- Fusion ---
        self.fusion = GatedFusion(hidden_dim)
        
        # --- Classifier ---
        self.classifier = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input_ids, style_features):
        # 1. Process Content
        embedded = self.embedding(input_ids)
        # LSTM output: (batch, seq, hidden), (h_n, c_n)
        _, (h_n, _) = self.lstm(embedded)
        content_rep = h_n[-1] # Take the last hidden state
        
        # 2. Process Style
        style_rep = self.style_mlp(style_features)
        
        # 3. Fuse
        fused_rep, alpha = self.fusion(content_rep, style_rep)
        
        # 4. Classify
        logits = self.classifier(fused_rep)
        probs = self.sigmoid(logits)
        
        return probs, alpha
