import torch
import torch.nn as nn
import torch.optim as optim
from features import StyleExtractor

def train_model(model, train_loader, val_loader, epochs=5, learning_rate=0.001):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    style_extractor = StyleExtractor()
    
    print("Starting training...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids']
            labels = batch['label'].unsqueeze(1)
            texts = batch['text']
            
            # Extract style features on the fly
            # In a production system, we would pre-compute this, but for this core project
            # doing it here keeps the code simpler and more readable.
            style_feats = [style_extractor.extract_features(t) for t in texts]
            style_feats = torch.stack(style_feats)
            
            optimizer.zero_grad()
            
            outputs, alpha = model(input_ids, style_feats)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
        train_acc = correct / total
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.4f}")
        
        # Validation
        validate_model(model, val_loader, style_extractor, criterion)

def validate_model(model, val_loader, style_extractor, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    total_alpha = 0
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids']
            labels = batch['label'].unsqueeze(1)
            texts = batch['text']
            
            style_feats = [style_extractor.extract_features(t) for t in texts]
            style_feats = torch.stack(style_feats)
            
            outputs, alpha = model(input_ids, style_feats)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            total_alpha += alpha.mean().item()
            
    val_acc = correct / total
    avg_alpha = total_alpha / len(val_loader)
    print(f"   >>> Val Loss: {total_loss/len(val_loader):.4f} | Val Acc: {val_acc:.4f} | Avg Gate Alpha: {avg_alpha:.4f}")
    print(f"   (Alpha close to 1 means Content focus, close to 0 means Style focus)")
