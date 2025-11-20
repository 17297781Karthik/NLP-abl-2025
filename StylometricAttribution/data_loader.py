import requests
import re
import random
import torch
from torch.utils.data import Dataset

class GutenbergLoader:
    """
    Helper class to download and clean books from Project Gutenberg.
    """
    def __init__(self):
        # Example: Jane Austen (Pride and Prejudice) vs Arthur Conan Doyle (Sherlock Holmes)
        self.books = {
            'austen': 'https://www.gutenberg.org/files/1342/1342-0.txt',
            'doyle': 'https://www.gutenberg.org/files/1661/1661-0.txt'
        }
    
    def fetch_text(self, url):
        print(f"Downloading from {url}...")
        response = requests.get(url)
        response.encoding = 'utf-8'
        text = response.text
        
        # Remove Gutenberg header/footer (heuristic)
        start_idx = text.find("*** START OF THE PROJECT GUTENBERG EBOOK")
        end_idx = text.find("*** END OF THE PROJECT GUTENBERG EBOOK")
        
        if start_idx != -1 and end_idx != -1:
            text = text[start_idx:end_idx]
        
        return text

    def clean_text(self, text):
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def create_chunks(self, text, chunk_size=100):
        """Splits text into chunks of roughly `chunk_size` words."""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i+chunk_size])
            if len(chunk.split()) > chunk_size * 0.8: # Filter very short chunks
                chunks.append(chunk)
        return chunks

    def get_dataset(self):
        all_samples = []
        labels = [] # 0 for Austen, 1 for Doyle
        
        for label_idx, (author, url) in enumerate(self.books.items()):
            raw_text = self.fetch_text(url)
            clean_text = self.clean_text(raw_text)
            chunks = self.create_chunks(clean_text)
            
            print(f"Processed {len(chunks)} chunks for {author}")
            
            all_samples.extend(chunks)
            labels.extend([label_idx] * len(chunks))
            
        return all_samples, labels

class AuthorDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=200):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenization (Simple whitespace + basic vocab mapping for this core project)
        # In a real project we might use a proper tokenizer, but here we keep it core.
        # We will assume 'tokenizer' is a simple dict or function passed in.
        
        token_ids = self.tokenizer(text, self.max_len)
        
        return {
            'text': text, # Raw text needed for stylometric features
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.float)
        }

def simple_tokenizer(text, vocab, max_len):
    """Core level tokenizer without heavy libraries"""
    words = text.lower().split()
    ids = [vocab.get(w, vocab.get('<UNK>', 0)) for w in words]
    
    if len(ids) < max_len:
        ids = ids + [0] * (max_len - len(ids)) # Pad
    else:
        ids = ids[:max_len] # Truncate
        
    return ids

def build_vocab(texts, vocab_size=5000):
    word_counts = {}
    for text in texts:
        for word in text.lower().split():
            word_counts[word] = word_counts.get(word, 0) + 1
            
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    vocab = {'<PAD>': 0, '<UNK>': 1}
    
    for i, (word, _) in enumerate(sorted_words[:vocab_size-2]):
        vocab[word] = i + 2
        
    return vocab
