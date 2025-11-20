import re
import torch
import numpy as np

class StyleExtractor:
    """
    Extracts explicit linguistic features from text.
    This represents the 'Style' branch of our hybrid model.
    """
    def __init__(self):
        self.punctuation_marks = ['.', ',', ';', ':', '!', '?', '-', '"', "'"]
        self.function_words = {'the', 'and', 'of', 'to', 'a', 'in', 'that', 'is', 'was', 'he', 'for', 'it', 'with', 'as', 'his', 'on', 'be', 'at', 'by', 'i'}

    def extract_features(self, text):
        features = []
        
        # 1. Average Sentence Length
        sentences = re.split(r'[.!?]+', text)
        sentences = [s for s in sentences if len(s.strip()) > 0]
        words = text.split()
        num_words = len(words)
        num_sentences = len(sentences) if len(sentences) > 0 else 1
        avg_sent_len = num_words / num_sentences
        features.append(avg_sent_len)
        
        # 2. Type-Token Ratio (Vocabulary Richness)
        unique_words = set(w.lower() for w in words)
        ttr = len(unique_words) / num_words if num_words > 0 else 0
        features.append(ttr)
        
        # 3. Punctuation Frequency
        for char in self.punctuation_marks:
            count = text.count(char)
            features.append(count / num_words if num_words > 0 else 0)
            
        # 4. Function Word Frequency
        text_lower = text.lower()
        for fw in self.function_words:
            # Simple count (could be improved with regex for exact word matching)
            count = text_lower.split().count(fw)
            features.append(count / num_words if num_words > 0 else 0)
            
        # 5. Average Word Length
        avg_word_len = sum(len(w) for w in words) / num_words if num_words > 0 else 0
        features.append(avg_word_len)
        
        return torch.tensor(features, dtype=torch.float32)

    def get_feature_dim(self):
        # 1 (Avg Sent Len) + 1 (TTR) + len(punct) + len(func_words) + 1 (Avg Word Len)
        return 1 + 1 + len(self.punctuation_marks) + len(self.function_words) + 1
