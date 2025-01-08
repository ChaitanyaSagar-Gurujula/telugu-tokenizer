from tqdm import tqdm
from collections import Counter, defaultdict
import json
from datasets import load_dataset
import time
import os

class BPETokenizer:
    def __init__(self, vocab_size=5000):
        self.vocab_size = vocab_size
        self.bpe_codes = {}
        self.vocab = {idx: bytes([idx]) for idx in range(256)}  # Initialize with byte vocabulary
        self.merges = {}  # (int, int) -> int
    
    def get_stats(self, ids):
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts
    
    def merge(self, ids, pair, idx):
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids

    def save(self, path):
        """Save the tokenizer vocabulary and merges to a file."""
        # Convert bytes to lists for JSON serialization
        serializable_vocab = {
            str(k): list(v) if isinstance(v, bytes) else v 
            for k, v in self.vocab.items()
        }
        
        data = {
            'vocab_size': self.vocab_size,
            'vocab': serializable_vocab,
            'merges': {f"{p0},{p1}": idx for (p0, p1), idx in self.merges.items()}
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"\nTokenizer saved to {path}")
    
    def load(self, path):
        """Load the tokenizer vocabulary and merges from a file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.vocab_size = data['vocab_size']
        self.vocab = {
            int(k): bytes(v) if isinstance(v, list) else v 
            for k, v in data['vocab'].items()
        }
        self.merges = {
            tuple(map(int, k.split(','))): idx 
            for k, idx in data['merges'].items()
        }
        print(f"Loaded tokenizer from {path}")

    def fit(self, text):
        """Train the BPE tokenizer."""
        # Convert text to bytes and then to integers
        tokens = list(text.encode('utf-8'))
        ids = list(tokens)
        
        print(f"Before training: ids length: {len(ids)}")
        print(f"Before training: tokens length: {len(tokens)}")
        print("Before training: merges length: ", len(self.merges))
        
        num_merges = self.vocab_size - 256  # Number of merges needed
        
        # Add progress bar for training
        pbar = tqdm(range(num_merges), desc="Training BPE")
        for i in pbar:
            stats = self.get_stats(ids)
            if not stats:
                break
            pair = max(stats, key=stats.get)
            idx = 256 + i  # Start new tokens after byte range
            ids = self.merge(ids, pair, idx)
            self.merges[pair] = idx
            # Update vocabulary
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
            
            # Update progress bar with current stats
            if i % 100 == 0:
                pbar.set_postfix({
                    'vocab_size': len(self.vocab),
                    'compression': f"{len(tokens)/len(ids):.2f}x"
                })
        
        print(f"\nAfter training: ids length: {len(ids)}")
        print(f"After training: tokens length: {len(tokens)}")
        print("After training: merges length: ", len(self.merges))
        print(f"Compression ratio: {len(tokens) / len(ids):.2f}X")

    def encode(self, text):
        """Encode text to token IDs."""
        tokens = list(text.encode('utf-8'))
        while len(tokens) >= 2:
            stats = self.get_stats(tokens)
            pair = min(stats, key=lambda p: self.merges.get(p, float('inf')))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            tokens = self.merge(tokens, pair, idx)
        return tokens

    def decode(self, tokens):
        """Decode token IDs back to text."""
        bytes_tokens = b''.join(self.vocab[idx] for idx in tokens)
        return bytes_tokens.decode('utf-8', errors='replace')

    def train_on_dataset(self):
        """Train tokenizer on the Telugu news dataset"""
        print("Loading dataset...")
        dataset = load_dataset("saidines12/telugu_news_dataset")
        
        print("Preparing training text...")
        training_text = []
        split = "train" if "train" in dataset else "default"
        
        for item in dataset[split]:
            if item["headline"]: training_text.append(item["headline"])
            if item["article"]: training_text.append(item["article"])
        
        full_text = "\n".join(training_text)
        
        print("\nTraining tokenizer...")
        start_time = time.time()
        self.fit(full_text)
        print(f"Training time: {time.time() - start_time:.2f} seconds")

# In the main block:
if __name__ == "__main__":
    tokenizer = BPETokenizer(vocab_size=5000)
    
    # Check if saved tokenizer exists
    if os.path.exists('telugu_tokenizer.json'):
        print("Loading pre-trained tokenizer...")
        tokenizer.load('telugu_tokenizer.json')
    else:
        print("Training new tokenizer...")
        tokenizer.train_on_dataset()
        tokenizer.save('telugu_tokenizer.json')
    
    # Test the tokenizer
    test_text = "తెలుగు భాష"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    
    print("\nTest Results:")
    print(f"Original: {test_text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    print(f"Matches original: {test_text == decoded}")
