from tqdm import tqdm
from collections import Counter
import json
from datasets import load_dataset
import time
import os
import re

def get_telugu_char_info():
    """
    Returns a dictionary of Telugu Unicode ranges with their descriptions.
    Based on Unicode 13.0 Telugu block (0C00-0C7F).
    """
    return {
        (0x0C00, 0x0C03): "Various forms of Telugu anusvara and visarga",
        (0x0C05, 0x0C14): "Telugu vowels (అ to ఔ)",
        (0x0C15, 0x0C39): "Telugu consonants (క to హ)",
        (0x0C3D, 0x0C44): "Telugu vowel signs (ఽ to ౄ)",
        (0x0C46, 0x0C48): "Telugu vowel signs (ె to ై)",
        (0x0C4A, 0x0C4D): "Telugu vowel signs and virama (ొ to ్)",
        (0x0C55, 0x0C56): "Telugu length marks",
        (0x0C58, 0x0C5A): "Additional Telugu consonants",
        (0x0C60, 0x0C63): "Telugu vocalic letters",
        (0x0C66, 0x0C6F): "Telugu digits (౦ to ౯)",
        (0x0C78, 0x0C7F): "Telugu fraction symbols"
    }

def create_base_vocab():
    """Create a base vocabulary with ASCII and Telugu characters."""
    vocab = {}
    token_id = 0
    
    # Add ASCII characters (0-127)
    print("Adding ASCII characters...")
    for i in range(128):
        char_bytes = bytes([i])
        try:
            char = char_bytes.decode('utf-8', errors='strict')
            vocab[token_id] = {
                'text': char,
                'bytes': list(char_bytes),
                'type': 'ASCII',
                'description': f"ASCII character: {repr(char)}"
            }
            token_id += 1
        except UnicodeDecodeError:
            continue
    
    # Add Extended ASCII characters (128-255)
    print("Adding Extended ASCII characters...")
    for i in range(128, 256):
        char_bytes = bytes([i])
        try:
            # Try to decode as UTF-8 first
            char = char_bytes.decode('utf-8', errors='strict')
            vocab[token_id] = {
                'text': char if char.isprintable() else f"<{hex(i)[2:].upper()}>",
                'bytes': list(char_bytes),
                'type': 'Extended ASCII',
                'description': f"Extended ASCII character: {char} ({hex(i)})"
            }
        except UnicodeDecodeError:
            # If not valid UTF-8, store as bytes representation
            vocab[token_id] = {
                'text': f"[Bytes: {list(char_bytes)}]",
                'bytes': list(char_bytes),
                'type': 'Extended ASCII',
                'description': f"Extended ASCII byte: {hex(i)}"
            }
        token_id += 1
    
    # Add Telugu Unicode characters (0C00-0C7F)
    print("Adding Telugu characters...")
    telugu_info = get_telugu_char_info()
    
    for i in range(0x0C00, 0x0C7F + 1):
        try:
            char = chr(i)
            char_bytes = char.encode('utf-8')
            # Only add if it's a valid character
            char.encode('utf-8').decode('utf-8')
            
            # Find the character's category
            char_type = "Other Telugu Character"
            char_description = "Telugu character"
            for (start, end), desc in telugu_info.items():
                if start <= i <= end:
                    char_type = desc
                    char_description = f"Telugu character: {char} ({hex(i)})"
                    break
            
            vocab[token_id] = {
                'text': char,
                'bytes': list(char_bytes),
                'type': char_type,
                'description': char_description
            }
            token_id += 1
        except UnicodeEncodeError:
            continue
    
    print(f"Created base vocabulary with {len(vocab)} tokens")
    return vocab

def save_base_vocab(vocab, path='telugu_base_vocab.json'):
    """Save the base vocabulary with character information."""
    # Sort by character type for better readability
    sorted_vocab = {}
    for k, v in sorted(vocab.items(), key=lambda x: (x[1]['type'], x[0])):
        sorted_vocab[str(k)] = v
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(sorted_vocab, f, ensure_ascii=False, indent=2)
    print(f"Base vocabulary saved to {path}")

def load_base_vocab(path='telugu_base_vocab.json'):
    """Load the base vocabulary."""
    with open(path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    return {int(k): bytes(v['bytes']) for k, v in vocab.items()}

class BPETokenizer:
    def __init__(self, vocab_size=5000, sample_size=None):
        self.vocab_size = vocab_size
        self.sample_size = sample_size
        
        # Try to load base vocabulary, or create if it doesn't exist
        base_vocab_path = 'telugu_base_vocab.json'
        if os.path.exists(base_vocab_path):
            print("Loading existing base vocabulary...")
            self.vocab = load_base_vocab(base_vocab_path)
        else:
            print("Creating new base vocabulary...")
            base_vocab = create_base_vocab()
            save_base_vocab(base_vocab)
            self.vocab = load_base_vocab(base_vocab_path)
        
        self.base_vocab_size = len(self.vocab)
        self.merges = {}
    
    def get_stats(self, ids):
        """Count token pair frequencies."""
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts
    
    def merge(self, ids, pair, idx):
        """Merge all occurrences of a token pair."""
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

    def fit(self, text):
        """Train the BPE tokenizer."""
        print("Converting text to token IDs using base vocabulary...")
        ids = []
        byte_string = text.encode('utf-8')
        
        # Convert text to initial tokens using base vocabulary
        i = 0
        pbar = tqdm(total=len(byte_string), desc="Tokenizing text")
        while i < len(byte_string):
            found = False
            for length in [3, 2, 1]:  # Try different lengths
                if i + length <= len(byte_string):
                    char_bytes = bytes(byte_string[i:i+length])
                    # Find token ID for these bytes
                    for token_id, token_bytes in self.vocab.items():
                        if token_bytes == char_bytes:
                            ids.append(token_id)
                            i += length
                            found = True
                            pbar.update(length)
                            break
                    if found:
                        break
            if not found:
                i += 1
                pbar.update(1)
        pbar.close()
        
        print(f"\nBase vocabulary size: {self.base_vocab_size}")
        print(f"Initial sequence length: {len(ids)}")
        
        # Perform merges
        num_merges = self.vocab_size - self.base_vocab_size
        for i in tqdm(range(num_merges), desc="Training BPE"):
            stats = self.get_stats(ids)
            if not stats:
                break
                
            pair = max(stats, key=stats.get)
            idx = len(self.vocab)
            ids = self.merge(ids, pair, idx)
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
            
            if i % 100 == 0:
                try:
                    text0 = self.vocab[pair[0]].decode('utf-8')
                    text1 = self.vocab[pair[1]].decode('utf-8')
                    merged = self.vocab[idx].decode('utf-8')
                    print(f"\nMerge {i}: {text0} + {text1} = {merged}")
                except UnicodeDecodeError:
                    continue

    def encode(self, text):
        """Encode text to token IDs."""
        ids = []
        byte_string = text.encode('utf-8')
        
        # First, break into base tokens
        char_tokens = []
        i = 0
        while i < len(byte_string):
            found = False
            for length in [3, 2, 1]:
                if i + length <= len(byte_string):
                    char_bytes = bytes(byte_string[i:i+length])
                    for token_id, token_bytes in self.vocab.items():
                        if token_bytes == char_bytes:
                            char_tokens.append(token_id)
                            i += length
                            found = True
                            break
                    if found:
                        break
            if not found:
                i += 1
        
        # Then apply merges
        tokens = char_tokens
        while len(tokens) >= 2:
            pairs = list(zip(tokens, tokens[1:]))
            found_merge = False
            
            for pair in pairs:
                if pair in self.merges:
                    found_merge = True
                    tokens = self.merge(tokens, pair, self.merges[pair])
                    break
            
            if not found_merge:
                break
                
        return tokens

    def decode(self, tokens):
        """Decode token IDs back to text."""
        bytes_tokens = b''.join(self.vocab[idx] for idx in tokens)
        return bytes_tokens.decode('utf-8')

    def save(self, path):
        """Save the tokenizer mappings to files."""
        base_path = path.rsplit('.', 1)[0]
        
        # Save vocabulary with human-readable form
        vocab_mapping = {}
        for token_id, byte_seq in self.vocab.items():
            try:
                text = byte_seq.decode('utf-8')
                vocab_mapping[token_id] = {
                    'text': text,
                    'bytes': list(byte_seq),
                    'is_base': token_id < self.base_vocab_size
                }
            except UnicodeDecodeError:
                vocab_mapping[token_id] = {
                    'text': f"[Bytes: {list(byte_seq)}]",
                    'bytes': list(byte_seq),
                    'is_base': token_id < self.base_vocab_size
                }
        
        # Save merge patterns with human-readable form
        merge_patterns = {}
        for (p0, p1), idx in self.merges.items():
            try:
                text0 = self.vocab[p0].decode('utf-8')
                text1 = self.vocab[p1].decode('utf-8')
                merged = self.vocab[idx].decode('utf-8')
                merge_patterns[idx] = {
                    'parts': [text0, text1],
                    'result': merged,
                    'token_ids': [p0, p1]
                }
            except UnicodeDecodeError:
                merge_patterns[idx] = {
                    'parts': [f"Token_{p0}", f"Token_{p1}"],
                    'result': f"Token_{idx}",
                    'token_ids': [p0, p1]
                }
        
        with open(f"{base_path}_vocab.json", 'w', encoding='utf-8') as f:
            json.dump(vocab_mapping, f, ensure_ascii=False, indent=2)
        
        with open(f"{base_path}_merges.json", 'w', encoding='utf-8') as f:
            json.dump(merge_patterns, f, ensure_ascii=False, indent=2)
        
        print(f"\nTokenizer mappings saved to {base_path}_vocab.json and {base_path}_merges.json")

    def load(self, path):
        """Load the tokenizer from mapping files."""
        base_path = path.rsplit('.', 1)[0]
        
        with open(f"{base_path}_vocab.json", 'r', encoding='utf-8') as f:
            vocab_mapping = json.load(f)
            self.vocab = {
                int(k): bytes(v['bytes']) 
                for k, v in vocab_mapping.items()
            }
            # Find base vocabulary size
            self.base_vocab_size = sum(1 for k, v in vocab_mapping.items() if v['is_base'])
        
        with open(f"{base_path}_merges.json", 'r', encoding='utf-8') as f:
            merge_patterns = json.load(f)
            self.merges = {
                tuple(v['token_ids']): int(k)
                for k, v in merge_patterns.items()
            }
        
        self.vocab_size = len(self.vocab)
        print(f"Loaded tokenizer from {base_path}_*.json files")

    def train_on_dataset(self):
        """Train tokenizer on the Telugu news dataset."""
        print("Loading dataset...")
        dataset = load_dataset("saidines12/telugu_news_dataset")
        
        print("Preparing training text...")
        training_text = []
        split = "train" if "train" in dataset else "default"
        
        for item in tqdm(dataset[split], desc="Loading documents"):
            if item["headline"]: training_text.append(item["headline"])
            if item["article"]: training_text.append(item["article"])
            
            if self.sample_size and len(training_text) >= self.sample_size:
                print(f"Using first {self.sample_size} documents for training")
                break
        
        full_text = "\n".join(training_text)
        print(f"\nTraining on {len(training_text)} documents...")
        print(f"Total characters in training data: {len(full_text):,}")
        
        start_time = time.time()
        self.fit(full_text)
        print(f"Training time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    # For quick testing, use a small sample
    tokenizer = BPETokenizer(vocab_size=5000, sample_size=2000)
    
    vocab_file = 'telugu_tokenizer_vocab.json'
    merges_file = 'telugu_tokenizer_merges.json'
    
    if os.path.exists(vocab_file) and os.path.exists(merges_file):
        print("Loading pre-trained tokenizer...")
        tokenizer.load('telugu_tokenizer')
    else:
        print("Training new tokenizer...")
        tokenizer.train_on_dataset()
        tokenizer.save('telugu_tokenizer')
    
    # Test the tokenizer
    test_text = "తెలుగు భాష"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    
    print("\nTest Results:")
    print(f"Original: {test_text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    print(f"Matches original: {test_text == decoded}")
