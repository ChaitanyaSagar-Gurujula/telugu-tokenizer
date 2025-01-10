from tqdm import tqdm
from collections import Counter
import json
from datasets import load_dataset
import time
import os
import re
import pandas as pd
from multiprocessing import Pool
import array

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
    """Create a base vocabulary with ASCII, Telugu characters, and common ligatures."""
    vocab = {}
    token_id = 0
    existing_tokens = set()  # Set to track existing tokens
    
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
    
    # Define Telugu consonants and vowel signs
    consonants = [
        'క', 'ఖ', 'గ', 'ఘ', 'ఙ', 'చ', 'ఛ', 'జ', 'ఝ', 'ఞ',
        'ట', 'ఠ', 'డ', 'ఢ', 'ణ', 'త', 'థ', 'ద', 'ధ', 'న',
        'ప', 'ఫ', 'బ', 'భ', 'మ', 'య', 'ర', 'ల', 'వ', 'శ',
        'ష', 'స', 'హ', 'ళ', 'క్ష', 'ఱ'
    ]
    
    vowel_signs = [
        '', 'ా', 'ి', 'ీ', 'ు', 'ూ', 'ృ', 'ౄ', 'ౢ', 'ౣ', 'ె', 'ే', 'ై', 'ొ', 'ో', 'ౌ', 'ం', 'ః', 'ఁ', '్'
    ]

    
    # Add common Telugu ligatures with existing vowel signs
    print("Adding common Telugu ligatures with existing vowel signs...")
    for consonant in consonants:
        for vowel_sign in vowel_signs:
            ligature = consonant + vowel_sign
            if ligature not in existing_tokens:  # Check for duplicates
                char_bytes = ligature.encode('utf-8')
                vocab[token_id] = {
                    'text': ligature,
                    'bytes': list(char_bytes),
                    'type': 'Ligature',
                    'description': f"Telugu ligature: {ligature}"
                }
                existing_tokens.add(ligature)  # Add to the set
                token_id += 1
    
    # Add valid consonant combinations
    print("Adding valid consonant combinations...")
    valid_consonant_combinations = [
        'క్క', 'క్ఖ', 'క్గ', 'క్ఘ', 'క్ఙ', 'క్చ', 'క్ఛ', 'క్జ', 'క్ఝ', 'క్ఞ',
        'క్ట', 'క్ఠ', 'క్డ', 'క్ఢ', 'క్ణ', 'క్త', 'క్థ', 'క్ద', 'క్ధ', 'క్న',
        'క్ప', 'క్ఫ', 'క్బ', 'క్భ', 'క్మ', 'క్య', 'క్ర', 'క్ల', 'క్వ', 'క్శ',
        'క్ష', 'క్స', 'క్హ', 'క్ళ', 'క్క్ష', 'క్ఱ',
        'ఖ్క', 'ఖ్ఖ', 'ఖ్గ', 'ఖ్ఘ', 'ఖ్ఙ', 'ఖ్చ', 'ఖ్ఛ', 'ఖ్జ', 'ఖ్ఝ', 'ఖ్ఞ',
        'ఖ్ట', 'ఖ్ఠ', 'ఖ్డ', 'ఖ్ఢ', 'ఖ్ణ', 'ఖ్త', 'ఖ్థ', 'ఖ్ద', 'ఖ్ధ', 'ఖ్న',
        'ఖ్ప', 'ఖ్ఫ', 'ఖ్బ', 'ఖ్భ', 'ఖ్మ', 'ఖ్య', 'ఖ్ర', 'ఖ్ల', 'ఖ్వ', 'ఖ్శ',
        'ఖ్ష', 'ఖ్స', 'ఖ్హ', 'ఖ్ళ', 'ఖ్క్ష', 'ఖ్ఱ',
        'గ్క', 'గ్ఖ', 'గ్గ', 'గ్ఘ', 'గ్ఙ', 'గ్చ', 'గ్ఛ', 'గ్జ', 'గ్ఝ', 'గ్ఞ',
        'గ్ట', 'గ్ఠ', 'గ్డ', 'గ్ఢ', 'గ్ణ', 'గ్త', 'గ్థ', 'గ్ద', 'గ్ధ', 'గ్న',
        'గ్ప', 'గ్ఫ', 'గ్బ', 'గ్భ', 'గ్మ', 'గ్య', 'గ్ర', 'గ్ల', 'గ్వ', 'గ్శ',
        'గ్ష', 'గ్స', 'గ్హ', 'గ్ళ', 'గ్క్ష', 'గ్ఱ',
        'ఘ్క', 'ఘ్ఖ', 'ఘ్గ', 'ఘ్ఘ', 'ఘ్ఙ', 'ఘ్చ', 'ఘ్ఛ', 'ఘ్జ', 'ఘ్ఝ', 'ఘ్ఞ',
        'ఘ్ట', 'ఘ్ఠ', 'ఘ్డ', 'ఘ్ఢ', 'ఘ్ణ', 'ఘ్త', 'ఘ్థ', 'ఘ్ద', 'ఘ్ధ', 'ఘ్న',
        'ఘ్ప', 'ఘ్ఫ', 'ఘ్బ', 'ఘ్భ', 'ఘ్మ', 'ఘ్య', 'ఘ్ర', 'ఘ్ల', 'ఘ్వ', 'ఘ్శ',
        'ఘ్ష', 'ఘ్స', 'ఘ్హ', 'ఘ్ళ', 'ఘ్క్ష', 'ఘ్ఱ',
        'ఙ్క', 'ఙ్ఖ', 'ఙ్గ', 'ఙ్ఘ', 'ఙ్ఙ', 'ఙ్చ', 'ఙ్ఛ', 'ఙ్జ', 'ఙ్ఝ', 'ఙ్ఞ',
        'ఙ్ట', 'ఙ్ఠ', 'ఙ్డ', 'ఙ్ఢ', 'ఙ్ణ', 'ఙ్త', 'ఙ్థ', 'ఙ్ద', 'ఙ్ధ', 'ఙ్న',
        'ఙ్ప', 'ఙ్ఫ', 'ఙ్బ', 'ఙ్భ', 'ఙ్మ', 'ఙ్య', 'ఙ్ర', 'ఙ్ల', 'ఙ్వ', 'ఙ్శ',
        'ఙ్ష', 'ఙ్స', 'ఙ్హ', 'ఙ్ళ', 'ఙ్క్ష', 'ఙ్ఱ',
        'చ్క', 'చ్ఖ', 'చ్గ', 'చ్ఘ', 'చ్ఙ', 'చ్చ', 'చ్ఛ', 'చ్జ', 'చ్ఝ', 'చ్ఞ',
        'చ్ట', 'చ్ఠ', 'చ్డ', 'చ్ఢ', 'చ్ణ', 'చ్త', 'చ్థ', 'చ్ద', 'చ్ధ', 'చ్న',
        'చ్ప', 'చ్ఫ', 'చ్బ', 'చ్భ', 'చ్మ', 'చ్య', 'చ్ర', 'చ్ల', 'చ్వ', 'చ్శ',
        'చ్ష', 'చ్స', 'చ్హ', 'చ్ళ', 'చ్క్ష', 'చ్ఱ',
        'ఛ్క', 'ఛ్ఖ', 'ఛ్గ', 'ఛ్ఘ', 'ఛ్ఙ', 'ఛ్చ', 'ఛ్ఛ', 'ఛ్జ', 'ఛ్ఝ', 'ఛ్ఞ',
        'ఛ్ట', 'ఛ్ఠ', 'ఛ్డ', 'ఛ్ఢ', 'ఛ్ణ', 'ఛ్త', 'ఛ్థ', 'ఛ్ద', 'ఛ్ధ', 'ఛ్న',
        'ఛ్ప', 'ఛ్ఫ', 'ఛ్బ', 'ఛ్భ', 'ఛ్మ', 'ఛ్య', 'ఛ్ర', 'ఛ్ల', 'ఛ్వ', 'ఛ్శ',
        'ఛ్ష', 'ఛ్స', 'ఛ్హ', 'ఛ్ళ', 'ఛ్క్ష', 'ఛ్ఱ',
        'జ్క', 'జ్ఖ', 'జ్గ', 'జ్ఘ', 'జ్ఙ', 'జ్చ', 'జ్ఛ', 'జ్జ', 'జ్ఝ', 'జ్ఞ',
        'జ్ట', 'జ్ఠ', 'జ్డ', 'జ్ఢ', 'జ్ణ', 'జ్త', 'జ్థ', 'జ్ద', 'జ్ధ', 'జ్న',
        'జ్ప', 'జ్ఫ', 'జ్బ', 'జ్భ', 'జ్మ', 'జ్య', 'జ్ర', 'జ్ల', 'జ్వ', 'జ్శ',
        'జ్ష', 'జ్స', 'జ్హ', 'జ్ళ', 'జ్క్ష', 'జ్ఱ',
        'ఝ్క', 'ఝ్ఖ', 'ఝ్గ', 'ఝ్ఘ', 'ఝ్ఙ', 'ఝ్చ', 'ఝ్ఛ', 'ఝ్జ', 'ఝ్ఝ', 'ఝ్ఞ',
        'ఝ్ట', 'ఝ్ఠ', 'ఝ్డ', 'ఝ్ఢ', 'ఝ్ణ', 'ఝ్త', 'ఝ్థ', 'ఝ్ద', 'ఝ్ధ', 'ఝ్న',
        'ఝ్ప', 'ఝ్ఫ', 'ఝ్బ', 'ఝ్భ', 'ఝ్మ', 'ఝ్య', 'ఝ్ర', 'ఝ్ల', 'ఝ్వ', 'ఝ్శ',
        'ఝ్ష', 'ఝ్స', 'ఝ్హ', 'ఝ్ళ', 'ఝ్క్ష', 'ఝ్ఱ',
        'ఞ్క', 'ఞ్ఖ', 'ఞ్గ', 'ఞ్ఘ', 'ఞ్ఙ', 'ఞ్చ', 'ఞ్ఛ', 'ఞ్జ', 'ఞ్ఝ', 'ఞ్ఞ',
        'ఞ్ట', 'ఞ్ఠ', 'ఞ్డ', 'ఞ్ఢ', 'ఞ్ణ', 'ఞ్త', 'ఞ్థ', 'ఞ్ద', 'ఞ్ధ', 'ఞ్న',
        'ఞ్ప', 'ఞ్ఫ', 'ఞ్బ', 'ఞ్భ', 'ఞ్మ', 'ఞ్య', 'ఞ్ర', 'ఞ్ల', 'ఞ్వ', 'ఞ్శ',
        'ఞ్ష', 'ఞ్స', 'ఞ్హ', 'ఞ్ళ', 'ఞ్క్ష', 'ఞ్ఱ',
        'ట్క', 'ట్ఖ', 'ట్గ', 'ట్ఘ', 'ట్ఙ', 'ట్చ', 'ట్ఛ', 'ట్జ', 'ట్ఝ', 'ట్ఞ',
        'ట్ట', 'ట్ఠ', 'ట్డ', 'ట్ఢ', 'ట్ణ', 'ట్త', 'ట్థ', 'ట్ద', 'ట్ధ', 'ట్న',
        'ట్ప', 'ట్ఫ', 'ట్బ', 'ట్భ', 'ట్మ', 'ట్య', 'ట్ర', 'ట్ల', 'ట్వ', 'ట్శ',
        'ట్ష', 'ట్స', 'ట్హ', 'ట్ళ', 'ట్క్ష', 'ట్ఱ',
        'ఠ్క', 'ఠ్ఖ', 'ఠ్గ', 'ఠ్ఘ', 'ఠ్ఙ', 'ఠ్చ', 'ఠ్ఛ', 'ఠ్జ', 'ఠ్ఝ', 'ఠ్ఞ',
        'ఠ్ట', 'ఠ్ఠ', 'ఠ్డ', 'ఠ్ఢ', 'ఠ్ణ', 'ఠ్త', 'ఠ్థ', 'ఠ్ద', 'ఠ్ధ', 'ఠ్న',
        'ఠ్ప', 'ఠ్ఫ', 'ఠ్బ', 'ఠ్భ', 'ఠ్మ', 'ఠ్య', 'ఠ్ర', 'ఠ్ల', 'ఠ్వ', 'ఠ్శ',
        'ఠ్ష', 'ఠ్స', 'ఠ్హ', 'ఠ్ళ', 'ఠ్క్ష', 'ఠ్ఱ',
        'డ్క', 'డ్ఖ', 'డ్గ', 'డ్ఘ', 'డ్ఙ', 'డ్చ', 'డ్ఛ', 'డ్జ', 'డ్ఝ', 'డ్ఞ',
        'డ్ట', 'డ్ఠ', 'డ్డ', 'డ్ఢ', 'డ్ణ', 'డ్త', 'డ్థ', 'డ్ద', 'డ్ధ', 'డ్న',
        'డ్ప', 'డ్ఫ', 'డ్బ', 'డ్భ', 'డ్మ', 'డ్య', 'డ్ర', 'డ్ల', 'డ్వ', 'డ్శ',
        'డ్ష', 'డ్స', 'డ్హ', 'డ్ళ', 'డ్క్ష', 'డ్ఱ',
        'ఢ్క', 'ఢ్ఖ', 'ఢ్గ', 'ఢ్ఘ', 'ఢ్ఙ', 'ఢ్చ', 'ఢ్ఛ', 'ఢ్జ', 'ఢ్ఝ', 'ఢ్ఞ',
        'ఢ్ట', 'ఢ్ఠ', 'ఢ్డ', 'ఢ్ఢ', 'ఢ్ణ', 'ఢ్త', 'ఢ్థ', 'ఢ్ద', 'ఢ్ధ', 'ఢ్న',
        'ఢ్ప', 'ఢ్ఫ', 'ఢ్బ', 'ఢ్భ', 'ఢ్మ', 'ఢ్య', 'ఢ్ర', 'ఢ్ల', 'ఢ్వ', 'ఢ్శ',
        'ఢ్ష', 'ఢ్స', 'ఢ్హ', 'ఢ్ళ', 'ఢ్క్ష', 'ఢ్ఱ',
        'ణ్క', 'ణ్ఖ', 'ణ్గ', 'ణ్ఘ', 'ణ్ఙ', 'ణ్చ', 'ణ్ఛ', 'ణ్జ', 'ణ్ఝ', 'ణ్ఞ',
        'ణ్ట', 'ణ్ఠ', 'ణ్డ', 'ణ్ఢ', 'ణ్ణ', 'ణ్త', 'ణ్థ', 'ణ్ద', 'ణ్ధ', 'ణ్న',
        'ణ్ప', 'ణ్ఫ', 'ణ్బ', 'ణ్భ', 'ణ్మ', 'ణ్య', 'ణ్ర', 'ణ్ల', 'ణ్వ', 'ణ్శ',
        'ణ్ష', 'ణ్స', 'ణ్హ', 'ణ్ళ', 'ణ్క్ష', 'ణ్ఱ',
        'త్క', 'త్ఖ', 'త్గ', 'త్ఘ', 'త్ఙ', 'త్చ', 'త్ఛ', 'త్జ', 'త్ఝ', 'త్ఞ',
        'త్ట', 'త్ఠ', 'త్డ', 'త్ఢ', 'త్ణ', 'త్త', 'త్థ', 'త్ద', 'త్ధ', 'త్న',
        'త్ప', 'త్ఫ', 'త్బ', 'త్భ', 'త్మ', 'త్య', 'త్ర', 'త్ల', 'త్వ', 'త్శ',
        'త్ష', 'త్స', 'త్హ', 'త్ళ', 'త్క్ష', 'త్ఱ',
        'థ్క', 'థ్ఖ', 'థ్గ', 'థ్ఘ', 'థ్ఙ', 'థ్చ', 'థ్ఛ', 'థ్జ', 'థ్ఝ', 'థ్ఞ',
        'థ్ట', 'థ్ఠ', 'థ్డ', 'థ్ఢ', 'థ్ణ', 'థ్త', 'థ్థ', 'థ్ద', 'థ్ధ', 'థ్న',
        'థ్ప', 'థ్ఫ', 'థ్బ', 'థ్భ', 'థ్మ', 'థ్య', 'థ్ర', 'థ్ల', 'థ్వ', 'థ్శ',
        'థ్ష', 'థ్స', 'థ్హ', 'థ్ళ', 'థ్క్ష', 'థ్ఱ',
        'ద్క', 'ద్ఖ', 'ద్గ', 'ద్ఘ', 'ద్ఙ', 'ద్చ', 'ద్ఛ', 'ద్జ', 'ద్ఝ', 'ద్ఞ',
        'ద్ట', 'ద్ఠ', 'ద్డ', 'ద్ఢ', 'ద్ణ', 'ద్త', 'ద్థ', 'ద్ద', 'ద్ధ', 'ద్న',
        'ద్ప', 'ద్ఫ', 'ద్బ', 'ద్భ', 'ద్మ', 'ద్య', 'ద్ర', 'ద్ల', 'ద్వ', 'ద్శ',
        'ద్ష', 'ద్స', 'ద్హ', 'ద్ళ', 'ద్క్ష', 'ద్ఱ',
        'ధ్క', 'ధ్ఖ', 'ధ్గ', 'ధ్ఘ', 'ధ్ఙ', 'ధ్చ', 'ధ్ఛ', 'ధ్జ', 'ధ్ఝ', 'ధ్ఞ',
        'ధ్ట', 'ధ్ఠ', 'ధ్డ', 'ధ్ఢ', 'ధ్ణ', 'ధ్త', 'ధ్థ', 'ధ్ద', 'ధ్ధ', 'ధ్న',
        'ధ్ప', 'ధ్ఫ', 'ధ్బ', 'ధ్భ', 'ధ్మ', 'ధ్య', 'ధ్ర', 'ధ్ల', 'ధ్వ', 'ధ్శ',
        'ధ్ష', 'ధ్స', 'ధ్హ', 'ధ్ళ', 'ధ్క్ష', 'ధ్ఱ',
        'న్క', 'న్ఖ', 'న్గ', 'న్ఘ', 'న్ఙ', 'న్చ', 'న్ఛ', 'న్జ', 'న్ఝ', 'న్ఞ',
        'న్ట', 'న్ఠ', 'న్డ', 'న్ఢ', 'న్ణ', 'న్త', 'న్థ', 'న్ద', 'న్ధ', 'న్న',
        'న్ప', 'న్ఫ', 'న్బ', 'న్భ', 'న్మ', 'న్య', 'న్ర', 'న్ల', 'న్వ', 'న్శ',
        'న్ష', 'న్స', 'న్హ', 'న్ళ', 'న్క్ష', 'న్ఱ',
        'ప్క', 'ప్ఖ', 'ప్గ', 'ప్ఘ', 'ప్ఙ', 'ప్చ', 'ప్ఛ', 'ప్జ', 'ప్ఝ', 'ప్ఞ',
        'ప్ట', 'ప్ఠ', 'ప్డ', 'ప్ఢ', 'ప్ణ', 'ప్త', 'ప్థ', 'ప్ద', 'ప్ధ', 'ప్న',
        'ప్ప', 'ప్ఫ', 'ప్బ', 'ప్భ', 'ప్మ', 'ప్య', 'ప్ర', 'ప్ల', 'ప్వ', 'ప్శ',
        'ప్ష', 'ప్స', 'ప్హ', 'ప్ళ', 'ప్క్ష', 'ప్ఱ',
        'ఫ్క', 'ఫ్ఖ', 'ఫ్గ', 'ఫ్ఘ', 'ఫ్ఙ', 'ఫ్చ', 'ఫ్ఛ', 'ఫ్జ', 'ఫ్ఝ', 'ఫ్ఞ',
        'ఫ్ట', 'ఫ్ఠ', 'ఫ్డ', 'ఫ్ఢ', 'ఫ్ణ', 'ఫ్త', 'ఫ్థ', 'ఫ్ద', 'ఫ్ధ', 'ఫ్న',
        'ఫ్ప', 'ఫ్ఫ', 'ఫ్బ', 'ఫ్భ', 'ఫ్మ', 'ఫ్య', 'ఫ్ర', 'ఫ్ల', 'ఫ్వ', 'ఫ్శ',
        'ఫ్ష', 'ఫ్స', 'ఫ్హ', 'ఫ్ళ', 'ఫ్క్ష', 'ఫ్ఱ',
        'బ్క', 'బ్ఖ', 'బ్గ', 'బ్ఘ', 'బ్ఙ', 'బ్చ', 'బ్ఛ', 'బ్జ', 'బ్ఝ', 'బ్ఞ',
        'బ్ట', 'బ్ఠ', 'బ్డ', 'బ్ఢ', 'బ్ణ', 'బ్త', 'బ్థ', 'బ్ద', 'బ్ధ', 'బ్న',
        'బ్ప', 'బ్ఫ', 'బ్బ', 'బ్భ', 'బ్మ', 'బ్య', 'బ్ర', 'బ్ల', 'బ్వ', 'బ్శ',
        'బ్ష', 'బ్స', 'బ్హ', 'బ్ళ', 'బ్క్ష', 'బ్ఱ',
        'భ్క', 'భ్ఖ', 'భ్గ', 'భ్ఘ', 'భ్ఙ', 'భ్చ', 'భ్ఛ', 'భ్జ', 'భ్ఝ', 'భ్ఞ',
        'భ్ట', 'భ్ఠ', 'భ్డ', 'భ్ఢ', 'భ్ణ', 'భ్త', 'భ్థ', 'భ్ద', 'భ్ధ', 'భ్న',
        'భ్ప', 'భ్ఫ', 'భ్బ', 'భ్భ', 'భ్మ', 'భ్య', 'భ్ర', 'భ్ల', 'భ్వ', 'భ్శ',
        'భ్ష', 'భ్స', 'భ్హ', 'భ్ళ', 'భ్క్ష', 'భ్ఱ',
        'మ్క', 'మ్ఖ', 'మ్గ', 'మ్ఘ', 'మ్ఙ', 'మ్చ', 'మ్ఛ', 'మ్జ', 'మ్ఝ', 'మ్ఞ',
        'మ్ట', 'మ్ఠ', 'మ్డ', 'మ్ఢ', 'మ్ణ', 'మ్త', 'మ్థ', 'మ్ద', 'మ్ధ', 'మ్న',
        'మ్ప', 'మ్ఫ', 'మ్బ', 'మ్భ', 'మ్మ', 'మ్య', 'మ్ర', 'మ్ల', 'మ్వ', 'మ్శ',
        'మ్ష', 'మ్స', 'మ్హ', 'మ్ళ', 'మ్క్ష', 'మ్ఱ',
        'య్క', 'య్ఖ', 'య్గ', 'య్ఘ', 'య్ఙ', 'య్చ', 'య్ఛ', 'య్జ', 'య్ఝ', 'య్ఞ',
        'య్ట', 'య్ఠ', 'య్డ', 'య్ఢ', 'య్ణ', 'య్త', 'య్థ', 'య్ద', 'య్ధ', 'య్న',
        'య్ప', 'య్ఫ', 'య్బ', 'య్భ', 'య్మ', 'య్య', 'య్ర', 'య్ల', 'య్వ', 'య్శ',
        'య్ష', 'య్స', 'య్హ', 'య్ళ', 'య్క్ష', 'య్ఱ',
        'ర్క', 'ర్ఖ', 'ర్గ', 'ర్ఘ', 'ర్ఙ', 'ర్చ', 'ర్ఛ', 'ర్జ', 'ర్ఝ', 'ర్ఞ',
        'ర్ట', 'ర్ఠ', 'ర్డ', 'ర్ఢ', 'ర్ణ', 'ర్త', 'ర్థ', 'ర్ద', 'ర్ధ', 'ర్న',
        'ర్ప', 'ర్ఫ', 'ర్బ', 'ర్భ', 'ర్మ', 'ర్య', 'ర్ర', 'ర్ల', 'ర్వ', 'ర్శ',
        'ర్ష', 'ర్స', 'ర్హ', 'ర్ళ', 'ర్క్ష', 'ర్ఱ',
        'ల్క', 'ల్ఖ', 'ల్గ', 'ల్ఘ', 'ల్ఙ', 'ల్చ', 'ల్ఛ', 'ల్జ', 'ల్ఝ', 'ల్ఞ',
        'ల్ట', 'ల్ఠ', 'ల్డ', 'ల్ఢ', 'ల్ణ', 'ల్త', 'ల్థ', 'ల్ద', 'ల్ధ', 'ల్న',
        'ల్ప', 'ల్ఫ', 'ల్బ', 'ల్భ', 'ల్మ', 'ల్య', 'ల్ర', 'ల్ల', 'ల్వ', 'ల్శ',
        'ల్ష', 'ల్స', 'ల్హ', 'ల్ళ', 'ల్క్ష', 'ల్ఱ',
        'వ్క', 'వ్ఖ', 'వ్గ', 'వ్ఘ', 'వ్ఙ', 'వ్చ', 'వ్ఛ', 'వ్జ', 'వ్ఝ', 'వ్ఞ',
        'వ్ట', 'వ్ఠ', 'వ్డ', 'వ్ఢ', 'వ్ణ', 'వ్త', 'వ్థ', 'వ్ద', 'వ్ధ', 'వ్న',
        'వ్ప', 'వ్ఫ', 'వ్బ', 'వ్భ', 'వ్మ', 'వ్య', 'వ్ర', 'వ్ల', 'వ్వ', 'వ్శ',
        'వ్ష', 'వ్స', 'వ్హ', 'వ్ళ', 'వ్క్ష', 'వ్ఱ',
        'శ్క', 'శ్ఖ', 'శ్గ', 'శ్ఘ', 'శ్ఙ', 'శ్చ', 'శ్ఛ', 'శ్జ', 'శ్ఝ', 'శ్ఞ',
        'శ్ట', 'శ్ఠ', 'శ్డ', 'శ్ఢ', 'శ్ణ', 'శ్త', 'శ్థ', 'శ్ద', 'శ్ధ', 'శ్న',
        'శ్ప', 'శ్ఫ', 'శ్బ', 'శ్భ', 'శ్మ', 'శ్య', 'శ్ర', 'శ్ల', 'శ్వ', 'శ్శ',
        'శ్ష', 'శ్స', 'శ్హ', 'శ్ళ', 'శ్క్ష', 'శ్ఱ',
        'ష్క', 'ష్ఖ', 'ష్గ', 'ష్ఘ', 'ష్ఙ', 'ష్చ', 'ష్ఛ', 'ష్జ', 'ష్ఝ', 'ష్ఞ',
        'ష్ట', 'ష్ఠ', 'ష్డ', 'ష్ఢ', 'ష్ణ', 'ష్త', 'ష్థ', 'ష్ద', 'ష్ధ', 'ష్న',
        'ష్ప', 'ష్ఫ', 'ష్బ', 'ష్భ', 'ష్మ', 'ష్య', 'ష్ర', 'ష్ల', 'ష్వ', 'ష్శ',
        'ష్ష', 'ష్స', 'ష్హ', 'ష్ళ', 'ష్క్ష', 'ష్ఱ',
        'స్క', 'స్ఖ', 'స్గ', 'స్ఘ', 'స్ఙ', 'స్చ', 'స్ఛ', 'స్జ', 'స్ఝ', 'స్ఞ',
        'స్ట', 'స్ఠ', 'స్డ', 'స్ఢ', 'స్ణ', 'స్త', 'స్థ', 'స్ద', 'స్ధ', 'స్న',
        'స్ప', 'స్ఫ', 'స్బ', 'స్భ', 'స్మ', 'స్య', 'స్ర', 'స్ల', 'స్వ', 'స్శ',
        'స్ష', 'స్స', 'స్హ', 'స్ళ', 'స్క్ష', 'స్ఱ',
        'హ్క', 'హ్ఖ', 'హ్గ', 'హ్ఘ', 'హ్ఙ', 'హ్చ', 'హ్ఛ', 'హ్జ', 'హ్ఝ', 'హ్ఞ',
        'హ్ట', 'హ్ఠ', 'హ్డ', 'హ్ఢ', 'హ్ణ', 'హ్త', 'హ్థ', 'హ్ద', 'హ్ధ', 'హ్న',
        'హ్ప', 'హ్ఫ', 'హ్బ', 'హ్భ', 'హ్మ', 'హ్య', 'హ్ర', 'హ్ల', 'హ్వ', 'హ్శ',
        'హ్ష', 'హ్స', 'హ్హ', 'హ్ళ', 'హ్క్ష', 'హ్ఱ',
        'ళ్క', 'ళ్ఖ', 'ళ్గ', 'ళ్ఘ', 'ళ్ఙ', 'ళ్చ', 'ళ్ఛ', 'ళ్జ', 'ళ్ఝ', 'ళ్ఞ',
        'ళ్ట', 'ళ్ఠ', 'ళ్డ', 'ళ్ఢ', 'ళ్ణ', 'ళ్త', 'ళ్థ', 'ళ్ద', 'ళ్ధ', 'ళ్న',
        'ళ్ప', 'ళ్ఫ', 'ళ్బ', 'ళ్భ', 'ళ్మ', 'ళ్య', 'ళ్ర', 'ళ్ల', 'ళ్వ', 'ళ్శ',
        'ళ్ష', 'ళ్స', 'ళ్హ', 'ళ్ళ', 'ళ్క్ష', 'ళ్ఱ',
        'క్ష్క', 'క్ష్ఖ', 'క్ష్గ', 'క్ష్ఘ', 'క్ష్ఙ', 'క్ష్చ', 'క్ష్ఛ', 'క్ష్జ', 'క్ష్ఝ', 'క్ష్ఞ',
        'క్ష్ట', 'క్ష్ఠ', 'క్ష్డ', 'క్ష్ఢ', 'క్ష్ణ', 'క్ష్త', 'క్ష్థ', 'క్ష్ద', 'క్ష్ధ', 'క్ష్న',
        'క్ష్ప', 'క్ష్ఫ', 'క్ష్బ', 'క్ష్భ', 'క్ష్మ', 'క్ష్య', 'క్ష్ర', 'క్ష్ల', 'క్ష్వ', 'క్ష్శ',
        'క్ష్ష', 'క్ష్స', 'క్ష్హ', 'క్ష్ళ', 'క్ష్క్ష', 'క్ష్ఱ',
        'ఱ్క', 'ఱ్ఖ', 'ఱ్గ', 'ఱ్ఘ', 'ఱ్ఙ', 'ఱ్చ', 'ఱ్ఛ', 'ఱ్జ', 'ఱ్ఝ', 'ఱ్ఞ',
        'ఱ్ట', 'ఱ్ఠ', 'ఱ్డ', 'ఱ్ఢ', 'ఱ్ణ', 'ఱ్త', 'ఱ్థ', 'ఱ్ద', 'ఱ్ధ', 'ఱ్న',
        'ఱ్ప', 'ఱ్ఫ', 'ఱ్బ', 'ఱ్భ', 'ఱ్మ', 'ఱ్య', 'ఱ్ర', 'ఱ్ల', 'ఱ్వ', 'ఱ్శ',
        'ఱ్ష', 'ఱ్స', 'ఱ్హ', 'ఱ్ళ', 'ఱ్క్ష', 'ఱ్ఱ'
        # Add more valid combinations as needed
    ]
    
    for combination in valid_consonant_combinations:
        if combination not in existing_tokens:  # Check for duplicates
            char_bytes = combination.encode('utf-8')
            vocab[token_id] = {
                'text': combination,
                'bytes': list(char_bytes),
                'type': 'Ligature',
                'description': f"Telugu ligature: {combination}"
            }
            existing_tokens.add(combination)  # Add to the set
            token_id += 1
    
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
        
        # First try to load trained vocabulary
        trained_vocab_path = 'telugu_tokenizer_vocab.json'
        if os.path.exists(trained_vocab_path):
            print("Loading trained vocabulary...")
            self.load('telugu_tokenizer')  # This loads both vocab and merges
            return
        
        # If no trained vocab exists, fall back to base vocabulary
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
        # Create the merged token
        merged_token = self.vocab[pair[0]] + self.vocab[pair[1]]
        
        # Check if the merged token already exists in the vocabulary
        for existing_id, existing_token in self.vocab.items():
            if existing_token == merged_token:
                # Instead of skipping, use the existing token ID for merging
                print(f"Merge for {pair} already exists in the vocabulary.")
                newids = []
                i = 0
                while i < len(ids):
                    if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                        newids.append(existing_id)
                        i += 2
                    else:
                        newids.append(ids[i])
                        i += 1
                return newids

        # If we get here, the merged token doesn't exist yet
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids

    def _process_chunk(self, args):
        """Process a chunk of text for parallel processing."""
        chunk, byte_to_token = args
        ids = array.array('I')  # Unsigned int array
        j = 0
        while j < len(chunk):
            if chunk[j] == 32:  # Space
                ids.append(32)
                j += 1
                continue
            
            found = False
            for length in [3, 2, 1]:
                if j + length <= len(chunk):
                    char_bytes = bytes(chunk[j:j+length])
                    if char_bytes in byte_to_token:
                        ids.append(byte_to_token[char_bytes])
                        j += length
                        found = True
                        break
            if not found:
                j += 1
        return ids

    def fit(self, text):
        """Train the BPE tokenizer."""
        print("Converting text to token IDs using base vocabulary...")
        
        original_bytes = text.encode('utf-8')
        original_length = len(original_bytes)
        print(f"\nBefore training: text bytes length: {original_length:,}")
        
        # Pre-compute byte sequences for faster lookup
        byte_to_token = {token_bytes: token_id for token_id, token_bytes in self.vocab.items()}
        
        # Parallel processing of chunks
        num_cores = os.cpu_count() or 1
        chunk_size = max(1024 * 64, len(original_bytes) // (num_cores * 4))  # Larger chunks
        chunks = [original_bytes[i:i + chunk_size] for i in range(0, len(original_bytes), chunk_size)]
        
        print(f"Processing {len(chunks)} chunks using {num_cores} cores...")
        
        # Process chunks in parallel
        with Pool(num_cores) as pool:
            chunk_results = list(tqdm(
                pool.imap(self._process_chunk, [(chunk, byte_to_token) for chunk in chunks]),
                total=len(chunks),
                desc="Initial tokenization"
            ))
        
        # Combine results
        ids = array.array('I')
        for result in chunk_results:
            ids.extend(result)
                
        print(f"\nBase vocabulary size: {self.base_vocab_size}")
        print(f"Initial sequence length: {len(ids)}")
        
        # Keep training until we reach the target vocab size
        target_vocab_size = self.vocab_size
        pbar = tqdm(total=target_vocab_size - self.base_vocab_size, desc="Training BPE")
        last_vocab_size = len(self.vocab)
        
        while len(self.vocab) < target_vocab_size:
            stats = self.get_stats(ids)
            if not stats:
                print("No more pairs to merge.")
                break
                
            pair = max(stats, key=stats.get)
            idx = len(self.vocab)
            ids = self.merge(ids, pair, idx)
            
            # Only update progress when vocabulary actually grows
            if len(self.vocab) > last_vocab_size:
                pbar.update(len(self.vocab) - last_vocab_size)
                last_vocab_size = len(self.vocab)
            
            # Add the merged token to the vocabulary
            if pair not in self.merges:  # Ensure we don't overwrite existing merges
                self.merges[pair] = idx
                self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
            
            # Print progress periodically
            if len(self.vocab) % 100 == 0:
                try:
                    text0 = self.vocab[pair[0]].decode('utf-8')
                    text1 = self.vocab[pair[1]].decode('utf-8')
                    merged = self.vocab[idx].decode('utf-8')
                    print(f"\nVocab size: {len(self.vocab)}: {text0} + {text1} = {merged}")
                except UnicodeDecodeError:
                    continue
        
        pbar.close()
        print("\nFinal statistics:")
        print(f"Final vocabulary size: {len(self.vocab):,}")
        print(f"Number of merges: {len(self.merges):,}")
        print(f"Final compression ratio: {original_length / len(ids):.2f}x")

    def encode(self, text):
        """Encode text to token IDs."""
        final_tokens = []
        i = 0
        text_bytes = text.encode('utf-8')
        
        while i < len(text_bytes):
            # If we're at a leading space, encode it separately
            if text_bytes[i] == 32:  # ASCII space
                final_tokens.append(32)  # Space token
                i += 1
                continue
            
            # Try to find the longest matching sequence (including potential trailing spaces)
            longest_match = None
            longest_length = 0
            matched_token = None
            
            # Sort vocab items by length (longest first)
            for token_id, token_bytes in sorted(self.vocab.items(), 
                                              key=lambda x: len(x[1]), 
                                              reverse=True):
                if (i + len(token_bytes) <= len(text_bytes) and 
                    text_bytes[i:i+len(token_bytes)] == token_bytes):
                    longest_length = len(token_bytes)
                    longest_match = token_bytes
                    matched_token = token_id
                    break
            
            if longest_match:
                final_tokens.append(matched_token)
                i += longest_length
            else:
                # If no match found, fall back to single byte
                for token_id, token_bytes in self.vocab.items():
                    if token_bytes == bytes([text_bytes[i]]):
                        final_tokens.append(token_id)
                        break
                i += 1
        
        return final_tokens

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
        try:
            # Load the local parquet file
            dataset = pd.read_parquet('telugu_news_dataset.parquet')
            
            print("Preparing training text...")
            training_text = []
            
            for _, row in tqdm(dataset.iterrows(), desc="Loading documents", total=len(dataset)):
                if not pd.isna(row["headline"]): training_text.append(row["headline"])
                if not pd.isna(row["article"]): training_text.append(row["article"])
                
                if self.sample_size and len(training_text) >= self.sample_size:
                    print(f"Using first {self.sample_size} documents for training")
                    break
            
            full_text = "\n".join(training_text)
            print(f"\nTraining on {len(training_text)} documents...")
            print(f"Total characters in training data: {len(full_text):,}")
            
            start_time = time.time()
            self.fit(full_text)
            print(f"Training time: {time.time() - start_time:.2f} seconds")
            
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            print("Falling back to sample text...")
            sample_text = """
            తెలుగు భాష దక్షిణ భారతదేశంలోని ద్రావిడ భాషల్లో ఒకటి.
            ఆంధ్ర ప్రదేశ్ మరియు తెలంగాణ రాష్ట్రాల అధికార భాష.
            """
            self.fit(sample_text)


if __name__ == "__main__":
    # For quick testing, use a small sample
    tokenizer = BPETokenizer(vocab_size=4999, sample_size=None)
    
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
