---
title: Telugu Tokenizer App
emoji: అ
colorFrom: indigo
colorTo: blue
sdk: docker
sdk_version: "1.0"
app_file: app:app
pinned: false
description: A tokenizer app for tokenizing Telugu text. It uses BPE (Byte Pair Encoding) to tokenize Telugu text. 5k is the vocab size.
tags:
  - telugu
  - tokenizer
  - NLP
  - transformers
license: apache-2.0
model: telugu-tokenizer-model
datasets:
  - telugu-dataset
isPrivate: false
---

# Telugu Tokenizer

This repository provides a tokenizer implementation for processing Telugu text, designed to handle both Telugu Unicode characters and ASCII characters. It uses a Byte Pair Encoding (BPE) approach to efficiently tokenize text and create a vocabulary optimized for Telugu language processing.

## Features

- **Comprehensive Telugu Support**: Includes all Telugu Unicode characters (0C00-0C7F), common ligatures, and valid consonant combinations.
- **Base Vocabulary Creation**: Generates a base vocabulary containing ASCII, Extended ASCII, and Telugu characters.
- **Byte Pair Encoding (BPE)**: Trains the tokenizer to merge frequently occurring token pairs, creating an optimized vocabulary.
- **Parallel Processing**: Utilizes multiprocessing for efficient tokenization of large text datasets.
- **Persistence**: Supports saving and loading the vocabulary to/from JSON files.

## Requirements

The tokenizer requires the following dependencies:

- Python 3.7+
- tqdm
- pandas
- datasets

Install the required packages using pip:
```bash
pip install tqdm pandas datasets
```

## Usage

### 1. Base Vocabulary Creation

The tokenizer first generates a base vocabulary containing ASCII, Extended ASCII, and Telugu characters.

```python
from telugu_tokenizer import create_base_vocab, save_base_vocab

base_vocab = create_base_vocab()
save_base_vocab(base_vocab, path='telugu_base_vocab.json')
```

### 2. Loading an Existing Vocabulary

You can load an existing base vocabulary from a JSON file:

```python
from telugu_tokenizer import load_base_vocab

vocab = load_base_vocab('telugu_base_vocab.json')
```

### 3. Training the Tokenizer

The `BPETokenizer` class can be used to train a tokenizer on a given text input:

```python
from telugu_tokenizer import BPETokenizer

text = "మీరు ఎలా ఉన్నారు?"  # Sample Telugu text
tokenizer = BPETokenizer(vocab_size=5000)
tokenizer.fit(text)
```

### 4. Saving and Loading the Tokenizer

After training, save the tokenizer's vocabulary and merges:

```python
tokenizer.save('telugu_tokenizer')
```

Load the trained tokenizer:

```python
tokenizer.load('telugu_tokenizer')
```

## Telugu Unicode Support

The tokenizer covers the full range of Telugu Unicode characters, including vowels, consonants, vowel signs, digits, and fraction symbols. Additionally, it supports:

- Common ligatures formed with Telugu consonants and vowel signs.
- Valid consonant combinations like `క్క`, `క్జ`, etc.

## File Structure

- **`bpe_tokenizer.py`**: Contains the implementation of the Telugu tokenizer.
- **`telugu_base_vocab.json`**: JSON file storing the base vocabulary.
- **`telugu_tokenizer_vocab.json`**: JSON file storing the trained vocabulary and merges (generated after training).

## Results

- **Final vocabulary size**: 4,999
- **Final compression ratio**: 8.63x

## Logs
- [View Training Logs ](./training_logs.log)

## Performance

The tokenizer uses multiprocessing to handle large datasets efficiently. It processes text in chunks and merges token pairs iteratively to optimize the vocabulary size. This is a simple implementation and can be improved for large-scale datasets.
## Future Enhancements

- Extend support for additional Telugu ligatures and symbols.
- Optimize BPE training for large-scale datasets.
- Provide pre-trained models for common Telugu NLP tasks.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Contributing

Contributions are welcome! Feel free to submit a pull request or open an issue if you encounter bugs or have suggestions for improvement.

## Acknowledgments

- Unicode Consortium for Telugu Unicode character information.
- Community contributions to Telugu NLP development.

---

Feel free to explore the tokenizer and adapt it for your Telugu language processing needs. Happy coding!

