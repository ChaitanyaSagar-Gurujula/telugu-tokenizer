from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from bpe_tokenizer import BPETokenizer, create_base_vocab
import os
import json

# Get the absolute path to the templates directory
TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")

app = FastAPI(title="Telugu BPE Tokenizer")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Templates with absolute path
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Initialize tokenizer
tokenizer = BPETokenizer(vocab_size=5000)

# Load the vocabulary file directly
print("Loading vocabulary...")
vocab_file = 'telugu_tokenizer_vocab.json'
with open(vocab_file, 'r', encoding='utf-8') as f:
    vocab_data = json.load(f)

class TokenizeRequest(BaseModel):
    text: str

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html", 
        {"request": request, "title": "Telugu BPE Tokenizer"}
    )

@app.post("/tokenize")
async def tokenize(request: TokenizeRequest):
    text = request.text
    try:
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        
        # Get token details from vocabulary for display
        token_details = []
        current_position = 0
        current_byte_position = 0
        text_bytes = text.encode('utf-8')
        
        while current_position < len(tokens):
            # Skip leading spaces in original text
            while current_byte_position < len(text_bytes) and text_bytes[current_byte_position] == 32:
                current_byte_position += 1
            
            # Get next word from original text
            word_start = current_byte_position
            word_end = word_start
            while word_end < len(text_bytes) and text_bytes[word_end] != 32:
                word_end += 1
            
            word_bytes = text_bytes[word_start:word_end]
            word = word_bytes.decode('utf-8')
            
            # Collect tokens for this word
            word_tokens = []
            decoded_bytes = b''
            
            while current_position < len(tokens):
                token = tokens[current_position]
                token_bytes = tokenizer.vocab[token]
                
                # If we've collected enough bytes for the word (plus possible space)
                if len(decoded_bytes) >= len(word_bytes):
                    break
                
                word_tokens.append(token)
                decoded_bytes += token_bytes
                current_position += 1
            
            # Update byte position for next word
            current_byte_position = word_end
            
            # Add word and its tokens to details
            if len(word_tokens) == 1:
                # Complete word case
                token_id = word_tokens[0]
                token_details.append({
                    "word": word,
                    "type": "complete_word",
                    "token_id": token_id,
                    "text": vocab_data.get(str(token_id), {}).get('text', '[UNKNOWN]')
                })
            else:
                # Subword tokens case
                token_details.append({
                    "word": word,
                    "type": "subword_tokens",
                    "tokens": [{
                        "id": t,
                        "text": vocab_data.get(str(t), {}).get('text', '[UNKNOWN]')
                    } for t in word_tokens]
                })
        
        return {
            "original": text,
            "tokens": tokens,
            "token_details": token_details,
            "decoded": decoded,
            "matches": text == decoded
        }
    except Exception as e:
        print(f"Error: {str(e)}")
        return {"error": str(e)}

@app.get("/vocab")
async def get_vocab():
    return {
        "vocab_size": len(vocab_data),
        "base_vocab_size": sum(1 for info in vocab_data.values() if info.get('is_base', False)),
        "num_merges": len(getattr(tokenizer, 'merges', {}))
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001) 