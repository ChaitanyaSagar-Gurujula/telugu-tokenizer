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
        # Split by spaces to check each word
        words = text.split()
        final_tokens = []
        token_details = []
        
        for word in words:
            # First check if the word exists in vocabulary
            word_found = False
            for token_id, info in vocab_data.items():
                if info.get('text') == word:
                    print(f"Found word '{word}' as token {token_id}")
                    final_tokens.append(int(token_id))
                    token_details.append({
                        "word": word,
                        "token_id": int(token_id),
                        "type": "complete_word",
                        "text": word
                    })
                    word_found = True
                    break
            
            if not word_found:
                print(f"Word '{word}' not found in vocabulary, using subword tokens")
                # Only use character-level tokenization if word isn't in vocabulary
                word_tokens = tokenizer.encode(word)
                final_tokens.extend(word_tokens)
                token_details.append({
                    "word": word,
                    "tokens": [
                        {
                            "id": t,
                            "text": vocab_data.get(str(t), {}).get('text', '[UNKNOWN]')
                        } for t in word_tokens
                    ],
                    "type": "subword_tokens"
                })
        
        try:
            decoded = tokenizer.decode(final_tokens)
        except Exception as e:
            print(f"Decoding error: {str(e)}")
            # Fall back to joining the original words
            decoded = " ".join(words)
        
        return {
            "original": text,
            "tokens": final_tokens,
            "token_details": token_details,
            "decoded": decoded,
            "matches": text == decoded
        }
    except Exception as e:
        print(f"Processing error: {str(e)}")
        return {
            "error": str(e),
            "original": text,
            "tokens": [],
            "token_details": [],
            "decoded": text,
            "matches": False
        }

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