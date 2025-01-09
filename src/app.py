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
        # Use the BPE tokenizer's encode method
        tokens = tokenizer.encode(text)
        
        # Use the BPE tokenizer's decode method
        decoded = tokenizer.decode(tokens)
        
        # Get token details from vocabulary for display
        token_details = []
        words = text.split()
        current_position = 0
        
        for word in words:
            # Find tokens that make up this word
            word_tokens = []
            word_length = len(word.encode('utf-8'))
            while current_position < len(tokens) and len(''.join(
                vocab_data.get(str(t), {}).get('text', '') 
                for t in word_tokens
            ).encode('utf-8')) < word_length:
                word_tokens.append(tokens[current_position])
                current_position += 1
            
            token_details.append({
                "word": word,
                "type": "subword_tokens",
                "tokens": [
                    {
                        "id": t,
                        "text": vocab_data.get(str(t), {}).get('text', '[UNKNOWN]')
                    } for t in word_tokens
                ]
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