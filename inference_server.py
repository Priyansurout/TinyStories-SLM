"""
Flask server to expose the PyTorch SLM model via a STREAMING API endpoint.
This version sends tokens one by one as they are generated.

To run this server:
1. Ensure 'best_model_params.pt' is in the same directory.
2. Install Flask, PyTorch, and Tiktoken:
   pip install Flask torch tiktoken
3. Run the script:
   python streaming_inference_server.py
"""
from flask import Flask, request, Response
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import tiktoken
import os
import time
from dataclasses import dataclass

# --- 1. MODEL ARCHITECTURE CLASSES (Copied from inference_only.py) ---
# We must include the classes so the torch.load function can reconstruct the model.

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.flash = False 
        if not self.flash:
            pass

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        bias = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        mask = bias[:, :, :T, :T] == 0
        att = att.masked_fill(mask, float('-inf'))

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = LayerNorm(config.n_embd, config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln2 = LayerNorm(config.n_embd, config.bias)
        self.mlp = MLP(config)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float = 0.0
    bias: bool = True

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.zeros_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.zeros_(module.weight)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # We need all logits for streaming
        return logits, None

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        This method is now a GENERATOR, yielding each new token ID.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            yield idx_next # Yield the newly generated token

# --- 2. FLASK SERVER SETUP ---

app = Flask(__name__)
CORS(app)

MODEL = None
ENCODER = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_slm_model():
    """Loads the model weights and tokenizer into memory."""
    global MODEL, ENCODER
    config = GPTConfig(
        vocab_size=50257, block_size=128, n_layer=6,
        n_head=6, n_embd=384, dropout=0.1, bias=True
    )
    best_model_params_path = "best_model_params.pt"
    if not os.path.exists(best_model_params_path):
        print(f"ERROR: Model file '{best_model_params_path}' not found!")
        return False
    MODEL = GPT(config)
    try:
        MODEL.load_state_dict(torch.load(best_model_params_path, map_location=torch.device(DEVICE)))
        MODEL.to(DEVICE)
        MODEL.eval()
        ENCODER = tiktoken.get_encoding("gpt2")
        print(f"SLM Model loaded successfully on {DEVICE}.")
        return True
    except Exception as e:
        print(f"ERROR: Failed to load model state: {e}")
        MODEL = None
        return False

@app.route('/stream', methods=['POST'])
def stream_story():
    """API endpoint for STREAMING generated text."""
    if MODEL is None or ENCODER is None:
        return Response("Model not loaded", status=500, mimetype='text/plain')

    data = request.json
    prompt = data.get('prompt', '')
    max_tokens = data.get('max_tokens', 200)

    if not prompt:
        return Response("No prompt provided", status=400, mimetype='text/plain')
        
    def generate_events():
        try:
            context = (torch.tensor(ENCODER.encode_ordinary(prompt)).unsqueeze(dim=0)).to(DEVICE)
            
            # First, yield the original prompt back so the UI can display it
            yield f"data: {prompt}\n\n"

            # Generate new tokens and stream them
            token_generator = MODEL.generate(context, max_new_tokens=max_tokens, temperature=0.9, top_k=50)
            
            for token_tensor in token_generator:
                # Decode the single token tensor
                token_id = token_tensor.squeeze().tolist()
                decoded_token = ENCODER.decode([token_id])
                
                # Format as a Server-Sent Event (SSE)
                yield f"data: {decoded_token}\n\n"
                time.sleep(0.05) # Increased delay to slow down streaming
        
        except Exception as e:
            print(f"Streaming runtime error: {e}")
            # Optionally, you can stream an error message to the client
            yield f"data: [ERROR: {str(e)}]\n\n"
        finally:
            # Signal the end of the stream
            yield "data: [DONE]\n\n"


    return Response(generate_events(), mimetype='text/event-stream')

# Server Initialization
if __name__ == '__main__':
    if load_slm_model():
        print("Starting Flask server for STREAMING on http://127.0.0.1:5000")
        app.run(debug=True, port=5000, threaded=True) # Threaded is good for streaming
    else:
        print("Server failed to start due to model loading error.")

