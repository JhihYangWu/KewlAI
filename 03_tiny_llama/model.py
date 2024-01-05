# inspired by https://github.com/facebookresearch/llama/blob/main/llama/model.py and
#             https://github.com/hkproj/pytorch-llama/blob/main/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional
import math

@dataclass
class Config:
    vocab_size: int = 1000
    dim: int = 256  # what len vector to represent each token with
    n_layers: int = 4
    norm_eps: float = 1e-5  # epsilon used in RMSNorm
    context_length: int = 128
    n_heads: int = 4  # multi-head attention

class TinyLlama(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.token_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.freqs_cis = self.precompute_freqs_cis()
        self.mode = None
        self.output_norm = nn.Parameter(torch.ones(config.dim))
        self.output_layer = nn.Linear(config.dim, config.vocab_size, bias=False)
        for i in range(config.n_layers):
            setattr(self, f"layer_{i}_attn_q", nn.Linear(config.dim, config.dim, bias=False))
            setattr(self, f"layer_{i}_attn_k", nn.Linear(config.dim, config.dim, bias=False))
            setattr(self, f"layer_{i}_attn_v", nn.Linear(config.dim, config.dim, bias=False))
            setattr(self, f"layer_{i}_attn_o", nn.Linear(config.dim, config.dim, bias=False))
            setattr(self, f"layer_{i}_mlp_gate", nn.Linear(config.dim, int(config.dim * 2.5), bias=False))
            setattr(self, f"layer_{i}_mlp_up", nn.Linear(config.dim, int(config.dim * 2.5), bias=False))
            setattr(self, f"layer_{i}_mlp_down", nn.Linear(int(config.dim * 2.5), config.dim, bias=False))
            setattr(self, f"layer_{i}_input_norm", nn.Parameter(torch.ones(config.dim)))
            setattr(self, f"layer_{i}_post_attn_norm", nn.Parameter(torch.ones(config.dim)))
        self.cache_k = torch.zeros((config.n_layers, 1, config.context_length, config.n_heads, config.dim // config.n_heads))
        self.cache_v = torch.zeros((config.n_layers, 1, config.context_length, config.n_heads, config.dim // config.n_heads))

    def forward(self, token_ids: torch.Tensor, start_pos: int) -> torch.Tensor:
        assert self.mode == "train" or self.mode == "eval"
        batch_size, seq_len = token_ids.shape
        h = self.token_embeddings(token_ids)
        freqs_cis = self.freqs_cis[:, start_pos: start_pos + seq_len]

        if self.mode == "eval":
            assert seq_len == 1, "only one token id at a time in inference mode"
            assert batch_size == 1, "only batch_size=1 in inference mode"
        mask = None
        if self.mode == "train" and seq_len > 1:  # need to create mask to prevent tokens from talking to future tokens
            # Andrej Karpathy explains it well in https://www.youtube.com/watch?v=kCc8FmEb1nY
            mask = torch.full((seq_len, seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            mask = torch.nan_to_num(mask)  # need this line on apple silicon

        for layer_num in range(self.config.n_layers):
            h = self.layer(h, start_pos, freqs_cis, mask, layer_num)
        h = self.norm(h, self.output_norm)
        logits = self.output_layer(h)
        return logits
    
    def norm(self, x: torch.Tensor, weight: nn.Parameter) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.config.norm_eps)
        x = (x * weight).type(orig_dtype)
        return x

    def precompute_freqs_cis(self) -> torch.Tensor:
        dim = self.config.dim // self.config.n_heads
        end = self.config.context_length
        theta = 10000.0  # from paper
        # copied from https://github.com/tinygrad/tinygrad/blob/master/extra/models/llama.py
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)] / dim))
        freqs = torch.arange(end).unsqueeze(dim=1)*freqs.unsqueeze(dim=0)
        return torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1).reshape(1, end, 1, dim//2, 2)
    
    def layer(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], layer_num: int) -> torch.Tensor:
        # first half of block
        x = x + self.attention(self.norm(x, getattr(self, f"layer_{layer_num}_input_norm")), start_pos, freqs_cis, mask, layer_num)  # (batch_size, seq_len, dim)
        # second half of block
        x = x + self.feed_forward(self.norm(x, getattr(self, f"layer_{layer_num}_post_attn_norm")), layer_num)  # (batch_size, seq_len, dim)
        return x

    def attention(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], layer_num: int) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape
        wq = getattr(self, f"layer_{layer_num}_attn_q")
        wk = getattr(self, f"layer_{layer_num}_attn_k")
        wv = getattr(self, f"layer_{layer_num}_attn_v")
        wo = getattr(self, f"layer_{layer_num}_attn_o")
        xq, xk, xv = wq(x), wk(x), wv(x)
        head_dim = dim // self.config.n_heads
        xq = xq.view(batch_size, seq_len, self.config.n_heads, head_dim)
        xk = xk.view(batch_size, seq_len, self.config.n_heads, head_dim)
        xv = xv.view(batch_size, seq_len, self.config.n_heads, head_dim)
        xq, xk = self.apply_rotary_emb(xq, xk, freqs_cis)
        # kv cache
        if self.mode == "eval":
            self.cache_k[layer_num, :1, start_pos: start_pos + seq_len] = xk
            self.cache_v[layer_num, :1, start_pos: start_pos + seq_len] = xv
            keys = self.cache_k[layer_num, :1, :start_pos + seq_len]
            values = self.cache_v[layer_num, :1, :start_pos + seq_len]
        else:
            keys = xk
            values = xv
        xq = xq.transpose(1, 2)  # (batch_size, n_heads, seq_len, head_dim)
        keys = keys.transpose(1, 2)  # (batch_size, n_heads, seq_len, head_dim)
        values = values.transpose(1, 2)  # (batch_size, n_heads, seq_len, head_dim)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(head_dim)  # (batch_size, n_heads, seq_len, seq_len)
        if mask is not None:
            scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (batch_size, n_heads, seq_len, head_dim)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)
        return wo(output)

    def feed_forward(self, x: torch.Tensor, layer_num: int) -> torch.Tensor:
        gate = getattr(self, f"layer_{layer_num}_mlp_gate")
        up = getattr(self, f"layer_{layer_num}_mlp_up")
        down = getattr(self, f"layer_{layer_num}_mlp_down")
        return down(F.silu(gate(x)) * up(x))

    def apply_rotary_emb(self, xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        # copied from https://github.com/tinygrad/tinygrad/blob/master/extra/models/llama.py
        def complex_mult(A, c, d):
            a,b = A[..., 0:1], A[..., 1:2]
            ro = a*c - b*d
            co = a*d + b*c
            return torch.cat((ro, co), dim=-1)
        assert freqs_cis.shape[1] == xq.shape[1] == xk.shape[1], f"freqs_cis shape mismatch {freqs_cis.shape} xq:{xq.shape} xk:{xk.shape}"
        xq = xq.reshape(*xq.shape[0:-1], -1, 2)
        xk = xk.reshape(*xk.shape[0:-1], -1, 2)
        c, d = freqs_cis[..., 0:1], freqs_cis[..., 1:2]
        xq_out = complex_mult(xq, c, d)
        xk_out = complex_mult(xk, c, d)
        return xq_out.flatten(3), xk_out.flatten(3)
