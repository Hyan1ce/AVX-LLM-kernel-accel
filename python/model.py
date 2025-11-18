import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from python import gemm, layernorm, softmax

class AVXLinear(nn.Module):
    """Linear layer using AVX-optimized GEMM."""
    def __init__(self, in_features, out_features, bias=True, use_avx=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_avx = use_avx
        
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, input):
        if self.use_avx:
            output = gemm(input, self.weight.t(), use_parallel=True)
            if self.bias is not None:
                output = output + self.bias.unsqueeze(0)
            return output
        else:
            return nn.functional.linear(input, self.weight, self.bias)


class AVXLayerNorm(nn.Module):
    """LayerNorm using AVX-optimized implementation."""
    def __init__(self, hidden_size, eps=1e-5, use_avx=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.use_avx = use_avx
        
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
    
    def forward(self, input):
        if self.use_avx:
            output, mean, var = layernorm(input, self.gamma, self.beta, 
                                         self.eps, use_parallel=True)
            return output
        else:
            mean = input.mean(dim=1, keepdim=True)
            var = input.var(dim=1, keepdim=True, unbiased=False)
            output = (input - mean) / torch.sqrt(var + self.eps)
            return output * self.gamma.unsqueeze(0) + self.beta.unsqueeze(0)


class AVXSoftmax(nn.Module):
    """Softmax using AVX-optimized implementation."""
    def __init__(self, dim=1, use_avx=True):
        super().__init__()
        self.dim = dim
        self.use_avx = use_avx
    
    def forward(self, input):
        if self.use_avx and self.dim == 1:
            return softmax(input, use_parallel=True)
        else:
            return torch.nn.functional.softmax(input, dim=self.dim)


class SimpleTransformerBlock(nn.Module):
    """A simple transformer block with AVX-optimized kernels."""
    def __init__(self, d_model=512, nhead=8, dim_feedforward=2048, 
                 dropout=0.1, use_avx=True):
        super().__init__()
        self.use_avx = use_avx
        
        # Self-attention
        self.self_attn_q = AVXLinear(d_model, d_model, use_avx=use_avx)
        self.self_attn_k = AVXLinear(d_model, d_model, use_avx=use_avx)
        self.self_attn_v = AVXLinear(d_model, d_model, use_avx=use_avx)
        self.self_attn_out = AVXLinear(d_model, d_model, use_avx=use_avx)
        self.softmax = AVXSoftmax(dim=-1, use_avx=use_avx)
        
        # Feedforward
        self.ffn1 = AVXLinear(d_model, dim_feedforward, use_avx=use_avx)
        self.ffn2 = AVXLinear(dim_feedforward, d_model, use_avx=use_avx)
        
        # LayerNorm
        self.norm1 = AVXLayerNorm(d_model, use_avx=use_avx)
        self.norm2 = AVXLayerNorm(d_model, use_avx=use_avx)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, x):
        # Self-attention
        residual = x
        q = self.self_attn_q(x)
        k = self.self_attn_k(x)
        v = self.self_attn_v(x)
        
        # Scaled dot-product attention (simplified)
        scores = torch.bmm(q.unsqueeze(1), k.unsqueeze(1).transpose(1, 2))
        scores = scores / (x.size(-1) ** 0.5)
        attn = self.softmax(scores.squeeze(1))
        attn = torch.bmm(attn.unsqueeze(1), v.unsqueeze(1)).squeeze(1)
        x = self.self_attn_out(attn)
        x = self.dropout(x)
        x = self.norm1(x + residual)
        
        # Feedforward
        residual = x
        x = self.ffn1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.ffn2(x)
        x = self.dropout(x)
        x = self.norm2(x + residual)
        
        return x


class SimpleTransformer(nn.Module):
    """A simple transformer model for language modeling."""
    def __init__(self, vocab_size=10000, d_model=512, nhead=8, 
                 num_layers=6, dim_feedforward=2048, max_seq_len=512,
                 dropout=0.1, use_avx=True):
        super().__init__()
        self.d_model = d_model
        self.use_avx = use_avx
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, d_model))
        
        self.layers = nn.ModuleList([
            SimpleTransformerBlock(d_model, nhead, dim_feedforward, 
                                 dropout, use_avx)
            for _ in range(num_layers)
        ])
        
        self.norm = AVXLayerNorm(d_model, use_avx=use_avx)
        self.lm_head = AVXLinear(d_model, vocab_size, use_avx=use_avx)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input_ids):
        # Embedding
        x = self.embedding(input_ids)
        seq_len = x.size(1)
        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x)
        
        # Transformer blocks
        for layer in self.layers:
            x = layer(x)
        
        # Final norm and output
        x = self.norm(x)
        logits = self.lm_head(x)
        
        return logits

