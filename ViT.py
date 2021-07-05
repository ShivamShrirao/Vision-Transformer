import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from collections import OrderedDict


class TransformerEncoderLayer(nn.Module):
    def __init__(self, emb_dim, nheads, dim_feedforward, dp_rate=0.1):
        super().__init__()
        self.mhsa = nn.MultiheadAttention(emb_dim, nheads)
        self.ln1 = nn.LayerNorm(emb_dim)
        self.mlp = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(emb_dim, dim_feedforward)),
            ('act', nn.GELU()),
            ('drop', nn.Dropout(dp_rate)),
            ('fc2', nn.Linear(dim_feedforward, emb_dim))
        ]))
        self.ln2 = nn.LayerNorm(emb_dim)
        self.dropout1 = nn.Dropout(dp_rate)
        self.dropout2 = nn.Dropout(dp_rate)

    def attention(self, x):
        return self.mhsa(x, x, x, need_weights=False)[0]

    def forward(self, x):       # [seq_len, B, emb_dim]
        x = x + self.dropout1(self.attention(self.ln1(x)))
        x = x + self.dropout2(self.mlp(self.ln2(x)))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, emb_dim, nheads, dim_feedforward, num_layers):
        super().__init__()
        self.blocks = nn.Sequential(*[TransformerEncoderLayer(emb_dim, nheads, dim_feedforward) for _ in range(num_layers)])

    def forward(self, x):       # [seq_len, B, emb_dim]
        return self.blocks(x)


class VisionTransformer(nn.Module):
    def __init__(self, transformer, img_size=224, patch_size=16, emb_dim=512, num_classes=100, representation_dim=None, dp_rate=0.1, reset_params=True):
        super().__init__()
        self.proj_patch = nn.Conv2d(3, emb_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        
        scale = emb_dim ** -0.5
        self.cls_token = nn.Parameter(scale * torch.randn(emb_dim))
        seq_len = (img_size//patch_size) ** 2
        self.pos_emb = nn.Parameter(scale * torch.randn(seq_len+1, 1, emb_dim))
        self.dropout = nn.Dropout(dp_rate)

        self.transformer = transformer
        self.post_norm = nn.LayerNorm(emb_dim)
        
        nfeatures = emb_dim
        if representation_dim is not None:
            nfeatures = representation_dim
            self.pre_logits = nn.Sequential(nn.Linear(emb_dim, representation_dim), nn.Tanh())
        else:
            self.pre_logits = nn.Identity()
        self.head = nn.Linear(nfeatures, num_classes)
        
        if reset_params: self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward_features(self, x):
        # [B, C, H, W]
        x = self.proj_patch(x)      # [B, emb_dim, H//patch, W//patch]
        x = x.flatten(-2)           # [B, emb_dim, seq_len]
        x = x.permute(2,0,1)        # [seq_len, B, emb_dim]

        cls_token = self.cls_token.expand(1, x.shape[1], -1)    # [1, B, emb_dim]
        x = torch.cat((cls_token,x), dim=0)     # [seq_len+1, B, emb_dim]
        x = self.dropout(x + self.pos_emb)      # [seq_len+1, B, emb_dim]
        x = self.transformer(x)                 # [seq_len+1, B, emb_dim]

        x = self.post_norm(x[0])    # [B, emb_dim] Take the output at cls_token.
        x = self.pre_logits(x)      # [B, nfeatures]
        return x

    def forward(self, x):
        x = self.forward_features(x)# [B, nfeatures]
        x = self.head(x)            # [B, classes]
        return x