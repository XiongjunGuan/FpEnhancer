'''
Description: from https://github.com/dome272/VQGAN-pytorch
Author: Xiongjun Guan
Date: 2024-12-05 17:12:49
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2024-12-20 11:14:57

Copyright (C) 2024 by Xiongjun Guan, Tsinghua University. All rights reserved.
'''
import math

import torch
import torch.nn as nn


class Codebook_choose(nn.Module):

    def __init__(self,
                 num_codebook_vectors=1024,
                 latent_dim=256,
                 codebook_path=None):
        super(Codebook_choose, self).__init__()
        self.num_codebook_vectors = num_codebook_vectors
        self.latent_dim = latent_dim

        self.embedding = nn.Embedding(self.num_codebook_vectors,
                                      self.latent_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_codebook_vectors,
                                            1.0 / self.num_codebook_vectors)

        self.transformer_encoder = TransformerCla(
            num_classes=num_codebook_vectors,
            latent_dim=latent_dim,
            d_model=latent_dim,
            nhead=8,
            num_encoder_layers=6)

        if codebook_path is not None:
            self.embedding.load_state_dict(torch.load(codebook_path))
            for param in self.embedding.parameters():
                param.requires_grad = False

    def forward(self, z):
        B, C, H, W = z.shape
        # Apply Transformer encoder
        z_transformed = self.transformer_encoder(z)  # (B, H*W, num)
        probabilities = z_transformed.reshape(
            -1, self.num_codebook_vectors)  # (B*H*W, num)
        # Use softmax to get probabilities and select the codebook vector
        min_encoding_indices = torch.argmax(probabilities, dim=-1)
        z_q = self.embedding(min_encoding_indices).reshape(
            B, H, W, C)  # (B*H*W, C) -> (B,H,W,C)
        z_q = z_q.permute(0, 3, 1, 2)  # (B,C,H,W)
        z_out = z + (z_q - z).detach()

        return z_out, z_q, z, min_encoding_indices, probabilities


class PositionEncodingSine(nn.Module):
    """
    This is a sinusoidal position encoding that generalized to 2-dimensional images
    """

    def __init__(self, d_model, max_shape=(1024, 1024), temp_bug_fix=True):
        """
        Args:
            max_shape (tuple): for 1/8 featmap, the max length of 256 corresponds to 2048 pixels
            temp_bug_fix (bool): As noted in this [issue](https://github.com/zju3dv/LoFTR/issues/41),
                the original implementation of LoFTR includes a bug in the pos-enc impl, which has little impact
                on the final performance. For now, we keep both impls for backward compatability.
                We will remove the buggy impl after re-training all variants of our released models.
        """
        super().__init__()

        pe = torch.zeros((d_model, *max_shape))
        y_position = torch.ones(max_shape).cumsum(0).float().unsqueeze(0)
        x_position = torch.ones(max_shape).cumsum(1).float().unsqueeze(0)
        if temp_bug_fix:
            div_term = torch.exp(
                torch.arange(0, d_model // 2, 2).float() *
                (-math.log(10000.0) / (d_model // 2)))
        else:  # a buggy implementation (for backward compatability only)
            div_term = torch.exp(
                torch.arange(0, d_model // 2, 2).float() *
                (-math.log(10000.0) / d_model // 2))
        div_term = div_term[:, None, None]  # [C//4, 1, 1]
        pe[0::4, :, :] = torch.sin(x_position * div_term)
        pe[1::4, :, :] = torch.cos(x_position * div_term)
        pe[2::4, :, :] = torch.sin(y_position * div_term)
        pe[3::4, :, :] = torch.cos(y_position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0),
                             persistent=False)  # [1, C, H, W]

    def forward(self, x):
        """
        Args:
            x: [N, C, H, W]
        """
        return x + self.pe[:, :, :x.size(2), :x.size(3)]


class Encoder(nn.Module):

    def __init__(self,
                 num_heads: int = 12,
                 embedding_dim: int = 768,
                 mlp_dim: int = 3072,
                 dropout: float = 0,
                 attention_dropout: float = 0):
        super(Encoder, self).__init__()
        self.num_heads = num_heads

        # MSA
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.attention = nn.MultiheadAttention(embedding_dim,
                                               num_heads,
                                               dropout=attention_dropout,
                                               batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # MLP
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, mlp_dim)
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(mlp_dim, embedding_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        """
        :param x: (B, seq_len, C)
        :return: (B, seq_len, C)
        """
        o = self.norm1(x)
        o, _ = self.attention(o, o, o, need_weights=False)
        o = self.dropout(o)
        o = o + x
        y = self.norm2(o)
        y = self.linear1(y)
        y = self.gelu(y)
        y = self.dropout1(y)
        y = self.linear2(y)
        y = self.dropout2(y)
        return y + o


class TransformerCla(nn.Module):

    def __init__(self,
                 num_classes=1024,
                 latent_dim=256,
                 d_model=512,
                 nhead=8,
                 num_encoder_layers=6,
                 dim_feedforward=2048,
                 dropout=0.1):
        super(TransformerCla, self).__init__()
        self.d_model = d_model
        self.positional_encoding = PositionEncodingSine(d_model=latent_dim)

        self.encoders = nn.Sequential()
        for _ in range(num_encoder_layers):
            self.encoders.append(
                Encoder(nhead, d_model, dim_feedforward, dropout))

        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.Tanh(),
                                  nn.Linear(d_model, num_classes))

    def forward(self, src):
        B, C, H, W = src.shape
        src = self.positional_encoding(src)  # Add positional encoding

        src = src.permute(0, 2, 3, 1).contiguous().reshape(B, H * W,
                                                           C)  # (B,L,C)

        src = self.encoders(src)

        src = self.head(src)  # (B, L, num_classes)

        return src


class Codebook_frosz(nn.Module):

    def __init__(self,
                 num_codebook_vectors=1024,
                 latent_dim=256,
                 codebook_path=None):
        super(Codebook_frosz, self).__init__()
        self.num_codebook_vectors = num_codebook_vectors
        self.latent_dim = latent_dim

        self.embedding = nn.Embedding(self.num_codebook_vectors,
                                      self.latent_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_codebook_vectors,
                                            1.0 / self.num_codebook_vectors)

        if codebook_path is not None:
            self.embedding.load_state_dict(torch.load(codebook_path))
            for param in self.embedding.parameters():
                param.requires_grad = False

    def forward(self, z):
        z = z.permute(0, 2, 3, 1).contiguous()  # (B,H,W,C)
        z_flattened = z.view(-1, self.latent_dim)

        d = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - \
            2*(torch.matmul(z_flattened, self.embedding.weight.t()))

        min_encoding_indices = torch.argmin(d, dim=1)

        z_q = self.embedding(min_encoding_indices).view(z.shape)  # (B,H,W,C)

        z_q = z_q.permute(0, 3, 1, 2)  # (B,C,H,W)

        return min_encoding_indices, z_q


class Codebook_clean(nn.Module):

    def __init__(self, num_codebook_vectors=1024, latent_dim=256):
        super(Codebook_clean, self).__init__()
        self.num_codebook_vectors = num_codebook_vectors
        self.latent_dim = latent_dim

        self.embedding = nn.Embedding(self.num_codebook_vectors,
                                      self.latent_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_codebook_vectors,
                                            1.0 / self.num_codebook_vectors)

    def forward(self, z):
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.latent_dim)

        d = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - \
            2*(torch.matmul(z_flattened, self.embedding.weight.t()))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        z_out = z + (z_q - z).detach()

        z_out = z_out.permute(0, 3, 1, 2)

        return z_out, z_q, z


class Codebook(nn.Module):

    def __init__(self, num_codebook_vectors=1024, latent_dim=256, beta=0.25):
        super(Codebook, self).__init__()
        self.num_codebook_vectors = num_codebook_vectors
        self.latent_dim = latent_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.num_codebook_vectors,
                                      self.latent_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_codebook_vectors,
                                            1.0 / self.num_codebook_vectors)

    def forward(self, z):
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.latent_dim)

        d = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - \
            2*(torch.matmul(z_flattened, self.embedding.weight.t()))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        loss = torch.mean((z_q - z.detach())**2) + self.beta * torch.mean(
            (z_q.detach() - z)**2)

        z_q = z + (z_q - z).detach()

        z_q = z_q.permute(0, 3, 1, 2)

        return z_q, min_encoding_indices, loss
