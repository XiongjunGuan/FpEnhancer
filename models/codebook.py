'''
Description: from https://github.com/dome272/VQGAN-pytorch
Author: Xiongjun Guan
Date: 2024-12-05 17:12:49
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2024-12-09 17:06:01

Copyright (C) 2024 by Xiongjun Guan, Tsinghua University. All rights reserved.
'''
import torch
import torch.nn as nn


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
