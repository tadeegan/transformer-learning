
import torch
# import numpy as np
from torch import nn
import math
import mnist
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from typing import List

class MLP(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels: List[int], activation='relu') -> None:
        super(MLP, self).__init__()
        self.layers = torch.nn.ModuleList()
        for i, c in enumerate(hidden_channels):
            in_features = input_channels if i == 0 else hidden_channels[i-1]
            self.layers.append(torch.nn.Linear(in_features=in_features, out_features=c))
        if activation == 'relu':
            self.activation_fn = F.relu
        if activation == 'gelu':
            self.activation_fn = F.gelu
        if activation == None:
            self.activation_fn = lambda x: x
    
    def __call__(self, input: torch.Tensor):
        x = input
        for l in self.layers:
            x = self.activation_fn(l(x))
        return x


class TransformerEncoderBlock(torch.nn.Module):

    def __init__(self, input_dim) -> None:
        super(TransformerEncoderBlock, self).__init__()

        self.norm1 = torch.nn.LayerNorm(normalized_shape=input_dim)
        self.norm2 = torch.nn.LayerNorm(normalized_shape=input_dim)
        self.msa = MultiHeadSelfAttention(d=input_dim)
        self.MLP = MLP(input_channels=input_dim, hidden_channels=[input_dim, input_dim], activation='gelu')

    def __call__(self, inputs):
        x = self.msa(self.norm1(inputs)) + inputs
        x = self.MLP(self.norm2(x)) + x
        return x


class MultiHeadSelfAttention(torch.nn.Module):

    def __init__(self, d):
        super(MultiHeadSelfAttention, self).__init__()
        self.d_k = d
        self.lin_k = torch.nn.Linear(in_features=d, out_features=self.d_k)
        self.lin_q = torch.nn.Linear(in_features=d, out_features=self.d_k)
        self.lin_v = torch.nn.Linear(in_features=d, out_features=self.d_k)
    
    def __call__(self, input : torch.Tensor) -> torch.Tensor:
        k = self.lin_k(input)
        v = self.lin_v(input)
        q = self.lin_q(input)

        kT = torch.transpose(k, dim0=-1, dim1=-2)
        x = F.softmax((q @ kT) / math.sqrt(self.d_k), dim=-1)

        x = torch.matmul(x, v)
        return x

class PositionalEmbedding2d(torch.nn.Module):

    def __init__(self, num_x, num_y, wavelengths=[1.0, 0.5, 0.25]):
        super(PositionalEmbedding2d, self).__init__()

        embeddings = torch.zeros([1, num_x*num_y, len(wavelengths)*4], dtype=torch.float32, requires_grad=False)
        for x in range(num_x):
            for y in range(num_y):
                for i, wavelength in enumerate(wavelengths):
                    index = y*num_y+x
                    norm_x = x / float(num_x)
                    norm_y = y / float(num_y)
                    embeddings[0][index][i] = math.sin(norm_x * wavelength)
                    embeddings[0][index][i+1] = math.cos(norm_x * wavelength)
                    embeddings[0][index][i+2] = math.sin(norm_y * wavelength)
                    embeddings[0][index][i+3] = math.cos(norm_y * wavelength)
        self.embeddings = nn.Parameter(embeddings)
        self.embedding_length = self.embeddings.size()[-1]

    
    def __call__(self, input : torch.Tensor) -> torch.Tensor:
        batch_size = input.shape[0]
        
        pos_embeddings_batch = torch.tile(self.embeddings, [batch_size, 1, 1])
        x = torch.concat((input, pos_embeddings_batch), dim=-1)
        return x

class ViT(torch.nn.Module):

    def __init__(self, input_channels, class_channels, grid_shape) -> None:
        super().__init__()
        self.positional_embedding = PositionalEmbedding2d(num_x=grid_shape[0], num_y=grid_shape[1])

        input_c = self.positional_embedding.embedding_length + input_channels
        self.encoder_blocks = torch.nn.Sequential(
            TransformerEncoderBlock(input_dim=input_c),
            TransformerEncoderBlock(input_dim=input_c),
            TransformerEncoderBlock(input_dim=input_c),
            TransformerEncoderBlock(input_dim=input_c),
        )
        self.learned_class_embedding = torch.nn.Parameter(torch.randn(input_c), requires_grad=True)
        self.final_mlp = MLP(input_channels=input_c, hidden_channels=[class_channels], activation=None)

    def __call__(self, inputs : torch.Tensor) -> torch.Tensor:
        batch_size = inputs.shape[0]
        x = self.positional_embedding(inputs)
        # B, 1, D
        batched_class_embedding = self.learned_class_embedding.expand(batch_size, -1).unsqueeze(1)
        x = torch.concat((batched_class_embedding, x), dim=1)
        x = self.encoder_blocks(x)
        x = self.final_mlp(x[..., 0, :])
        return F.softmax(x, dim=-1)
