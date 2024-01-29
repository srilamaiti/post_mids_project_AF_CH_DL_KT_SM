"""
    Base Transformer Block Architecture including Self-Attention:
        - Multi-Headed Self-Attention
        - Add & Layer Normalization
        - Position-wise Feed Forward
        - Add & Layer Normalization
    Reference:
        - https://arxiv.org/pdf/1706.03762.pdf
        - https://machinelearningmastery.com/the-transformer-model/
"""
# NN import
import torch
import torch.nn as nn
import torch.nn.functional as F

# local imports
from .MultiHeadedSelfAttention import MultiHeadedSelfAttention


class TransfomerBlock(nn.Module):
    def __init__(self, dim, heads=8, dim_heads=None, dim_linear_block=1024, dropout=0.1, activation=nn.GELU()):
        """
            Implementation of the base transformer block
            --------------------------------------------
            Inputs:
                dim {int}: token dimension or the embedding size of the input token
                heads {int, default: 1}: number of distinct representations to learn (pay attention to)
                dim_heads {int, default: None, Optional}: dimension of each head (default: dim/heads)
                dim_linear_block {int, default: 1024}: dimension of the linear block
                dropout {float, default: 0.1}: dropout rate
                activation {torch.nn.activation, default: nn.GELU()}: activation function to use
        """

        super().__init__()                  # inherit from nn.Module
        self.attention = MultiHeadedSelfAttention(dim=dim, heads=heads, dim_heads=dim_heads)   # multi-headed self attention layer
        self.dropout = nn.Dropout(dropout)
        self.norm_1 = nn.LayerNorm(dim)      # layer normalization 1
        self.norm_2 = nn.LayerNorm(dim)      # layer normalization 2

        # Define the Linear Layer
        self.linear = nn.Sequential(
            nn.Linear(dim, dim_linear_block),
            activation(),                           # activation function (ReLU, GELU, etc.)
            nn.Dropout(dropout),
            nn.Linear(dim_linear_block, dim),
            nn.Dropout(dropout)
        )


    def forward(self, x, mask=None):
        """
            Method to perform one forward pass through a batch of the transformer block
            Inputs:
                x {tensor}: input tensor of shape [batch, tokens, dim]
                mask {tensor}: mask of the input
            Outputs:
                out {tensor}: output of the self attention layer
        """


        # start the attention layer
        attention = self.attention(x, mask)
        
        # first skip connection with layer normalization and dropout layer
        x = self.norm_1(self.dropout(attention + x))

        # pass through the feed forward layer
        # add and normalization and dropout layer
        out = self.norm_2(self.dropout(self.linear(x) + x))
        return out
    

class TransformerEncoder(nn.Module):
    def __init__(self, dim, blocks=6, heads=8, dim_head=None, dim_linear_block=1024, dropout=0):
        """
            Implementation of the Transformer Encoder
            -----------------------------------------
            Inputs:
                dim {int}: token dimension or the embedding size of the input token
                blocks {int, default: 6}: number of transformer blocks
                heads {int, default: 8}: number of distinct representations to learn (pay attention to)
                dim_head {int, default: None, Optional}: dimension of each head (default: dim/heads)
                dim_linear_block {int, default: 1024}: dimension of the linear block
                dropout {float, default: 0.1}: dropout rate
        """

        super().__init__()                  # inherit from nn.Module
        self.blocks = nn.ModuleList(
            [
                TransfomerBlock(
                    dim=dim,
                    heads=heads,
                    dim_heads=dim_head,
                    dim_linear_block=dim_linear_block,
                    dropout=dropout
                )
                for _ in range(blocks)
            ]
        )


    def forward(self, x, mask=None):
        """
            Method to perform one forward pass through a batch of the transformer encoder
            Inputs:
                x {tensor}: input tensor of shape [batch, tokens, dim]
                mask {tensor}: mask of the input
            Outputs:
                out {tensor}: output of the self attention layer
        """

        # iterate through the transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        return x