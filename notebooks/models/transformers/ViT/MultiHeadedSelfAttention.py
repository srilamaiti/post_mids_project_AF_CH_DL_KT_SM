"""
    Multi-headed self attention layer
    Following the Attention mechanism formula from the paper (Attention is All You Need)
        - Attention(Q, K, V) = softmax( ( Q @ K^T ) / sqrt(d_k) ) @ V
        - d_k = embed_size // heads
        - Q = queries, K = keys, V = values
        - fc_out = fully connected layer to convert the concatenated heads back to the original embed_size
    Reference:
        - https://arxiv.org/pdf/1706.03762.pdf
"""
# tensor operation 
import einops
import numpy as np

# NN import
import torch
import torch.nn as nn
import torch.nn.functional as F



class MultiHeadedSelfAttention(nn.Module):
    def __init__(self, dim, heads=1, dim_heads=None):
        """
            Implementation of multi-headed self attention layer (Attention is All You Need)
            -------------------------------------------------------------------------------
            Inputs:
                dim     {int}: token dimension or the embedding size of the input token
                heads   {int, default: 1}: number of distinct representations to learn (pay attention to)
                dim_heads {int, default: None, Optional}: dimension of each head (default: dim/heads)
            Outputs:
                out {tensor}: output of the self attention layer
        """
        super(MultiHeadedSelfAttention, self).__init__()                  # inherit from nn.Module
        self.dim = dim                      # token dimension
        self.heads = heads                  # number of heads
        self.dim_heads = (dim // heads) if dim_heads is None else dim_heads     # dimension of each head

        # check if the dimension of each head is divisible by the number of heads
        assert (self.dim_heads * heads == dim), "Embedding dimension needs to be divisible by number of heads"

        # Step 1. Define the query, key, and value matrices for all heads 
        _dim = self.dim_heads * self.heads
        self.to_qkv = nn.Linear(dim, _dim * 3, bias=False) 

        # Step 2. Define the denominator of the softmax function
        self.scale = self.dim_heads ** -0.5

        # Step 3. Define the fully connected layer to convert the concatenated heads back to the original embed_size
        self.fc_out = nn.Linear(_dim, dim, bias=False)


    def forward(self, x, mask=None):
        """
            Method to perform one forward pass through a batch of the self attention layer
            -------------------------------------------------------------------------------
            Inputs:
                x {tensor}: input tensor of shape [batch, tokens, dim]
                mask {tensor}: mask of the input
            Outputs:
                out {tensor}: output of the self attention layer
        """
        assert x.dim() == 3, "Input tensor must have 3 dimensions: [batch, tokens, dim]"

        # Step 1: get QKV
        qkv = self.to_qkv(x)                        # [batch, tokens, dim*3*heads]

        # Step 2: 
        # split QKV into Q, K, V
        # decompose the axis in 3 equal parts
        # [batch, tokens, dim*3*heads] -> [3, batch, heads, tokens, dim]
        q, k, v = tuple(einops.rearrange(qkv, "b t (d k h) -> k b h t d", k = 3, h = self.heads))

        # Step 3:
        # compute the dot product between the query and the keys
        # for each word in the input sentence, how much do we need to pay attention to each word in the input sentence
        # [batch, heads, tokens, dim] @ [batch, heads, tokens, dim].T -> [batch, heads, tokens, tokens]
        scaled_dot_product = torch.einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        # Step 4:
        # apply masking (if any) to the scaled dot product
        # prevent the model from paying attention to the preceeding token
        if mask is not None:
            assert mask.shape == scaled_dot_product.shape[-2:], "Mask shape must be broadcastable to the scaled dot product"
            scaled_dot_product = scaled_dot_product.masked_fill(mask, -np.inf)

        # Step 5: apply softmax to the scaled dot product
        attention = torch.softmax(scaled_dot_product, dim=-1)

        # Step 6:
        # apply dot product of scale dot product and value
        # [batch, heads, tokens, tokens] @ [batch, heads, tokens, dim].T -> [batch, heads, tokens, dim]
        out = torch.einsum("b h i j, b h j d -> b h i d", attention, v)

        # Step 7:
        # recompose / merge head with the dimension of each head
        # [batch, heads, tokens, dim] -> [batch, tokens, dim*heads]
        out = einops.rearrange(out, "b h t d -> b t (h d)")

        # Step 8:
        # apply fully connected layer to convert the concatenated heads back to the original embed_size
        out = self.fc_out(out)

        return out

