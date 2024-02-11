"""
    Implementation of ViT: Vision Transformer from 
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
    Ideas:
        - break image into patches and flatten them
        - add a CLS token
        - pass through a transformer encoder
        - pass through a linear layer
        - pass through softmax
    Reference: 
        - https://arxiv.org/abs/2010.11929
"""

# NN import
import torch
import torch.nn as nn
from einops import rearrange

# local imports
from .utils.TransformerBlock import TransformerEncoder


class ViT(nn.Module):
    def __init__(self,
                 img_dim,
                 in_channels=3,
                 patch_dim=16,
                 num_classes=10,
                 dim=512,
                 blocks=6,
                 heads=4,
                 dim_linear_block=1024,
                 dim_head=None,
                 dropout=0, 
                 transformer=None, 
                 classification=True):
        """
            Implementation of the Vision Transformer
            ----------------------------------------
            Inputs:
                img_dim {int}: image dimension
                in_channels {int, default: 3}: number of image channels
                patch_dim {int, default: 16}: patch dimension
                num_classes {int, default: 10}: number of classes
                dim {int, default: 512} the linear layer's dim to project the patches for MHSA
                blocks {int, default: 6}: number of transformer blocks
                heads {int, default: 4}: number of distinct representations to learn (pay attention to)
                dim_linear_block {int, default: 1024}: dimension of the linear block
                dim_head {int, default: None, Optional}: dimension of each head (default: dim/heads)
                dropout {float, default: 0.1}: dropout rate
                transformer {nn.Module, default: None, Optional}: transformer block to use
                classification {bool, default: True}: creates an extra CLS token
        """
        super(ViT, self).__init__()                  # inherit from nn.Module   
        assert img_dim % patch_dim == 0, f"patch size {patch_dim} not divisible"
        
        self.p = patch_dim                                  # patch dimension
        self.tokens = (img_dim // patch_dim) ** 2           # number of patches; effective input sequence length
        self.token_dim = in_channels * (patch_dim ** 2)     # token dimension; dimension of sequence of flattened 2D patches
        self.dim = dim                                      # lienar layer dimension
        self.classification = classification                # added CLS toekn flag
        self.heads = heads                                  # number of heads
        self.dim_head = (int(dim // heads)) if dim_head is None else dim_head     # dimension of each head
        
        self.project_patches = nn.Linear(self.token_dim, dim)                     # project flattened patches to token dimension; the patch embeddings layer
        self.emb_dropout = nn.Dropout(dropout)                                    # embedding dropout layer

        # Add learnable CLS Token and postional Embedding
        if classification:
            self.cls_token = nn.Parameter(torch.randn(1, 1, dim))                 # CLS token
            self.pos_emb1D = nn.Parameter(torch.randn(self.tokens + 1, dim))      # positional embedding
            self.mlp_head = nn.Linear(dim, num_classes)                           # MLP head; the output to the classifcation layer 
        else:
            self.pos_emb1D = nn.Parameter(torch.randn(self.tokens, dim))          # positional embedding


        # Enable default VIT Transformer of use the one passed in
        if transformer is None:
            self.transformer = TransformerEncoder(dim, blocks=blocks, heads=heads,
                                                  dim_head=self.dim_head,
                                                  dim_linear_block=dim_linear_block,
                                                  dropout=dropout)
        else:
            self.transformer = transformer


    def expand_cls_to_batch(self, batch):
        """
            Method to expand the CLS token to all sample within a batch
            ----------------------------------------------------------------
            Inputs:
                batch {tensor}: batch of samples
            Outputs:
                batch {tensor}: batch of samples with CLS token expanded
        """
        return self.cls_token.expand([batch, -1, -1])


    def forward(self, img, mask=None):
        """
            Method to perform one forward pass through a batch of the Vision Transformer
            -------------------------------------------------------------------------------
            Inputs:
                x {tensor}: input tensor of shape [batch, channels, height, width]
                mask {tensor}: mask of the input
            Outputs:
                out {tensor}: output of the self attention layer
        """

        batch_size = img.shape[0]                # batch size

        # rearrange and flatten the image
        # [batch, channels, height, width] -> [batch, tokens, token_dim]
        x_p = rearrange(img, 'b c (p1 h) (p2 w) -> b (h w) (p1 p2 c)', p1=self.p, p2=self.p)

        # project the flattened patches into patch embeddings
        # add position embedding
        # add CLS token

        x_p = self.project_patches(x_p)


        if self.classification:
            x_p = torch.cat((self.expand_cls_to_batch(batch_size), x_p), dim=1)

        x_pe = self.emb_dropout(x_p + self.pos_emb1D)

        # pass through the transformer encoder
        # [batch, tokens, dim]
        y = self.transformer(x_pe, mask)

        # pass through the MLP head as needed
        if self.classification:
            # similar to BERT, takes only the CLS token for the classification task
            return self.mlp_head(y[:, 0, :])
        else:
            return y