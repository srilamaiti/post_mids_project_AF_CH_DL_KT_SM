"""
    VATT Video model component described in:
    "VATT: Transformers for Multimodal Self-Supervised Learning from Raw Video, Audio and Text"
    Ideas:
        - Tokenization and Encoding of video frames
        - DropToken
        - Transformer Encoder
    Reference: 
        - https://arxiv.org/abs/2101.01169

"""
import torch
import torch.nn as nn
from einops import rearrange



