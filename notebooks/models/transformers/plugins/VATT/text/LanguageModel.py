"""
    VATT Language model component described in:
    "VATT: Transformers for Multimodal Self-Supervised Learning from Raw Video, Audio and Text"
    Ideas:
        - Tokenization and Encoding of text
        - 
    Reference: 
        - https://arxiv.org/abs/2101.01169

"""
import torch
import torch.nn as nn


class LanguageModel(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_size,
                 hidden_size,
                 trainable=True,
                 activation=nn.GELU()):
        """
            VATT Language model component
            ------------------------------------------
            Inputs:
                -  vocab_size {int}: The size of the vocabulary
                -  embedding_size {int}: The size of the token embedding
                -  hidden_size {int}: The size of the hidden layer


        """

        super(LanguageModel, self).__init__()

        # define the model components
        self.vocab_size = vocab_size                          # vocabulary size
        self.embedding = nn.Embedding(vocab_size, 
                                      embedding_size, 
                                      trainable=trainable)    # token embedding 


        # specify dense linear projection; else use identity matrix
        if hidden_size is not None:
            self.linear = nn.Linear(embedding_size, 
                                    hidden_size, 
                                    bias=False, 
                                    activation = activation,
                                    name='Linear Projection')
        else:
            self.linear = nn.Identity()


    def forward(self, x):
        """
            Perfrom one forward pass of the model
            ------------------------------------------
            Inputs:
                -  x {tensor}: The input tensor
            Outputs:
                -  out {tensor}: The output tensor
        """

        # token embedding
        x = self.embedding(x)

        # linear projection
        x = self.linear(x)

        return x
