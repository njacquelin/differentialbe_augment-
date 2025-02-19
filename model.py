import torch
from torch import nn
from torch.nn import TransformerDecoder, TransformerDecoderLayer

from math import log


class Image_Augmenter(nn.Module):
    def __init__(self, cifar=False):
        super().__init__()
        self.n_head = 6
        self.n_layers = 1
        self.h_dim = 1024
        self.d_model = 384 # must be divisible by n_head
        self.d_input_augment_vector = 15 if cifar else 17

        input_size = 32 if cifar else 224
        kernel = 4 if cifar else 16
        self.map_size = input_size // kernel
        self.seq_size = self.map_size ** 2
        
        self.positional_encoding = PositionalEncoding(d_model=self.d_model, dropout=0.1, max_len=self.seq_size)

        self.transform_encoder = nn.Sequential(
            nn.Linear(self.d_input_augment_vector, self.h_dim),
            nn.BatchNorm1d(self.h_dim),
            nn.ReLU(),
            ##### ONE HIDDEN LAYER BLOC #####
            nn.Linear(self.h_dim, self.h_dim),
            nn.BatchNorm1d(self.h_dim),
            nn.ReLU(),
            ########### BLOC END ############
            nn.Linear(self.h_dim, self.d_model),
            nn.BatchNorm1d(self.d_model),
            nn.ReLU(),
        )

        self.conv_input = nn.Sequential(
            nn.Conv2d(3, self.d_model, kernel_size=kernel, stride=kernel, padding=0),
            nn.BatchNorm2d(self.d_model),
            nn.ReLU(),
        )

        decoder_layer = TransformerDecoderLayer(d_model = self.d_model,
                                                  nhead = self.n_head,
                                                  dim_feedforward = self.h_dim)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=6)

        self.deconv_output = nn.Sequential(
            nn.ConvTranspose2d(self.d_model, 3, kernel_size=kernel, stride=kernel, padding=0),
            nn.BatchNorm2d(3, affine=False),
        )
        

    
    def forward(self, x, params):
        """
        x : (B, 3, 224, 224)
        params : (B, self.d_input_augment_vector)

        (17) elt params vector => (d_model) embedding vector
        (224 x 224) image => (14 x 14) feature map => (196) tokens sequence => (14 x 14) feature map => (224 x 224) image
        """
        transformation_embedding = self.transform_encoder(params).unsqueeze(0)
        
        feature_map = self.conv_input(x) # (B, D, H, W)
        sequence = feature_map.view(-1, self.d_model, self.seq_size) # (B, D, S)
        sequence = torch.permute(sequence, (2, 0, 1)) # (S, B, D)
        
        sequence = self.positional_encoding(sequence)
        seq_embedding = self.transformer_decoder(sequence, transformation_embedding) # (S, B, D)

        out_feature_map = torch.permute(seq_embedding, (1, 2, 0)) # (B, D, S)
        out_feature_map = out_feature_map.view(-1, self.d_model, self.map_size, self.map_size) # (B, D, H, W)

        x_hat = self.deconv_output(out_feature_map)

        return x_hat




class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 196):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (- log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
            x : (S, B, D)
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
