import torch
import torchvision
from torch import nn
from torch.nn import TransformerDecoder, TransformerDecoderLayer, TransformerEncoder, TransformerEncoderLayer

from math import log


class Image_Augmenter(nn.Module):
    def __init__(self, cifar=False):
        super().__init__()
        self.n_head = 6
        self.n_layers = 1
        self.h_dim = 1024
        self.d_model = self.n_head * 64 # = 384 must be divisible by n_head
        self.d_input_augment_vector = 15 if cifar else 17

        input_size = 32 if cifar else 224
        kernel = 4 if cifar else 16
        self.map_size = input_size // kernel
        self.seq_size = self.map_size ** 2
        
        self.positional_encoding = PositionalEncoding(d_model=self.d_model, dropout=0.1, max_len=1+self.seq_size)

        self.augment_encoder = nn.Sequential(
            nn.Linear(self.d_input_augment_vector, self.h_dim, bias=False),
            nn.BatchNorm1d(self.h_dim),
            nn.ReLU(),
            ##### ONE HIDDEN LAYER BLOC #####
            nn.Linear(self.h_dim, self.h_dim, bias=False),
            nn.BatchNorm1d(self.h_dim),
            nn.ReLU(),
            ########### BLOC END ############
            nn.Linear(self.h_dim, self.d_model, bias=False),
            nn.BatchNorm1d(self.d_model),
            nn.ReLU(),
        )

        self.conv_input = nn.Sequential(
            # nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(32),
            # nn.ReLU(),
            # nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(64),
            # nn.ReLU(),
            nn.Conv2d(3, self.d_model, kernel_size=kernel, stride=kernel, padding=0, bias=False),
            nn.BatchNorm2d(self.d_model),
            nn.ReLU(),
        )

        encoder_layer = TransformerEncoderLayer(d_model = self.d_model,
                                                nhead = self.n_head,
                                                dim_feedforward = self.h_dim)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=self.n_layers)
        
        decoder_layer = TransformerDecoderLayer(d_model = self.d_model,
                                                  nhead = self.n_head,
                                                  dim_feedforward = self.h_dim)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=self.n_layers)

        self.deconv_output = nn.Sequential(
            nn.ConvTranspose2d(self.d_model, 64, kernel_size=kernel, stride=kernel, padding=0, bias=False),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(3, affine=False),
        )
        
    
    def forward(self, x, params):
        """
        x : (B, 3, 224, 224)
        params : (B, self.d_input_augment_vector)

        imagenet :
        (17) elt params vector => (d_model) embedding vector
        (224 x 224) image => (14 x 14) feature map => (196) tokens sequence => (14 x 14) feature map => (224 x 224) image

        cifar :
        (15) elt params vector => (d_model) embedding vector
        (32 x 32) image => (8 x 8) feature map => (94) tokens sequence => (8 x 8) feature map => (32 x 32) image
        """
        
        ### Image to Sequence ###
        feature_map = self.conv_input(x) # (B, D, H, W)
        B, _, H, W = feature_map.shape
        img_sequence = feature_map.view(B, self.d_model, self.seq_size) # (B, D, S)
        img_sequence = torch.permute(img_sequence, (2, 0, 1)) # (S, B, D)

        ### Prepare Keys ###
        augmentation_embedding = self.augment_encoder(params).unsqueeze(0) # (1, B, D)
        keys = torch.cat((augmentation_embedding, img_sequence), dim=0) # (S+1, B, D)
        keys = self.positional_encoding(keys)
        keys = self.transformer_encoder(keys)

        ### Prepare Queries ###
        # transfo_queries = torch.zeros_like(img_sequence)
        # transfo_queries = self.positional_encoding(transfo_queries)
        transfo_queries = keys[1:] # we remove the "augmentation_embedding" token
        
        ### Transformer inference ###
        seq_embedding = self.transformer_decoder(transfo_queries, keys) # Q: (S, B, D)  @  K/V: (S+1, B, D) => (S, B, D)

        ### Premare Upsample ###
        seq_embedding = torch.permute(seq_embedding, (1, 2, 0)) # (B, D, S)
        out_feature_map = seq_embedding.view(B, self.d_model, H, W) # (B, D, H, W)

        ### Upsample => back to img shape ###
        x_hat = self.deconv_output(out_feature_map) # (B, 3, H, W)

        return x_hat

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load_weights(self, path):
        self.load_state_dict(torch.load(path, weights_only=True))
    


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
        y = x + self.pe[:x.shape[0]]
        return self.dropout(y)
