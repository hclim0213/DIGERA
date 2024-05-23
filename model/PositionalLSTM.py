import math
import os
import numpy as np
from tempfile import TemporaryDirectory
from typing import Tuple

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
from scipy.stats import spearmanr, rankdata

from .ltr_loss import point_wise_mse, list_wise_rankcosine

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).permute(1, 0, 2)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:, :x.size(0)]
        return self.dropout(x)
        
class LSTMModel(nn.Module):
    def __init__(self, input_dim, d_model: int, output_dim:int, d_hid: int, meta_dim: int, 
                 nlayers: int, dropout, loss_type, device):
        super().__init__()
        
        max_len = input_dim[0]
        ntoken = input_dim[1]
        self.d_model = d_model
        self.nhid = d_hid
        self.nlayers = nlayers 
        
        self.loss_type = loss_type
        self.device = device

        self.dropout = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, d_model) # ntoken = 50

        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)
        self.lstm = nn.LSTM(max_len, d_hid, nlayers, dropout=dropout) # 120, 512, 2 

        self.meta_embed = nn.Sequential(nn.Linear(meta_dim, meta_dim*2), nn.ReLU(), nn.Dropout(dropout), 
                                  nn.Linear(meta_dim*2, d_hid), nn.ReLU(), nn.Dropout(dropout))
        
        self.decoder = nn.Linear(d_hid, output_dim)

        self.init_weights()
        

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, cond, hidden):
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        src = src.permute(0, 3, 2, 1) # pos v1 
        src = src.mean(dim=1)
        ## positional encoding parts 

        output, hidden = self.lstm(src, hidden)
        output = self.dropout(output)

        meta = self.meta_embed(cond).unsqueeze(2).permute(0, 2, 1)
        output = torch.cat((output, meta), dim=1)

        decoded = self.decoder(output).mean(dim=1)
        return decoded, hidden


    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                weight.new_zeros(self.nlayers, bsz, self.nhid))

    def loss(self, label, predict):
        if self.loss_type == 'point_wise_mse':
            loss = point_wise_mse(label, predict)
        elif self.loss_type == 'list_wise_rankcosine':
            loss = list_wise_rankcosine(label, predict)
        else:
            raise ValueError('Unknown loss: %s' % self.loss_type)
        return loss
        
    def predict(self, x, cond, hidden):
        
        y_pred, hidden = self.forward(x, cond, hidden)
        
        ranks_up = rankdata(y_pred, method='min', axis=1)
        ranks_down = rankdata(-y_pred, method='min', axis=1)

        y_pred_rank = np.where(ranks_down <=1, 1.0,
                      np.where(ranks_down <=10, 0.9,
                      np.where(ranks_down <=50, 0.8,
                      np.where(ranks_down <=100, 0.7,
                      np.where(ranks_down <=200, 0.6,
                      np.where(ranks_up <= 1, 0.0,
                      np.where(ranks_up <= 10, 0.1, 
                      np.where(ranks_up <= 50, 0.2,
                      np.where(ranks_up <= 100, 0.3,
                      np.where(ranks_up <= 200, 0.4, 0.5))))))))))
        
        return y_pred_rank
