import torch
import torch.nn as nn
from .ltr_loss import point_wise_mse, list_wise_rankcosine
from .neural_fingerprint import NeuralFingerprint
from .attention import Attention

from scipy.stats import spearmanr, rankdata
import numpy as np


class Graph_Transformer(nn.Module):
    def __init__(self, drug_input_dim, gene_embed, gene_input_dim, encode_dim, fp_type, loss_type, label_type, device,
                 n_layers, n_heads, 
                 initializer=None, pert_type_input_dim=None, cell_id_input_dim=None, pert_idose_input_dim=None,
                 use_pert_type=False, use_cell_id=False, use_pert_idose=False):
        super(Graph_Transformer, self).__init__()
        self.fp_type = fp_type
        self.use_pert_type = use_pert_type
        self.use_cell_id = use_cell_id
        self.use_pert_idose = use_pert_idose
        if self.fp_type == 'neural':
            self.input_dim = 1024 + gene_input_dim
            self.drug_fp = NeuralFingerprint(drug_input_dim['atom'], drug_input_dim['bond'], conv_layer_sizes=[64, 64],
                                             output_size=1024, degree_list=[0, 1, 2, 3, 4, 5], device=device)
        else:
            self.input_dim = drug_input_dim + gene_input_dim
        if self.use_pert_type:
            self.input_dim += pert_type_input_dim
        if self.use_cell_id:
            self.input_dim += cell_id_input_dim
        if self.use_pert_idose:
            self.input_dim += pert_idose_input_dim
        self.encode_dim = encode_dim
        self.gene_embed = nn.Embedding(978, gene_input_dim).from_pretrained(gene_embed, freeze=True)
        self.encoder = nn.Sequential(nn.Linear(self.input_dim, 2 * self.encode_dim), nn.ReLU(), nn.Dropout(0.1),
                                     nn.Linear(2 * self.encode_dim, self.encode_dim), nn.ReLU(), nn.Dropout(0.1))
        self.decoder = nn.Sequential(nn.Linear(self.encode_dim, 128), nn.ReLU(), nn.Dropout(0.1),
                                     nn.Linear(128, 32), nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, 1))
        self.attention = Attention(self.encode_dim, n_layers=n_layers, n_heads=n_heads, pf_dim=self.encode_dim, dropout=0.1,
                                   device=device)
        
        self.initializer = initializer
        # self.init_weights()
        self.sigmoid = nn.Sigmoid()
        self.loss_type = loss_type
        self.label_type = label_type
        self.device = device

    def init_weights(self):
        if self.initializer is None:
            return
        for name, parameter in self.named_parameters():
            if parameter.dim() == 1:
                nn.init.constant_(parameter, 0.)
            else:
                self.initializer(parameter)

    def forward(self, input_drug, input_gene, input_pert_type, input_cell_id, input_pert_idose):
        n_batch, num_gene = input_gene.shape
        if self.fp_type == 'neural':
            input_drug = self.drug_fp(input_drug)
        if self.use_pert_type:
            input_drug = torch.cat((input_drug, input_pert_type), dim=1)
        if self.use_cell_id:
            input_drug = torch.cat((input_drug, input_cell_id), dim=1)
        if self.use_pert_idose:
            input_drug = torch.cat((input_drug, input_pert_idose), dim=1)
        input_drug = input_drug.unsqueeze(1)
        input_drug = input_drug.repeat(1, num_gene, 1)
        input_gene = self.gene_embed(input_gene)
        input = torch.cat((input_drug, input_gene), dim=2)
        input_encode = self.encoder(input)
        input_attn, attn = self.attention(input_encode, None)
        input_attn = input_encode + input_attn
        output = self.decoder(input_attn)
        if self.label_type == 'binary' or self.label_type == 'binary_reverse':
            out = self.sigmoid(output.squeeze(2))
        elif self.label_type == 'real' or self.label_type == 'real_reverse':
            out = output.squeeze(2)
        else:
            raise ValueError('Unknown label_type: %s' % self.label_type)
        return out
    
    def predict(self, input_drug, input_gene, input_pert_type, input_cell_id, input_pert_idose):
        
        y_pred = self.forward(input_drug, input_gene, input_pert_type, input_cell_id, input_pert_idose).cpu().numpy()
        
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

    def loss(self, label, predict):
        if self.loss_type == 'point_wise_mse':
            loss = point_wise_mse(label, predict)
        elif self.loss_type == 'list_wise_rankcosine':
            loss = list_wise_rankcosine(label, predict)
        else:
            raise ValueError('Unknown loss: %s' % self.loss_type)
        return loss
