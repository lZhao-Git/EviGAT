import torch.nn as nn
import torch.nn.functional as F
import dgl
from layers import GATLayer
import torch
from weight_sum_and_max import WeightedSumAndMax
import numpy as np
import csv

class GATModel(nn.Module):
    def __init__(self, in_feats, hidden_feats=None, num_heads=None, feat_drops=None,
                 attn_drops=None, alphas=None, residuals=None, agg_modes=None, activations=None,
                 n_tasks=1,predictor_dropout=0.,lamb2=None):
        super(GATModel, self).__init__()

        self.lamb2 = lamb2
        n_layers = len(hidden_feats)
        self.gnn_layers = nn.ModuleList()
        for i in range(n_layers):
            self.gnn_layers.append(GATLayer(in_feats, hidden_feats[i], num_heads[i],
                                            feat_drops[i], attn_drops[i], alphas[i],
                                            residuals[i], agg_modes[i], activations[i]))
            if agg_modes[i] == 'flatten':
                in_feats = hidden_feats[i] * num_heads[i]
            else:
                in_feats = hidden_feats[i]

        self.readout = WeightedSumAndMax(in_feats)

        self.predict = nn.Sequential(
            nn.Linear(2*in_feats, 64),  
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),  
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),  
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(predictor_dropout),
            nn.Linear(64,2)   
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, bg, nodefeats,return_output='alpha'):
        for gnn in self.gnn_layers:
            nodefeats = gnn(bg, nodefeats)
        graph_feats = self.readout(bg, nodefeats)  
        logits = self.predict(graph_feats)  

        evidence = F.softplus(logits)
        alpha = evidence + self.lamb2

        if return_output == 'hard':
            return predict(logits),logits,evidence
        elif return_output == 'soft':
            return self.softmax(logits),logits,evidence
        elif return_output == 'alpha':
            return alpha,logits,evidence
        else:
            raise AssertionError
    
def predict(p):
    output_pred = torch.max(p, dim=-1)[1]
    return output_pred
