from torch_geometric.nn import HypergraphConv

import torch.nn.functional as F
import torch.nn as nn
class HGNN(nn.Module):
    def __init__(self, kg_emb_dim, dropout=0.3):
        super(HGNN, self).__init__()
        self.dropout = dropout
        self.gat1 = HypergraphConv(kg_emb_dim, kg_emb_dim)
        self.gat2 = HypergraphConv(kg_emb_dim, kg_emb_dim)

    def forward(self, kg_embedding, session_hyper_edge_index):

        x = self.gat1(kg_embedding, session_hyper_edge_index)
        #x= self.gat2(x, session_hyper_edge_index)
        # x = self.gat2(x, session_hyper_edge_index)
        # x = self.gat2(x, session_hyper_edge_index)

        return x