
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GATConv

# https://github1s.com/Diego999/pyGAT

class GAT(nn.Module):
    def __init__(self, in_channels, hidden, out_channels, dropout, heads=1, args=None):
        super().__init__()
        self.dropout = dropout
        self.heads = heads
        self.conv1 = GATConv(in_channels, hidden, heads=self.heads, dropout=dropout)
        self.conv2 = GATConv(hidden * self.heads, out_channels, dropout=dropout)

    def forward(self, x, edge_index, return_attn=False):
        # x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x, edge_alpha = self.conv1(x, edge_index,return_attention_weights=True)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        logits = self.conv2(x, edge_index)
        # return F.log_softmax(x, dim=1)
        if return_attn:
            return x , edge_alpha[1]
        return logits


class GAT_body(nn.Module):
    def __init__(self, in_channels, hidden, dropout=0.0, heads=1, args=None):
        super().__init__()
        self.dropout = dropout
        self.heads = heads
        self.conv1 = GATConv(in_channels, hidden, heads=self.heads, dropout=dropout)
        self.conv2 = GATConv(hidden * self.heads, hidden, dropout=dropout)

    def forward(self, x, edge_index, return_attn=False):
        # x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x,attn = self.conv1(x, edge_index, return_attention_weights=True)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        logits = self.conv2(x, edge_index)
        # return F.log_softmax(x, dim=1)
        if return_attn:
            return logits,attn
        return logits

# import torch.nn as nn
# import torch.nn.functional as F
# # from dgl.nn.pytorch import GATConv
# from torch_geometric.nn import GATConv

# class GAT_body(nn.Module):
#     def __init__(self,
#                  num_layers,
#                  in_dim,
#                  num_hidden,
#                  heads,
#                  feat_drop,
#                  attn_drop,
#                  negative_slope,
#                  residual):
#         super(GAT_body, self).__init__()
#         self.num_layers = num_layers
#         self.gat_layers = nn.ModuleList()
#         self.activation = F.elu
#         # input projection (no residual)
#         self.gat_layers.append(GATConv(
#             in_dim, num_hidden, heads[0],
#             feat_drop, attn_drop, negative_slope, False, self.activation))
#         # hidden layers
#         for l in range(1, num_layers):
#             # due to multi-head, the in_dim = num_hidden * num_heads
#             self.gat_layers.append(GATConv(
#                 num_hidden * heads[l-1], num_hidden, heads[l],
#                 feat_drop, attn_drop, negative_slope, residual, self.activation))
#         # output projection
#         self.gat_layers.append(GATConv(
#             num_hidden * heads[-2], num_hidden, heads[-1],
#             feat_drop, attn_drop, negative_slope, residual, None))
        
#     def forward(self, x, edge_index):
#         h = x
#         for l in range(self.num_layers):
#             h = self.gat_layers[l](h, edge_index).flatten(1)

#         # output projection
#         logits = self.gat_layers[-1](h, edge_index)  # .mean(1)

#         return logits


# class GAT(nn.Module):
#     def __init__(self,
#                  num_layers,
#                  in_dim,
#                  num_hidden,
#                  num_classes,
#                  heads,
#                  feat_drop,
#                  attn_drop,
#                  negative_slope,
#                  residual):
#         super(GAT, self).__init__()

#         self.body = GAT_body(num_layers, in_dim, num_hidden, heads, feat_drop, attn_drop, negative_slope, residual)
#         self.fc = nn.Linear(num_hidden,num_classes)
#     def forward(self, x, edge_index):
#         logits = self.body(x, edge_index)
#         logits = self.fc(logits)

#         return logits