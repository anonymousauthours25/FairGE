# import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# https://github1s.com/tkipf/pygcn/blob/HEAD/pygcn

# class GCN(nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels,dropout):
#         super().__init__()
#         self.dropout = dropout
#         self.conv1 = GCNConv(in_channels, hidden_channels)
#         self.conv2 = GCNConv(hidden_channels, out_channels)

#     def forward(self, x, edge_index, edge_weight=None):
#         x = F.dropout(x, p=self.dropout, training=self.training)
#         x = F.relu(self.conv1(x, edge_index, edge_weight)) 
#         x = F.dropout(x, p=self.dropout, training=self.training)
#         x = self.conv2(x, edge_index, edge_weight)
#         return x


class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, nlayer=2, args=None):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.nlayer = nlayer
        self.convs = nn.ModuleList()
        # self.convs.append(GCNConv(in_channels, hidden_channels))
        for layer in range(self.nlayer-2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        # self.convs.append(GCNConv(hidden_channels, out_channels))
        
    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.conv1(x, edge_index, edge_weight)) 
        for conv in self.convs:
            x = F.dropout(x, self.dropout, training=self.training)
            x = conv(x, edge_index, edge_weight).relu()
        # x = F.dropout(x, p=self.dropout, training=self.training)
        # x = F.relu(self.conv1(x, edge_index, edge_weight)) 
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x


class GCN_fair(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super().__init__()
        self.body = GCN_body(nfeat,nhid,dropout)
        self.fc = nn.Linear(nhid, nclass)

    def forward(self, x, edge_index):
        x = self.body(x, edge_index)
        x = self.fc(x)
        return x

class GCN_body(nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout, nlayer=2, args=None):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GCNConv(in_channels, hidden_channels)
        # self.conv2 = GCNConv(hidden_channels, out_channels)
        self.nlayer = nlayer
        self.convs = nn.ModuleList()
        # self.convs.append(GCNConv(in_channels, hidden_channels))
        for layer in range(self.nlayer-1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        # self.convs.append(GCNConv(hidden_channels, out_channels))
        
    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.conv1(x, edge_index, edge_weight)) 
        # x = F.dropout(x, p=self.dropout, training=self.training)
        for i in range(len(self.convs)):
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.convs[i](x, edge_index, edge_weight)
            if i != len(self.convs)-1:
                x = F.relu(x)
        return x

# class GCN(nn.Module):
#     def __init__(self, nfeat, nhid, nclass, dropout):
#         super(GCN, self).__init__()
#         self.body = GCN_Body(nfeat,nhid,dropout)
#         self.fc = nn.Linear(nhid, nclass)

#         for m in self.modules():
#             self.weights_init(m)

#     def weights_init(self, m):
#         if isinstance(m, nn.Linear):
#             torch.nn.init.xavier_uniform_(m.weight.data)
#             if m.bias is not None:
#                 m.bias.data.fill_(0.0)

#     def forward(self, x, edge_index):
#         x = self.body(x, edge_index)
#         x = self.fc(x)
#         return x


# class GCN_Body(nn.Module):
#     def __init__(self, nfeat, nhid, dropout):
#         super(GCN_Body, self).__init__()
#         self.gc1 = GCNConv(nfeat, nhid)

#     def forward(self, x, edge_index):
#         x = self.gc1(x, edge_index)
#         return x    




