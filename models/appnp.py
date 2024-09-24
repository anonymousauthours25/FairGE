
import torch
import torch.nn.functional as F
from torch.nn import Linear

from torch_geometric.nn import APPNP


# https://github1s.com/benedekrozemberczki/APPNP

class APPNPNet(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5, K=10, alpha=0.1, args=None):
        super().__init__()
        self.lin1 = Linear(nfeat, nhid)
        self.lin2 = Linear(nhid, nclass)
        self.prop1 = APPNP(K, alpha)
        self.dropout = dropout

    # def reset_parameters(self):
    #     self.lin1.reset_parameters()
    #     self.lin2.reset_parameters()

    def forward(self, x, edge_index):
        # x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop1(x, edge_index)
        # return F.log_softmax(x, dim=1)
        return x