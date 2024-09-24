import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import get_optimizer

# 
class SineEncoding(nn.Module):
    def __init__(self, hidden_dim=128):
        super(SineEncoding, self).__init__()
        self.constant = 100
        self.hidden_dim = hidden_dim
        self.eig_w = nn.Linear(hidden_dim + 1, hidden_dim)
        # self.eig_w = nn.Linear(1, hidden_dim)

    def forward(self, eignvalue):

        eignvalue_con = eignvalue * self.constant
        div = torch.exp(torch.arange(0, self.hidden_dim, 2) * (-math.log(10000)/self.hidden_dim)).to(eignvalue.device)
        position = eignvalue_con.unsqueeze(1) * div
        eignvalue_pos = torch.cat((eignvalue.unsqueeze(1), torch.sin(position), torch.cos(position)), dim=1)
        # eignvalue_pos = eignvalue.unsqueeze(1) # [4,1]
        return self.eig_w(eignvalue_pos)

# mlp
class FeedForward(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedForward, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, feature):
        feature = self.layer1(feature)
        feature = self.gelu(feature)
        feature = self.layer2(feature)
        return feature


class GELayer(nn.Module):

    def __init__(self, nbases, ncombines, prop_dropout=0.0, norm='none'):
        super(GELayer, self).__init__()
        # self.nheads+1, hidden_dim
        self.prop_dropout = nn.Dropout(prop_dropout)

        if norm == 'none':
            self.weight = nn.Parameter(torch.ones((1, nbases, ncombines)))
        else:
            self.weight = nn.Parameter(torch.empty((1, nbases, ncombines)))
            nn.init.normal_(self.weight, mean=0.0, std=0.01)

        if norm == 'layer':
            self.norm = nn.LayerNorm(ncombines)
        elif norm == 'batch':
            self.norm = nn.BatchNorm1d(ncombines)
        else:
            self.norm = None

    def forward(self, feature):
        # print('feature.shape',feature.shape) # [67796, 5, 128]
        # print('self.weight',self.weight.shape) # [1, 2, 128]
        feature = self.prop_dropout(feature) * self.weight
        feature = torch.sum(feature, dim=1)

        if self.norm is not None:
            feature = self.norm(feature)
            feature = F.relu(feature)

        return feature


class FairGE_body(nn.Module):

    def __init__(self, nfeat, hidden_dim, nlayer=1, nheads=1,
                tran_dropout=0.0, feat_dropout=0.0, prop_dropout=0.0, norm='none',args=None):
        super(FairGE_body, self).__init__()
        self.args = args
        self.norm = norm
        self.nfeat = nfeat
        self.nlayer = nlayer
        self.nheads = nheads
        self.hidden_dim = hidden_dim


        # linear_encoder = onne layer Linear
        self.linear_encoder = nn.Linear(nfeat, hidden_dim)
        
        # self.classify = nn.Linear(hidden_dim, nclass)

        # position encoding
        self.eignvalue_encoder = SineEncoding(hidden_dim)
        
        # linear decoder
        self.decoder = nn.Linear(hidden_dim, nheads)

        self.mha_norm = nn.LayerNorm(hidden_dim)
        self.oc_norm = nn.LayerNorm(hidden_dim)
        self.mha_dropout = nn.Dropout(tran_dropout)
        self.oc_dropout = nn.Dropout(tran_dropout)
        self.mha = nn.MultiheadAttention(hidden_dim, nheads, tran_dropout)
        
        # mlp with gelu
        self.oc = FeedForward(hidden_dim, hidden_dim, hidden_dim)

        self.feat_dp1 = nn.Dropout(feat_dropout)
        self.feat_dp2 = nn.Dropout(feat_dropout)

        self.layers = nn.ModuleList([GELayer(self.nheads+1, hidden_dim, prop_dropout, norm=norm) for i in range(nlayer)])

    def forward(self, eignvalue, eignvector, feature):
        eignvector_T = eignvector.permute(1, 0)
        
        h = self.feat_dp1(feature)
        h = self.linear_encoder(h) # only this
            
        # position encoding - no h 
        eig = self.eignvalue_encoder(eignvalue) 
        # eig = self.eignvalue_encoder(eignvector) 

        # multi head attention with residual connection - no h
        mha_eig = self.mha_norm(eig)
        mha_eig, _ = self.mha(mha_eig, mha_eig, mha_eig)
        eig = eig + self.mha_dropout(mha_eig)

        # mlp with gelu - no h
        oc_eig = self.oc_norm(eig)
        oc_eig = self.oc(oc_eig)
        # eig = self.oc_dropout(oc_eig)
        eig = self.oc_dropout(oc_eig)+ eig

        # eig_gnn = eig # 
        eig_gnn = self.decoder(eig)
        
        for conv in self.layers:
            basic_feats = [h]
            eignvector_conv = eignvector_T @ h # 
            for i in range(self.nheads):
                basic_feats.append(eignvector @ (eig_gnn[:, i].unsqueeze(1) * eignvector_conv)) 
            basic_feats = torch.stack(basic_feats, axis=1)
            h = conv(basic_feats)


        h = self.feat_dp2(h)
            # h = self.classify(h)
        return h


class FairGE(nn.Module):

    def __init__(self, nfeat, hidden_dim, nclass, nlayer=1, nheads=1,
                tran_dropout=0.0, feat_dropout=0.0, prop_dropout=0.0, norm='none',args=None):
        super(FairGE, self).__init__()
        self.args = args
        self.norm = norm
        self.GNN = FairGE_body(nfeat, hidden_dim, nlayer, nheads, tran_dropout, feat_dropout, prop_dropout, norm, args)
        self.classifier = nn.Linear(hidden_dim, nclass)
        self.adv = nn.Linear(hidden_dim, 1) # for sensitive attribute
        
        optimizer_G = get_optimizer(args.optimizer)
        optimizer_A = get_optimizer(args.optimizer)
        
        self.G_params = list(self.GNN.parameters()) + list(self.classifier.parameters())
        
        self.optimizer_G = optimizer_G(self.G_params, lr = args.lr, weight_decay = args.ge_wd)
        self.optimizer_A = optimizer_A(self.adv.parameters(), lr = args.lr, weight_decay = args.ge_wd)
        self.criterion = nn.BCEWithLogitsLoss()
        self.G_loss = 0
        self.A_loss = 0
        
    def forward(self, eignvalue, eignvector, feature):
        h = self.GNN(eignvalue, eignvector, feature)
        logits = self.classifier(h)
        return logits
    
    def optimize(self, features, eignvalue, eignvector, labels, sens, idx_train_mask):
        self.train()
        self.adv.requires_grad_(False) # adv can not update weight
        self.optimizer_G.zero_grad()

        # s = self.estimator(features,edge_index) # a GCN , sensitive value (both label and psedo label) 伪标签
        h = self.GNN(eignvalue, eignvector, features) # learn a embeddings
        logits = self.classifier(h) # node classification

        s_g = self.adv(h).squeeze(1) # 预测的敏感属性 from embeddings
        y_score = F.softmax(logits,dim=1)[:,1]
        self.cov =  torch.abs(
            torch.mean(
                (sens[idx_train_mask] - torch.mean(sens[idx_train_mask])) * (y_score[idx_train_mask] - torch.mean(y_score[idx_train_mask]))
            )
        ) # correlation 
        self.cls_loss = self.criterion(logits[:,1][idx_train_mask], labels[idx_train_mask].float())
        self.adv_loss = self.criterion(s_g[idx_train_mask], sens[idx_train_mask]) # 两个gcn，试图让s_的分数接近s_score（部分标签）,adv(s_g) frozen,s_score is a sens label estimator

        self.G_loss = self.cls_loss  + self.args.ge_alpha * self.cov - self.args.ge_beta * self.adv_loss # but want to improve adv_loss (make s_g not similar to g_score)
        self.G_loss.backward()
        self.optimizer_G.step()
      
        self.adv.requires_grad_(True) # adv can update weight
        self.optimizer_A.zero_grad()
        # s_g = self.adv(h.detach())[:,1]
        s_g = self.adv(h.detach()).squeeze(1)
        self.A_loss = self.criterion(s_g[idx_train_mask], sens[idx_train_mask]) # reduce the loss
        self.A_loss.backward()
        self.optimizer_A.step()
        
        return logits, self.G_loss.item(), self.A_loss.item(), self.cls_loss.item(), self.cov.item()

