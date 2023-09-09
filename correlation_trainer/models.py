import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GINConv, global_add_pool
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
from utils import preprocessing, normalize_adj
import time
from mlp import MLP

from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv
from torch_geometric.nn import global_mean_pool, global_add_pool


# =============================================================================

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter


def init_tensor(tensor, init_type, nonlinearity):
    if tensor is None or init_type is None:
        return

    if init_type =='thomas':
        size = tensor.size(-1)
        stdv = 1. / math.sqrt(size)
        nn.init.uniform_(tensor, -stdv, stdv)
    elif init_type == 'kaiming_normal_in':
        nn.init.kaiming_normal_(tensor, mode='fan_in', nonlinearity=nonlinearity)
    elif init_type == 'kaiming_normal_out':
        nn.init.kaiming_normal_(tensor, mode='fan_out', nonlinearity=nonlinearity)
    elif init_type == 'kaiming_uniform_in':
        nn.init.kaiming_uniform_(tensor, mode='fan_in', nonlinearity=nonlinearity)
    elif init_type == 'kaiming_uniform_out':
        nn.init.kaiming_uniform_(tensor, mode='fan_out', nonlinearity=nonlinearity)
    elif init_type == 'orthogonal':
        nn.init.orthogonal_(tensor, gain=nn.init.calculate_gain(nonlinearity))
    else:
        raise ValueError(f'Unknown initialization type: {init_type}')


class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True, weight_init='thomas', bias_init='thomas'):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.DoubleTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.DoubleTensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.weight_init = weight_init
        self.bias_init = bias_init
        self.reset_parameters()

    def reset_parameters(self):
        init_tensor(self.weight, self.weight_init, 'relu')
        init_tensor(self.bias, self.bias_init, 'relu')

    def forward(self, adjacency, features):
        support = torch.matmul(features, self.weight)
        output = torch.bmm(adjacency, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class EAGLE_M_GINADJ(Module):
    def __init__(self,
                num_features=0, 
                num_layers=5,
                num_hidden=512,
                dropout_ratio=0,
                weight_init='thomas',
                bias_init='thomas',
                binary_classifier=False,
                augments=0):

        super(EAGLE_M_GINADJ, self).__init__()
        self.nfeat = num_features
        self.nlayer = num_layers
        self.nhid = num_hidden
        self.dropout_ratio = dropout_ratio
        self.gc = nn.ModuleList([GraphConvolution(self.nfeat if i==0 else self.nhid, self.nhid, bias=True, weight_init=weight_init, bias_init=bias_init) for i in range(self.nlayer)])
        self.bn = nn.ModuleList([nn.LayerNorm(self.nhid).double() for i in range(self.nlayer)])
        self.relu = nn.ModuleList([nn.ReLU().double() for i in range(self.nlayer)])
        if not binary_classifier:
            self.fc = nn.Linear(self.nhid + augments, 1).double()
        else:
            if binary_classifier == 'naive':
                self.fc = nn.Linear(self.nhid + augments, 1).double()
            elif binary_classifier == 'oneway' or binary_classifier == 'oneway-hard':
                self.fc = nn.Linear((self.nhid + augments) * 2, 1).double()
            else:
                self.fc = nn.Linear((self.nhid + augments) * 2, 2).double()

            if binary_classifier != 'oneway' and binary_classifier != 'oneway-hard':
                self.final_act = nn.LogSoftmax(dim=1)
            else:
                self.final_act = nn.Sigmoid()

        self.dropout = nn.ModuleList([nn.Dropout(self.dropout_ratio).double() for i in range(self.nlayer)])

        self.binary_classifier = binary_classifier

    def forward_single_model(self, adjacency, features):
        x = self.relu[0](self.bn[0](self.gc[0](adjacency, features)))
        x = self.dropout[0](x)
        for i in range(1,self.nlayer):
            x = self.relu[i](self.bn[i](self.gc[i](adjacency, x)))
            x = self.dropout[i](x)

        return x

    def extract_features(self, adjacency, features, augments=None):
        x = self.forward_single_model(adjacency, features)
        x = x[:,0] # use global node
        if augments is not None:
            x = torch.cat([x, augments], dim=1)
        return x

    def regress(self, features, features2=None):
        if not self.binary_classifier:
            assert features2 is None
            return self.fc(features)

        assert features2 is not None
        if self.binary_classifier == 'naive':
            x1 = self.fc(features)
            x2 = self.fc(features2)
        else:
            x1 = features
            x2 = features2

        x = torch.cat([x1, x2], dim=1)
        if self.binary_classifier != 'naive':
            x = self.fc(x)

        x = self.final_act(x)
        return x

    def forward(self, features, adjacency, augments=None):
        adjacency, features = adjacency.double(), features.double()
        x = self.forward_single_model(adjacency, features)
        x = x[:,0] # use global node
        if augments is not None:
            x = torch.cat([x, augments], dim=1)
        return self.fc(x)

    def reset_last(self):
        self.fc.reset_parameters()

    def final_params(self):
        return self.fc.parameters()
    
class EAGLE_M(Module):
    def __init__(self,
                num_features=0, 
                num_layers=5,
                num_hidden=512,
                dropout_ratio=0,
                weight_init='thomas',
                bias_init='thomas',
                binary_classifier=False,
                augments=0):

        super(EAGLE_M, self).__init__()
        self.nfeat = num_features
        self.nlayer = num_layers
        self.nhid = num_hidden
        self.dropout_ratio = dropout_ratio
        self.gc = nn.ModuleList([GraphConvolution(self.nfeat if i==0 else self.nhid, self.nhid, bias=True, weight_init=weight_init, bias_init=bias_init) for i in range(self.nlayer)])
        self.bn = nn.ModuleList([nn.LayerNorm(self.nhid).double() for i in range(self.nlayer)])
        self.relu = nn.ModuleList([nn.ReLU().double() for i in range(self.nlayer)])
        if not binary_classifier:
            self.fc = nn.Linear(self.nhid + augments, 1).double()
        else:
            if binary_classifier == 'naive':
                self.fc = nn.Linear(self.nhid + augments, 1).double()
            elif binary_classifier == 'oneway' or binary_classifier == 'oneway-hard':
                self.fc = nn.Linear((self.nhid + augments) * 2, 1).double()
            else:
                self.fc = nn.Linear((self.nhid + augments) * 2, 2).double()

            if binary_classifier != 'oneway' and binary_classifier != 'oneway-hard':
                self.final_act = nn.LogSoftmax(dim=1)
            else:
                self.final_act = nn.Sigmoid()

        self.dropout = nn.ModuleList([nn.Dropout(self.dropout_ratio).double() for i in range(self.nlayer)])

        self.binary_classifier = binary_classifier
        
        # Add fully connected layers for zcp tensor
        self.zcp_fc1 = nn.Linear(13, 512).double()  # Assuming 32 as the hidden size for the first layer
        self.zcp_relu1 = nn.ReLU().double()
        self.zcp_fc2 = nn.Linear(512, 16).double()  # Assuming 16 as the hidden size for the second layer
        self.zcp_relu2 = nn.ReLU().double()

        # Add an additional fc-relu layer before the final fc
        self.pre_final_fc = nn.Linear(self.nhid + 16 + augments, self.nhid).double()  # Assuming nhid as the output size
        self.pre_final_relu = nn.ReLU().double()

    def forward_single_model(self, adjacency, features):
        x = self.relu[0](self.bn[0](self.gc[0](adjacency, features)))
        x = self.dropout[0](x)
        for i in range(1,self.nlayer):
            x = self.relu[i](self.bn[i](self.gc[i](adjacency, x)))
            x = self.dropout[i](x)

        return x

    def extract_features(self, adjacency, features, augments=None):
        x = self.forward_single_model(adjacency, features)
        x = x[:,0] # use global node
        if augments is not None:
            x = torch.cat([x, augments], dim=1)
        return x

    def regress(self, features, features2=None):
        if not self.binary_classifier:
            assert features2 is None
            return self.fc(features)

        assert features2 is not None
        if self.binary_classifier == 'naive':
            x1 = self.fc(features)
            x2 = self.fc(features2)
        else:
            x1 = features
            x2 = features2

        x = torch.cat([x1, x2], dim=1)
        if self.binary_classifier != 'naive':
            x = self.fc(x)

        x = self.final_act(x)
        return x

    def forward(self, features, adjacency, zcp, augments=None):
        features, adjacency, zcp = features.double(), adjacency.double(), zcp.double()
        x = self.forward_single_model(adjacency, features)
        x = x[:,0]  # use global node

        # Process zcp tensor through the fully connected layers
        zcp = self.zcp_relu1(self.zcp_fc1(zcp))
        zcp = self.zcp_relu2(self.zcp_fc2(zcp))
        # Concatenate processed zcp tensor with x
        x = torch.cat([x, zcp], dim=1)
        if augments is not None:
            x = torch.cat([x, augments], dim=1)

        # Pass through the additional fc-relu layer
        x = self.pre_final_relu(self.pre_final_fc(x))

        return self.fc(x)
        # # if not self.binary_classifier:
        # x = self.forward_single_model(adjacency, features)
        # x = x[:,0] # use global node
        # if augments is not None:
        #     x = torch.cat([x, augments], dim=1)
        # return self.fc(x)

    def reset_last(self):
        self.fc.reset_parameters()

    def final_params(self):
        return self.fc.parameters()

# =============================================================================

class FullyConnectedNN(nn.Module):
    def __init__(self, layer_sizes):
        super(FullyConnectedNN, self).__init__()

        self.layers = nn.ModuleList()

        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                self.layers.append(nn.BatchNorm1d(layer_sizes[i + 1]))  # Add batch normalization
                self.layers.append(nn.ReLU())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class GIN_Emb_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_hops, num_mlp_layers, readout, emb_dim, dropout, **kwargs):
        super(GIN_Emb_Model, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_hops
        self.emb_dim = emb_dim
        
        self.eps = nn.Parameter(torch.zeros(self.num_layers - 1))
        self.op_emb = nn.Embedding(self.input_dim, self.emb_dim)
        
        self.mlps = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        
        for layer in range(self.num_layers - 1):
            input_dim = self.emb_dim if layer == 0 else hidden_dim
            self.mlps.append(MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        self.fc1 = nn.Linear(self.hidden_dim, self.latent_dim)

    def _encoder(self, ops, adj):
        batch_size, node_num, _ = ops.shape
        x = self.op_emb(ops).squeeze(1)
        
        for l in range(self.num_layers - 1):
            neighbor = torch.matmul(adj.float(), x)
            agg = (1 + self.eps[l]) * x.view(batch_size * node_num, -1) + neighbor.view(batch_size * node_num, -1)
            x = F.relu(self.batch_norms[l](self.mlps[l](agg)).view(batch_size, node_num, -1))
        
        mu = self.fc1(x)
        mu = torch.nn.functional.adaptive_avg_pool2d(mu, (1, 1))
        
        return mu
    
    def forward(self, ops, adj):
        return self._encoder(ops, adj)


class GIN_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_hops, num_mlp_layers, readout, dropout, **kwargs):
        super(GIN_Model, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_hops
        
        self.eps = nn.Parameter(torch.zeros(self.num_layers - 1))
        
        self.mlps = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        
        for layer in range(self.num_layers - 1):
            mlp_input_dim = input_dim if layer == 0 else hidden_dim
            self.mlps.append(MLP(num_mlp_layers, mlp_input_dim, hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        self.fc1 = nn.Linear(self.hidden_dim, self.latent_dim)

    def _encoder(self, ops, adj):
        batch_size, node_num, _ = ops.shape
        x = ops
        
        for l in range(self.num_layers - 1):
            neighbor = torch.matmul(adj.float(), x)
            agg = (1 + self.eps[l]) * x.view(batch_size * node_num, -1) + neighbor.view(batch_size * node_num, -1)
            x = F.relu(self.batch_norms[l](self.mlps[l](agg)).view(batch_size, node_num, -1))
        
        mu = self.fc1(x)
        mu = torch.nn.functional.adaptive_avg_pool2d(mu, (1, 1))
        
        return mu
    
    def forward(self, ops, adj):
        return self._encoder(ops, adj)

class GIN_Emb_Model_NDS(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_hops, num_mlp_layers, readout, emb_dim, dropout, **kwargs):
        super(GIN_Emb_Model_NDS, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_hops
        self.emb_dim = emb_dim
        
        self.op_emb_1 = nn.Embedding(input_dim, self.emb_dim)
        self.op_emb_2 = nn.Embedding(input_dim, self.emb_dim)
        
        self.eps_1 = nn.Parameter(torch.zeros(self.num_layers - 1))
        self.eps_2 = nn.Parameter(torch.zeros(self.num_layers - 1))
        
        self.mlps_1 = self._build_mlp_layers(num_mlp_layers)
        self.mlps_2 = self._build_mlp_layers(num_mlp_layers)
        
        self.batch_norms_1 = self._build_batch_norm_layers()
        self.batch_norms_2 = self._build_batch_norm_layers()
        
        self.fc1_1 = nn.Linear(self.hidden_dim, 64)
        self.fc1_2 = nn.Linear(self.hidden_dim, 64)
        
        self.fc_comb_a = nn.Linear(128, 128)
        self.fc_comb_b = nn.Linear(128, self.latent_dim)

    def _build_mlp_layers(self, num_mlp_layers):
        mlps = torch.nn.ModuleList()
        for layer in range(self.num_layers - 1):
            input_dim = self.emb_dim if layer == 0 else self.hidden_dim
            mlps.append(MLP(num_mlp_layers, input_dim, self.hidden_dim, self.hidden_dim))
        return mlps

    def _build_batch_norm_layers(self):
        return torch.nn.ModuleList([nn.BatchNorm1d(self.hidden_dim) for _ in range(self.num_layers - 1)])

    def _encoder(self, ops, adj, op_emb, eps, mlps, batch_norms, fc):
        batch_size, node_num, _ = ops.shape
        x = op_emb(ops).squeeze(1)
        
        for l in range(self.num_layers - 1):
            neighbor = torch.matmul(adj.float(), x)
            agg = (1 + eps[l]) * x.view(batch_size * node_num, -1) + neighbor.view(batch_size * node_num, -1)
            x = F.relu(batch_norms[l](mlps[l](agg)).view(batch_size, node_num, -1))
        
        return fc(x)

    def forward(self, ops1, adj1, ops2, adj2):
        mu_1 = self._encoder(ops1, adj1, self.op_emb_1, self.eps_1, self.mlps_1, self.batch_norms_1, self.fc1_1)
        mu_2 = self._encoder(ops2, adj2, self.op_emb_2, self.eps_2, self.mlps_2, self.batch_norms_2, self.fc1_2)
        
        mu = torch.cat((mu_1, mu_2), dim=2)
        mu = F.relu(self.fc_comb_a(mu))
        mu = self.fc_comb_b(mu)
        mu = torch.nn.functional.adaptive_avg_pool2d(mu, (1, 1))
        
        return mu


class GIN_Model_NDS(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_hops, num_mlp_layers, readout, dropout, **kwargs):
        super(GIN_Model_NDS, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_hops
        
        self.eps_1 = nn.Parameter(torch.zeros(self.num_layers - 1))
        self.eps_2 = nn.Parameter(torch.zeros(self.num_layers - 1))
        
        self.mlps_1 = self._build_mlp_layers(num_mlp_layers)
        self.mlps_2 = self._build_mlp_layers(num_mlp_layers)
        
        self.batch_norms_1 = self._build_batch_norm_layers()
        self.batch_norms_2 = self._build_batch_norm_layers()
        
        self.fc1_1 = nn.Linear(self.hidden_dim, 64)
        self.fc1_2 = nn.Linear(self.hidden_dim, 64)
        
        self.fc_comb_a = nn.Linear(128, 128)
        self.fc_comb_b = nn.Linear(128, self.latent_dim)

    def _build_mlp_layers(self, num_mlp_layers):
        mlps = torch.nn.ModuleList()
        for layer in range(self.num_layers - 1):
            input_dim = self.input_dim if layer == 0 else self.hidden_dim
            mlps.append(MLP(num_mlp_layers, input_dim, self.hidden_dim, self.hidden_dim))
        return mlps

    def _build_batch_norm_layers(self):
        return torch.nn.ModuleList([nn.BatchNorm1d(self.hidden_dim) for _ in range(self.num_layers - 1)])

    def _encoder_helper(self, ops, adj, eps, mlps, batch_norms, fc):
        batch_size, node_num, _ = ops.shape
        x = ops
        
        for l in range(self.num_layers - 1):
            neighbor = torch.matmul(adj.float(), x)
            agg = (1 + eps[l]) * x.view(batch_size * node_num, -1) + neighbor.view(batch_size * node_num, -1)
            x = F.relu(batch_norms[l](mlps[l](agg)).view(batch_size, node_num, -1))
        
        return fc(x)

    def _encoder(self, ops1, adj1, ops2, adj2):
        mu_1 = self._encoder_helper(ops1, adj1, self.eps_1, self.mlps_1, self.batch_norms_1, self.fc1_1)
        mu_2 = self._encoder_helper(ops2, adj2, self.eps_2, self.mlps_2, self.batch_norms_2, self.fc1_2)
        
        mu = torch.cat((mu_1, mu_2), dim=2)
        mu = F.relu(self.fc_comb_a(mu))
        mu = self.fc_comb_b(mu)
        mu = torch.nn.functional.adaptive_avg_pool2d(mu, (1, 1))
        
        return mu
    
    def forward(self, ops1, adj1, ops2, adj2):
        return self._encoder(ops1, adj1, ops2, adj2)



class GATLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GATLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.matmul(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + torch.transpose(Wh2, -2, -1)
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT_ZCP_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, zcp_dim, zcp_gin_dim, num_hops, num_mlp_layers, readout, dropout, **kwargs):
        super(GAT_ZCP_Model, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.zcp_dim = zcp_dim
        self.zcp_gin_dim = zcp_gin_dim
        self.num_layers = num_hops
        self.emb_dim = 16
        
        self.eps = nn.Parameter(torch.zeros(self.num_layers - 1))
        
        self.mlps = self._build_mlp_layers(num_mlp_layers)
        self.batch_norms = self._build_batch_norm_layers()
        self.gat_layers = self._build_gat_layers()
        
        self.fc1 = nn.Linear(self.hidden_dim, self.zcp_gin_dim)
        self.dropout = nn.Dropout(dropout)  # Dropout layer

        # New layers for processing mu
        self.flatten_dim = input_dim * self.zcp_gin_dim 
        self.fc_mu = nn.Linear(self.flatten_dim, self.zcp_gin_dim)
        self.bn_mu = nn.BatchNorm1d(1)  # 1 for the number of channels
        
        self.fc1_zcp = nn.Linear(self.zcp_dim, 64)
        self.bn1_zcp = nn.BatchNorm1d(64)
        self.fc2_zcp = nn.Linear(64 + self.zcp_gin_dim, 64 + self.zcp_gin_dim)
        self.bn2_zcp = nn.BatchNorm1d(64 + self.zcp_gin_dim)
        self.fcout = nn.Linear(self.zcp_gin_dim + 64, self.latent_dim)

    def _build_mlp_layers(self, num_mlp_layers):
        mlps = torch.nn.ModuleList()
        for layer in range(self.num_layers - 1):
            input_dim = self.input_dim if layer == 0 else self.hidden_dim
            mlps.append(MLP(num_mlp_layers, input_dim, self.hidden_dim, self.hidden_dim))
        return mlps

    def _build_batch_norm_layers(self):
        return torch.nn.ModuleList([nn.BatchNorm1d(self.hidden_dim) for _ in range(self.num_layers - 1)])

    def _build_gat_layers(self):
        gat_layers = torch.nn.ModuleList()
        for layer in range(self.num_layers - 1):
            input_dim = self.input_dim if layer == 0 else self.hidden_dim
            gat_layers.append(GATLayer(input_dim, self.hidden_dim, dropout=0.6, alpha=0.2))
        return gat_layers

    def _encoder(self, ops, adj, zcp):
        batch_size, node_num, _ = ops.shape
        x = ops
        
        for l in range(self.num_layers - 1):
            x = self.gat_layers[l](x, adj)
            x = self.dropout(x)  # Apply dropout after activation
        
        mu = self.fc1(x)
        mu = self.dropout(mu)  # Apply dropout after linear layer
        mu = torch.nn.functional.adaptive_avg_pool2d(mu, (1, self.zcp_gin_dim))
        
        zcp = F.relu(self.bn1_zcp(self.fc1_zcp(zcp)))
        zcp = self.dropout(zcp)  # Apply dropout after activation
        zcp = torch.cat((mu, zcp.unsqueeze(1)), dim=2).squeeze(1)
        zcp = torch.sigmoid(self.fcout(F.relu(self.bn2_zcp(self.fc2_zcp(zcp)))))
        
        return zcp
    
    def forward(self, ops, adj, zcp):
        return self._encoder(ops, adj, zcp)



class GIN_ZCP_Model(nn.Module):
    def __init__(self, op_emb, input_dim, hidden_dim, latent_dim, zcp_dim, zcp_gin_dim, num_hops, num_mlp_layers, readout, dropout, **kwargs):
        super(GIN_ZCP_Model, self).__init__()
        
        self.input_dim = input_dim
        self.op_emb = op_emb
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.zcp_dim = zcp_dim
        self.zcp_gin_dim = zcp_gin_dim
        self.num_layers = num_hops
        self.emb_dim = 16
        
        self.op_emb_table = nn.Embedding(16, 48)

        self.eps = nn.Parameter(torch.zeros(self.num_layers - 1))
        
        self.mlps = self._build_mlp_layers(num_mlp_layers)
        self.batch_norms = self._build_batch_norm_layers()
        
        self.fc1 = nn.Linear(self.hidden_dim, self.zcp_gin_dim)
        self.dropout = nn.Dropout(dropout)  # Dropout layer
        
        # New layers for processing mu
        # self.flatten_dim = input_dim * self.zcp_gin_dim 
        # self.fc_mu = nn.Linear(self.flatten_dim, self.zcp_gin_dim)
        # self.bn_mu = nn.BatchNorm1d(1)  # 1 for the number of channels
        
        self.fc1_zcp = nn.Linear(self.zcp_dim, 64)
        self.bn1_zcp = nn.BatchNorm1d(64)
        self.fc2_zcp = nn.Linear(64 + self.zcp_gin_dim, 64 + self.zcp_gin_dim)
        self.bn2_zcp = nn.BatchNorm1d(64 + self.zcp_gin_dim)
        self.fcout = nn.Linear(self.zcp_gin_dim + 64, self.latent_dim)

        self.new_fc = nn.Linear(self.zcp_gin_dim, 128)
        self.new_fcf = nn.Linear(128, 1)
        
        # Create mlp
        # nn Linear 7 to 64
        self.fc1_zsp = nn.Linear(7, 64)
        # nn Linear 64 to 128
        self.fc2_zsp = nn.Linear(64, 128)
        # nn Linear 128 to 48
        self.fc3_zsp = nn.Linear(128, 48)

    def _build_mlp_layers(self, num_mlp_layers):
        mlps = torch.nn.ModuleList()
        for layer in range(self.num_layers - 1):
            input_dim = 48 if layer == 0 else self.hidden_dim
            mlps.append(MLP(num_mlp_layers, input_dim, self.hidden_dim, self.hidden_dim))
        return mlps

    def _build_batch_norm_layers(self):
        return torch.nn.ModuleList([nn.BatchNorm1d(self.hidden_dim) for _ in range(self.num_layers - 1)])

    def _encoder(self, ops, adj, zsp_, zcp):
        # import pdb; pdb.set_trace()
        if self.op_emb:
            batch_size, node_num = ops.shape
            x = ops
            x = self.op_emb_table(x).squeeze()
        else:
            batch_size, node_num, _ = ops.shape
            x = ops

        zsp_ = self.fc3_zsp(F.relu(self.fc2_zsp(F.relu(self.fc1_zsp(zsp_)))))

        x += 0.1*zsp_

        for l in range(self.num_layers - 1):
            neighbor = torch.matmul(adj.float(), x)
            agg = (1 + self.eps[l]) * x.view(batch_size * node_num, -1) + neighbor.view(batch_size * node_num, -1)
            x = F.relu(self.batch_norms[l](self.mlps[l](agg)).view(batch_size, node_num, -1))
            x = self.dropout(x)  # Apply dropout after activation
        
        mu = self.fc1(x)
        mu = self.dropout(mu)  # Apply dropout after linear layer
        mu = torch.nn.functional.adaptive_avg_pool2d(mu, (1, self.zcp_gin_dim))
        mu = self.new_fc(mu)
        mu = self.new_fcf(F.relu(mu))
        return mu 
        # zcp = F.relu(self.bn1_zcp(self.fc1_zcp(zcp)))
        # zcp = self.dropout(zcp)  # Apply dropout after activation
        # zcp = torch.cat((mu, zcp.unsqueeze(1)), dim=2).squeeze(1)
        # zcp = torch.sigmoid(self.fcout(F.relu(self.bn2_zcp(self.fc2_zcp(zcp)))))
        # return zcp
    
    def forward(self, ops, adj, zsp_, zcp):
        return self._encoder(ops, adj, zsp_, zcp)

# class GIN_ZCP_Final_Model(nn.Module):
#     def __init__(self, op_emb, input_dim, hidden_dim, latent_dim, zcp_dim, zcp_gin_dim, num_hops, num_mlp_layers, readout, dropout, **kwargs):
#         super(GIN_ZCP_Model, self).__init__()
        
#         self.input_dim = input_dim
#         self.op_emb = op_emb
#         self.hidden_dim = hidden_dim
#         self.latent_dim = latent_dim
#         self.zcp_dim = zcp_dim
#         self.zcp_gin_dim = zcp_gin_dim
#         self.num_layers = num_hops
#         self.emb_dim = 16
        
#         self.op_emb_table = nn.Embedding(16, 48)

#         self.eps = nn.Parameter(torch.zeros(self.num_layers - 1))
        
#         self.mlps = self._build_mlp_layers(num_mlp_layers)
#         self.batch_norms = self._build_batch_norm_layers()
        
#         self.fc1 = nn.Linear(self.hidden_dim, self.zcp_gin_dim)
#         self.dropout = nn.Dropout(dropout)  # Dropout layer
        
#         # New layers for processing mu
#         self.flatten_dim = input_dim * self.zcp_gin_dim 
#         self.fc_mu = nn.Linear(self.flatten_dim, self.zcp_gin_dim)
#         self.bn_mu = nn.BatchNorm1d(1)  # 1 for the number of channels
        
#         self.fc1_zcp = nn.Linear(self.zcp_dim, 64)
#         self.bn1_zcp = nn.BatchNorm1d(64)
#         self.fc2_zcp = nn.Linear(64 + self.zcp_gin_dim, 64 + self.zcp_gin_dim)
#         self.bn2_zcp = nn.BatchNorm1d(64 + self.zcp_gin_dim)
#         self.fcout = nn.Linear(self.zcp_gin_dim + 64, self.latent_dim)
        
#         # Create mlp
#         # nn Linear 7 to 64
#         self.fc1_zsp = nn.Linear(7, 64)
#         # nn Linear 64 to 128
#         self.fc2_zsp = nn.Linear(64, 128)
#         # nn Linear 128 to 48
#         self.fc3_zsp = nn.Linear(128, 48)



#     def _build_mlp_layers(self, num_mlp_layers):
#         mlps = torch.nn.ModuleList()
#         for layer in range(self.num_layers - 1):
#             input_dim = self.input_dim if layer == 0 else self.hidden_dim
#             mlps.append(MLP(num_mlp_layers, input_dim, self.hidden_dim, self.hidden_dim))
#         return mlps

#     def _build_batch_norm_layers(self):
#         return torch.nn.ModuleList([nn.BatchNorm1d(self.hidden_dim) for _ in range(self.num_layers - 1)])

#     def _encoder(self, ops, adj, zsp_, zcp):
#         batch_size, node_num, _ = ops.shape
#         x = ops

#         if self.op_emb:
#             x = self.op_emb_table(x).squeeze()
        
#         zsp_ = self.fc3_zsp(F.relu(self.fc2_zsp(F.relu(self.fc1_zsp(zsp_)))))

#         x += 0.1*zsp_

#         for l in range(self.num_layers - 1):
#             neighbor = torch.matmul(adj.float(), x)
#             agg = (1 + self.eps[l]) * x.view(batch_size * node_num, -1) + neighbor.view(batch_size * node_num, -1)
#             x = F.relu(self.batch_norms[l](self.mlps[l](agg)).view(batch_size, node_num, -1))
#             x = self.dropout(x)  # Apply dropout after activation
        
#         mu = self.fc1(x)
#         mu = self.dropout(mu)  # Apply dropout after linear layer
#         mu = torch.nn.functional.adaptive_avg_pool2d(mu, (1, self.zcp_gin_dim))
        
#         zcp = F.relu(self.bn1_zcp(self.fc1_zcp(zcp)))
#         zcp = self.dropout(zcp)  # Apply dropout after activation
#         zcp = torch.cat((mu, zcp.unsqueeze(1)), dim=2).squeeze(1)
#         zcp = torch.sigmoid(self.fcout(F.relu(self.bn2_zcp(self.fc2_zcp(zcp)))))
        
#         return zcp
    
#     def forward(self, ops, adj, zcp):
#         return self._encoder(ops, adj, zcp)



class GIN_ZCP_Model_NDS(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, zcp_dim, zcp_gin_dim, num_hops, 
                 num_mlp_layers, readout, dropout, **kwargs):
        super(GIN_ZCP_Model_NDS, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.zcp_dim = zcp_dim
        self.zcp_gin_dim = zcp_gin_dim
        self.num_layers = num_hops
        self.flatten_dim = input_dim * self.zcp_gin_dim 
        
        self.eps_1 = nn.Parameter(torch.zeros(self.num_layers - 1))
        self.mlps_1 = self._build_mlp_layers(num_mlp_layers, input_dim)
        self.batch_norms_1 = self._build_batch_norm_layers()
        self.fc1_1 = nn.Linear(self.hidden_dim, self.zcp_gin_dim)

        self.eps_2 = nn.Parameter(torch.zeros(self.num_layers - 1))
        self.mlps_2 = self._build_mlp_layers(num_mlp_layers, input_dim)
        self.batch_norms_2 = self._build_batch_norm_layers()
        self.fc1_2 = nn.Linear(self.hidden_dim, self.zcp_gin_dim)

        self.fc_comb_a = nn.Linear(2*64, 128)
        self.fc_comb_b = nn.Linear(128, self.zcp_gin_dim)
        self.fc_mu = nn.Linear(self.flatten_dim, self.zcp_gin_dim)
        self.bn_mu = nn.BatchNorm1d(1)  # 1 for the number of channels

        self.fc1_zcp = nn.Linear(self.zcp_dim, 64)
        self.bn1_zcp = nn.BatchNorm1d(64)
        self.fc2_zcp = nn.Linear(64 + self.zcp_gin_dim, 64 + self.zcp_gin_dim)
        self.bn2_zcp = nn.BatchNorm1d(64 + self.zcp_gin_dim)
        self.fcout = nn.Linear(self.zcp_gin_dim + 64, self.latent_dim)

    def _build_mlp_layers(self, num_mlp_layers, input_dim):
        mlps = torch.nn.ModuleList()
        for layer in range(self.num_layers - 1):
            layer_input_dim = self.emb_dim if layer == 0 else self.hidden_dim
            mlps.append(MLP(num_mlp_layers, layer_input_dim, self.hidden_dim, self.hidden_dim))
        return mlps

    def _build_batch_norm_layers(self):
        return torch.nn.ModuleList([nn.BatchNorm1d(self.hidden_dim) for _ in range(self.num_layers - 1)])

    def _encoder(self, ops1, adj1, ops2, adj2, zcp):
        mu_1 = self._process_ops(ops1, adj1, self.mlps_1, self.batch_norms_1, self.fc1_1)
        mu_2 = self._process_ops(ops2, adj2, self.mlps_2, self.batch_norms_2, self.fc1_2)
        
        mu = torch.cat((mu_1, mu_2), dim=2)
        mu = F.relu(self.fc_comb_a(mu))
        mu = F.relu(self.fc_comb_b(mu))
        mu = torch.nn.functional.adaptive_avg_pool2d(mu, (self.input_dim, self.zcp_gin_dim))
        mu = mu.view(-1, self.flatten_dim)  
        mu = self.fc_mu(mu).unsqueeze(1)  
        
        zcp = F.relu(self.fc1_zcp(zcp))
        zcp = torch.cat((mu, zcp.unsqueeze(1)), dim=2).squeeze(1)
        zcp = torch.sigmoid(self.fcout(F.relu(self.fc2_zcp(zcp))))
        
        return zcp

    def _process_ops(self, ops, adj, mlps, batch_norms, fc):
        batch_size, node_num, _ = ops.shape
        x = ops
        for l in range(self.num_layers - 1):
            neighbor = torch.matmul(adj.float(), x)
            agg = (1 + self.eps_2[l]) * x.view(batch_size * node_num, -1) + neighbor.view(batch_size * node_num, -1)
            x = F.relu(batch_norms[l](mlps[l](agg)).view(batch_size, node_num, -1))
        return fc(x)

    def forward(self, ops1, adj1, ops2, adj2, zcp):
        return self._encoder(ops1, adj1, ops2, adj2, zcp)


class GIN_Emb_ZCP_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, zcp_dim, zcp_gin_dim, num_hops, 
                 num_mlp_layers, readout, emb_dim, dropout, **kwargs):
        super(GIN_Emb_ZCP_Model, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.zcp_dim = zcp_dim
        self.zcp_gin_dim = zcp_gin_dim
        self.emb_dim = emb_dim
        self.num_layers = num_hops
        
        self.op_emb = nn.Embedding(input_dim, self.emb_dim)
        self.eps = nn.Parameter(torch.zeros(self.num_layers - 1))
        
        self.mlps = self._build_mlp_layers(num_mlp_layers)
        self.batch_norms = self._build_batch_norm_layers()
        
        self.fc1 = nn.Linear(self.hidden_dim, self.zcp_gin_dim)
        self.fc1_zcp = nn.Linear(self.zcp_dim, 64)
        self.bn1_zcp = nn.BatchNorm1d(64)
        self.fc2_zcp = nn.Linear(64 + self.zcp_gin_dim, 64 + self.zcp_gin_dim)
        self.bn2_zcp = nn.BatchNorm1d(64 + self.zcp_gin_dim)
        self.fcout = nn.Linear(self.zcp_gin_dim + 64, self.latent_dim)

    def _build_mlp_layers(self, num_mlp_layers):
        mlps = torch.nn.ModuleList()
        for layer in range(self.num_layers - 1):
            input_dim = self.emb_dim if layer == 0 else self.hidden_dim
            mlps.append(MLP(num_mlp_layers, input_dim, self.hidden_dim, self.hidden_dim))
        return mlps

    def _build_batch_norm_layers(self):
        return torch.nn.ModuleList([nn.BatchNorm1d(self.hidden_dim) for _ in range(self.num_layers - 1)])

    def _encoder(self, ops, adj, zcp):
        batch_size, node_num, _ = ops.shape
        x = self.op_emb(ops).squeeze(1)
        
        for l in range(self.num_layers - 1):
            neighbor = torch.matmul(adj.float(), x)
            agg = (1 + self.eps[l]) * x.view(batch_size * node_num, -1) + neighbor.view(batch_size * node_num, -1)
            x = F.relu(self.batch_norms[l](self.mlps[l](agg)).view(batch_size, node_num, -1))
        
        mu = self.fc1(x)
        mu = torch.nn.functional.adaptive_avg_pool2d(mu, (1, self.zcp_gin_dim))
        
        zcp = F.relu(self.bn1_zcp(self.fc1_zcp(zcp)))
        zcp = torch.cat((mu, zcp.unsqueeze(1)), dim=2).squeeze(1)
        zcp = self.fcout(F.relu(self.bn2_zcp(self.fc2_zcp(zcp))))
        
        return zcp
    
    def forward(self, ops, adj, zcp):
        return self._encoder(ops, adj, zcp)


# This one has not been cleaned (EmbZCPNDS)
class GIN_Emb_ZCP_Model_NDS(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, zcp_dim, zcp_gin_dim, num_hops, num_mlp_layers, readout, emb_dim,
                 dropout, **kwargs):
        super(GIN_Emb_ZCP_Model_NDS, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.zcp_dim = zcp_dim
        self.zcp_gin_dim = zcp_gin_dim
        self.num_layers = num_hops
        self.emb_dim = emb_dim
        self.op_emb_1 = nn.Embedding(input_dim, self.emb_dim)
        self.eps_1 = nn.Parameter(torch.zeros(self.num_layers - 1))
        self.mlps_1 = torch.nn.ModuleList()
        self.batch_norms_1 = torch.nn.ModuleList()
        for layer in range(self.num_layers - 1):
            if layer == 0:
                self.mlps_1.append(MLP(num_mlp_layers, self.emb_dim, hidden_dim, hidden_dim))
            else:
                self.mlps_1.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))
            self.batch_norms_1.append(nn.BatchNorm1d(hidden_dim))
        self.fc1_1 = nn.Linear(self.hidden_dim, 64)
        self.op_emb_2 = nn.Embedding(input_dim, self.emb_dim)
        self.mlps_2 = torch.nn.ModuleList()
        self.eps_2 = nn.Parameter(torch.zeros(self.num_layers - 1))
        self.batch_norms_2 = torch.nn.ModuleList()
        for layer in range(self.num_layers - 1):
            if layer == 0:
                self.mlps_2.append(MLP(num_mlp_layers, self.emb_dim, hidden_dim, hidden_dim))
            else:
                self.mlps_2.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))
            self.batch_norms_2.append(nn.BatchNorm1d(hidden_dim))
        self.fc1_2 = nn.Linear(self.hidden_dim, 64)

        self.fc_comb_a = nn.Linear(192, 128)
        self.fc_comb_b = nn.Linear(128, self.zcp_gin_dim)

        self.fc1_zcp = nn.Linear(self.zcp_dim, 64)
        self.bn1_zcp = nn.BatchNorm1d(64)
        self.fc2_zcp = nn.Linear(64 + self.zcp_gin_dim, 64 + self.zcp_gin_dim)
        self.bn2_zcp = nn.BatchNorm1d(64 + self.zcp_gin_dim)
        
        self.fcout = nn.Linear(self.zcp_gin_dim + 64, self.latent_dim)

    def _encoder(self, ops1, adj1, ops2, adj2, zcp):
        batch_size_1, node_num_1, opt_num_1 = ops1.shape
        x_1 = self.op_emb_1(ops1).squeeze(1)
        for l in range(self.num_layers - 1):
            neighbor = torch.matmul(adj1.float(), x_1)
            agg = (1 + self.eps_2[l]) * x_1.view(batch_size_1 * node_num_1, -1) \
                  + neighbor.view(batch_size_1 * node_num_1, -1)
            x_1 = F.relu(self.batch_norms_1[l](self.mlps_1[l](agg)).view(batch_size_1, node_num_1, -1))
        mu_1 = self.fc1_1(x_1)
        batch_size_2, node_num_2, opt_num_2 = ops2.shape
        x_2 = self.op_emb_2(ops2).squeeze(1)
        for l in range(self.num_layers - 1):
            neighbor = torch.matmul(adj2.float(), x_2)
            agg = (1 + self.eps_2[l]) * x_2.view(batch_size_2 * node_num_2, -1) \
                  + neighbor.view(batch_size_2 * node_num_2, -1)
            x_2 = F.relu(self.batch_norms_2[l](self.mlps_2[l](agg)).view(batch_size_2, node_num_2, -1))
        mu_2 = self.fc1_2(x_2)
        
        mu = torch.cat((mu_1, mu_2, zcp.unsqueeze(1)), dim=2)
        mu = F.relu(self.fc_comb_a(mu))
        mu = self.fc_comb_b(mu)
        mu = torch.nn.functional.adaptive_avg_pool2d(mu, (1, self.zcp_gin_dim))

        zcp = F.relu(self.bn1_zcp(self.fc1_zcp(zcp)))
        zcp = torch.cat((mu, zcp.unsqueeze(1)), dim=2).squeeze(1)
        zcp = self.fcout(F.relu(self.bn2_zcp(self.fc2_zcp(zcp))))
        return zcp
    def forward(self, ops1, adj1, ops2, adj2, zcp):
        mu = self._encoder(ops1, adj1, ops2, adj2, zcp)
        return mu

class Reconstructed_Loss(object):
    def __init__(self, w_ops=1.0, w_adj=1.0, loss_ops=None, loss_adj=None):
        super().__init__()
        self.w_ops = w_ops
        self.w_adj = w_adj
        self.loss_ops = loss_ops
        self.loss_adj = loss_adj

    def __call__(self, inputs, targets):
        ops_recon, adj_recon = inputs[0], inputs[1]
        ops, adj = targets[0], targets[1]
        loss_ops = self.loss_ops(ops_recon, ops)
        loss_adj = self.loss_adj(adj_recon, adj)
        loss = self.w_ops * loss_ops + self.w_adj * loss_adj
        return loss


class VAEReconstructed_Loss(object):
    def __init__(self, w_ops=1.0, w_adj=1.0, loss_ops=None, loss_adj=None):
        super().__init__()
        self.w_ops = w_ops
        self.w_adj = w_adj
        self.loss_ops = loss_ops
        self.loss_adj = loss_adj

    def __call__(self, inputs, targets, mu, logvar):
        ops_recon, adj_recon = inputs[0], inputs[1]
        ops, adj = targets[0], targets[1]
        loss_ops = self.loss_ops(ops_recon, ops)
        loss_adj = self.loss_adj(adj_recon, adj)
        loss = self.w_ops * loss_ops + self.w_adj * loss_adj
        KLD = -0.5 / (ops.shape[0] * ops.shape[1]) * torch.mean(torch.sum(1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 2))
        return loss + KLD


class WeightedLoss(nn.MSELoss):
    def __init__(self, weight=None):
        super(WeightedLoss, self).__init__()
        self.weight = weight


    def forward(self, inputs, targets):
        res = (torch.exp(inputs)-1.0) * F.mse_loss(inputs, targets, size_average=False)
        return torch.mean(res, dim=0) / (self.weight - 1)



class LinearModel(nn.Module):
    def __init__(self, input_dim, hid_dim, activation=F.relu):
        super(LinearModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, 1)
        self.activation = activation

    def forward(self, x):
        h = self.activation(self.fc1(x))
        y = torch.sigmoid(self.fc2(h))
        return y

class DecoderNN(object):
    def __init__(self, model, ops, adj, cfg):
        print('Initializing NN decoder')
        t_s = time.time()
        self.model = model
        self.ops = ops
        self.adj = adj
        self.cfg = cfg
        with torch.no_grad():
            adj_prep, ops_prep, _ = preprocessing(self.adj, self.ops, **self.cfg['prep'])
            self.embedding = self.model.encoder(ops_prep, adj_prep)
        assert len(self.embedding.shape) == 3
        print('Using {} seconds to initialize NN decoder'.format(time.time()-t_s))

    def find_NN(self, ops, adj, ind, k = 10):
        assert len(ops.shape)==3
        ind_t1_list = []
        ind_tk_list = []
        with torch.no_grad():
            adj_prep, ops_prep, _ = preprocessing(adj, ops, **self.cfg['prep'])
            embeddings = self.model.encoder(ops_prep, adj_prep)
            for e in embeddings:
                dist = torch.sum( (self.embedding - e) ** 2, dim=[1,2])
                _, ind_t1 = torch.topk(dist, 1, largest=False)
                _, ind_tk = torch.topk(dist, k, largest=False)
                ind_t1_list.append(ind_t1.item())
                ind_tk_list.append(ind_tk.tolist())
        op_recon, adj_recon = self.ops[ind_t1_list], self.adj[ind_t1_list]
        op_recon_tk, adj_recon_tk = self.ops[ind_t1_list], self.adj[ind_t1_list]
        return op_recon, adj_recon, op_recon_tk, adj_recon_tk, ind_t1_list, ind_tk_list
