import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
import math

class DenseGraphFlow(nn.Module):

    def __init__(self, in_features, out_features, op_emb_dim):
        super(DenseGraphFlow, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.op_emb_dim = op_emb_dim

        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.op_attention = nn.Linear(op_emb_dim, out_features)
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs, adj, op_emb):
        adj_aug = adj
        support = torch.matmul(inputs, self.weight)
        output = torch.sigmoid(self.op_attention(op_emb)) * torch.matmul(adj_aug, support) + support
        return output + self.bias

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GIN_Model():
    def __init__(
            self,
            num_zcps = 13,
            vertices = 7,
            none_op_ind = 3,
            op_embedding_dim = 48,
            node_embedding_dim = 48,
            zcp_embedding_dim = 48,
            hid_dim = 96,
            gcn_out_dims = [128, 128, 128, 128, 128],
            mlp_dims = [200, 200, 200],
            dropout = 0.0,
            num_time_steps = 1,
            fb_conversion_dims = [128, 128],
            backward_gcn_out_dims = [128, 128, 128, 128, 128],
            updateopemb_dims = [128],
            updateopemb_scale = 0.1,
            nn_emb_dims = 128,
            input_zcp = False,
            zcp_embedder_dims = [128, 128],
    ):
        super(GIN_Model, self).__init__()
        if num_time_steps > 1:
            raise NotImplementedError
        self.num_zcps = num_zcps
        self.op_embedding_dim = op_embedding_dim
        self.node_embedding_dim = node_embedding_dim
        self.zcp_embedding_dim = zcp_embedding_dim
        self.hid_dim = hid_dim
        self.gcn_out_dims = gcn_out_dims
        self.dropout = dropout
        self.num_time_steps = num_time_steps
        self.fb_conversion_dims = fb_conversion_dims
        self.backward_gcn_out_dims = backward_gcn_out_dims
        self.updateopemb_dims = updateopemb_dims
        self.updateopemb_scale = updateopemb_scale
        self.mlp_dims = mlp_dims
        self.nn_emb_dims = nn_emb_dims
        self.input_zcp = input_zcp
        self.zcp_embedder_dims = zcp_embedder_dims
        self.vertices = vertices
        self.none_op_ind = none_op_ind
        
        self.mlp_dropout = 0.1
        self.training = True

        # regression MLP
        self.mlp = []
        reg_inp_dims = self.nn_emb_dims
        if self.input_zcp:
            reg_inp_dims += self.zcp_embedding_dim
        for hidden_size in self.mlp_dims:
            self.mlp.append(nn.Sequential(
                nn.Linear(dim, hidden_size),
                nn.ReLU(inplace=False),
                nn.Dropout(p=self.mlp_dropout)))
            dim = hidden_size
        self.mlp.append(nn.Linear(dim, 1))
        self.mlp = nn.Sequential(*self.mlp)
        
        # op embeddings
        self.input_node_emb = nn.Embedding(1, self.node_embedding_dim)
        self.other_node_emb = nn.Parameter(
            torch.zeros(1, self.node_embedding_dim), requires_grad = True
        )
        self.input_op_emb = nn.Parameter(
            torch.zeros(1, self.op_embedding_dim),
            requires_grad = False
        )
        self.op_emb = nn.Embedding(32, self.op_embedding_dim)
        self.output_op_emb = nn.Embedding(1, self.op_embedding_dim)
        self.x_hidden = nn.Linear(self.node_embedding_dim, self.hid_dim)

        # gcn
        self.gcns = []
        in_dim = self.hid_dim
        for dim in self.gcn_out_dims:
            self.gcns.append(
                DenseGraphFlow(
                    in_dim, dim, self.op_embedding_dim # potential issue
                )
            )
            in_dim = dim
        self.gcns = nn.ModuleList(self.gcns)
        self.num_gcn_layers = len(self.gcns)
        self.out_dim = in_dim

        # zcp
        self.zcp_embedder = []
        zin_dim = self.num_zcps
        for zcp_emb_dim in self.zcp_embedder_dims:
            self.zcp_embedder.append(
                nn.Sequential(
                    nn.Linear(zin_dim, zcp_emb_dim),
                    nn.ReLU(inplace=False),
                    nn.Dropout(p=self.mlp_dropout)
                )
            )
            zin_dim = zcp_emb_dim
        self.zcp_embedder.append(nn.Linear(zin_dim, self.zcp_embedding_dim))
        self.zcp_embedder = nn.Sequential(*self.zcp_embedder)
        
    def embed_and_transform_arch(self, archs):
        adjs = self.input_op_emb.new([arch.T for arch in archs[0]])
        op_inds = self.input_op_emb.new([arch for arch in archs[1]]).long()
        op_embs = self.op_emb(op_inds)
        b_size = op_embs.shape[0]
        op_embs = torch.cat(
            (
                self.input_op_emb.unsqueeze(0).repeat([b_size, 1, 1]),
                op_embs,
                self.output_op_emb.weight.unsqueeze(0).repeat([b_size, 1, 1])
            ), dim = 1
        )
        node_embs = torch.cat(
            (
                self.input_node_emb.weight.unsqueeze(0).repeat([b_size, 1, 1]),
                self.other_node_emb.unsqueeze(0).repeat([b_size, self.vertices - 1, 1])
            ), dim = 1
        )
        x = self.x_hidden(node_embs)
        return adjs, x, op_embs, op_inds

    def _forward_pass(self, x, adjs, auged_op_emb):
        # --- forward pass ---
        y = x
        for i_layer, gcn in enumerate(self.gcns):
            y = gcn(y, adjs, auged_op_emb)
            if self.use_bn:
                shape_y = y.shape
                y = self.bns[i_layer](y.reshape(shape_y[0], -1, shape_y[-1])).reshape(
                    shape_y
                )
            if i_layer != self.num_gcn_layers - 1:
                y = F.relu(y)
            y = F.dropout(y, self.dropout, training = self.training)
        return y

    # def _backward_pass(self, y, adjs, zs_as_l, auged_op_emb):
    # If activating, define b_gcns, fb_conversion, b_bns
    #     # --- backward pass ---
    #     b_info = y[:, -1:, :]
    #     b_info = self.fb_conversion(b_info)
    #     b_info = torch.cat(
    #         (
    #             torch.zeros([y.shape[0], self.vertices - 1, b_info.shape[-1]], device = y.device),
    #             b_info
    #         ),
    #         dim = 1
    #     )
    #     # start backward flow
    #     b_adjs = adjs.transpose(1, 2)
    #     b_y = b_info
    #     for i_layer, gcn in enumerate(self.b_gcns):
    #         b_y = gcn(b_y, b_adjs, auged_op_emb)
    #         if self.b_use_bn:
    #             shape_y = b_y.shape
    #             b_y = self.b_bns[i_layer](b_y.reshape(shape_y[0], -1, shape_y[-1]))\
    #                         .reshape(shape_y)
    #         if i_layer != self.num_b_gcn_layers - 1:
    #             b_y = F.relu(b_y)
    #             b_y = F.dropout(b_y, self.dropout, training = self.training)
    #     return b_y

    # def _update_op_emb(self, y, b_y, op_emb, concat_op_emb_mask):
    # If activating, define updateop_embedder
    #     # --- UpdateOpEmb ---
    #     in_embedding = torch.cat(
    #         (
    #             op_emb.detach(),
    #             y.detach(),
    #             b_y
    #         ),
    #         dim = -1)
    #     update = self.updateop_embedder(in_embedding)
    #     op_emb = op_emb + self.updateopemb_scale * update
    #     return op_emb

    def _final_process(self, y, op_inds):
        y = y[:, 1:, :]
        y = torch.cat(
            (
                y[:, :-1, :] * (op_inds != self.none_op_ind)[:, :, None].to(torch.float32),
                y[:, -1:, :],
            ),
            dim = 1
        )
        y = torch.mean(y, dim = 1)
        return y

    def forward(self, x_ops, x_adj, zcp):
        archs = [[np.asarray(x.cpu()) for x in x_adj], [np.asarray(x.cpu()) for x in x_ops]]
        zcp = zcp.cpu()
        adjs, x, op_emb, op_inds = self.embed_and_transform_arch(archs)
        y = self._forward_pass(x, adjs, op_emb)
        y = self._final_process(y, op_inds)
        if self.input_zcp:
            zcp = self.zcp_embedder(zcp)
            y = torch.cat((y, zcp), dim = -1)
        y = self.mlp(y)
        return y