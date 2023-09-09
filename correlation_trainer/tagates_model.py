import abc
import copy
import os
import re
import random
import collections
import itertools
import yaml
from typing import List

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import math


class DenseGraphFlow(nn.Module):

    def __init__(self, in_features, out_features, op_emb_dim,
                 has_attention=True, plus_I=False, normalize=False, bias=True,
                 residual_only=None, reverse=False):
        super(DenseGraphFlow, self).__init__()

        self.plus_I = plus_I
        self.normalize = normalize
        self.in_features = in_features
        self.out_features = out_features
        self.op_emb_dim = op_emb_dim
        self.residual_only = residual_only
        self.reverse = reverse

        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if has_attention:
            self.op_attention = nn.Linear(op_emb_dim, out_features)
        else:
            assert self.op_emb_dim == self.out_features
            self.op_attention = nn.Identity()
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs, adj, op_emb):
        if self.plus_I:
            adj_aug = adj + torch.eye(adj.shape[-1], device=adj.device).unsqueeze(0)
            if self.normalize:
                degree_invsqrt = 1. / adj_aug.sum(dim=-1).float().sqrt()
                degree_norm = degree_invsqrt.unsqueeze(2) * degree_invsqrt.unsqueeze(1)
                adj_aug = degree_norm * adj_aug
        else:
            adj_aug = adj
        support = torch.matmul(inputs, self.weight)
        if self.residual_only is None:
            # use residual
            output = torch.sigmoid(self.op_attention(op_emb)) * torch.matmul(adj_aug, support) \
                     + support
        else:
            # residual only the first `self.residual_only` nodes
            if self.residual_only == 0:
                residual = 0
            else:
                if self.reverse:
                    residual = torch.cat(
                        (torch.zeros([support.shape[0], support.shape[1] - self.residual_only,
                                      support.shape[2]], device=support.device),
                         support[:, -self.residual_only:, :]),
                        dim=1)
                else:
                    residual = torch.cat(
                        (support[:, :self.residual_only, :],
                         torch.zeros([support.shape[0], support.shape[1] - self.residual_only,
                                      support.shape[2]], device=support.device)),
                        dim=1)
            output = torch.sigmoid(self.op_attention(op_emb)) * torch.matmul(adj_aug, support)\
                     + residual

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GIN_ZCP_Model():
    """
    Implement of TA-GATES architecture embedder on NAS-Bench-101.
    """
    NAME = "nb101-fbflow"

    def __init__(
        self,
        op_embedding_dim: int = 48,
        node_embedding_dim: int = 48,
        hid_dim: int = 96,
        gcn_out_dims = [128, 128, 128, 128, 128],
        share_op_attention: bool = False,
        other_node_zero: bool = False,
        gcn_kwargs: dict = None,
        use_bn: bool = False,
        use_global_node: bool = False,
        use_final_only: bool = False,
        input_op_emb_trainable: bool = False,
        dropout: float = 0.,

        ## newly added
        # construction (tagates)
        num_time_steps: int = 2,
        fb_conversion_dims = [128, 128],
        backward_gcn_out_dims = [128, 128, 128, 128, 128],
        updateopemb_method: str = "concat_ofb", # concat_ofb, concat_fb, concat_b
        updateopemb_dims = [128],
        updateopemb_scale: float = 0.1,
        b_use_bn: bool = False,
        # construction (l): concat arch-level zeroshot as l
        concat_arch_zs_as_l_dimension=None,
        concat_l_layer: int = 0,
        # construction (symmetry breaking)
        symmetry_breaking_method: str = "param_zs_add", # None, "random", "param_zs", "param_zs_add"
        concat_param_zs_as_opemb_dimension = 7,
        concat_param_zs_as_opemb_mlp = [64, 128],
        concat_param_zs_as_opemb_scale: float = 0.1,

        # gradident flow configurations
        detach_vinfo: bool = False,
        updateopemb_detach_opemb: bool = True,
        updateopemb_detach_finfo: bool = True,

        mask_nonparametrized_ops: bool = False,
        schedule_cfg = None
    ) -> None:
        super(GIN_ZCP_Model, self).__init__()

        # configs
        self.op_embedding_dim = op_embedding_dim
        self.node_embedding_dim = node_embedding_dim
        self.hid_dim = hid_dim
        self.gcn_out_dims = gcn_out_dims
        self.dropout = dropout
        self.use_bn = use_bn
        self.use_final_only = use_final_only
        self.use_global_node = use_global_node
        self.share_op_attention = share_op_attention
        self.input_op_emb_trainable = input_op_emb_trainable
        self.vertices = 7
        self.num_op_choices = 4
        self.none_op_ind = 3
        self.training = True
        self.mlp_dropout = 0.1

        # newly added
        self.detach_vinfo = detach_vinfo
        self.num_time_steps = num_time_steps
        self.fb_conversion_dims = fb_conversion_dims
        self.backward_gcn_out_dims = backward_gcn_out_dims
        self.b_use_bn = b_use_bn
        self.updateopemb_method = updateopemb_method
        self.updateopemb_detach_opemb = updateopemb_detach_opemb
        self.updateopemb_detach_finfo = updateopemb_detach_finfo
        self.updateopemb_dims = updateopemb_dims
        self.updateopemb_scale = updateopemb_scale
        # concat arch-level zs as l
        self.concat_arch_zs_as_l_dimension = concat_arch_zs_as_l_dimension
        self.concat_l_layer = concat_l_layer
        if self.concat_arch_zs_as_l_dimension is not None:
            assert self.concat_l_layer < len(self.fb_conversion_dims)

        # symmetry breaking
        self.symmetry_breaking_method = symmetry_breaking_method
        self.concat_param_zs_as_opemb_dimension = concat_param_zs_as_opemb_dimension
        assert self.symmetry_breaking_method in {None, "param_zs", "random", "param_zs_add"}
        self.concat_param_zs_as_opemb_scale = concat_param_zs_as_opemb_scale
        self.mlp = []
        mlp_hiddens=(200, 200, 200)
        dim = 128
        mlp_dropout = self.mlp_dropout
        for hidden_size in mlp_hiddens:
            self.mlp.append(nn.Sequential(
                nn.Linear(dim, hidden_size),
                nn.ReLU(inplace=False),
                nn.Dropout(p=mlp_dropout)))
            dim = hidden_size
        self.mlp.append(nn.Linear(dim, 1))
        self.mlp = nn.Sequential(*self.mlp)
        if self.symmetry_breaking_method == "param_zs_add":
            in_dim = self.concat_param_zs_as_opemb_dimension
            self.param_zs_embedder = []
            for embedder_dim in concat_param_zs_as_opemb_mlp:
                self.param_zs_embedder.append(nn.Linear(in_dim, embedder_dim))
                self.param_zs_embedder.append(nn.ReLU(inplace = False))
                in_dim = embedder_dim
            self.param_zs_embedder.append(nn.Linear(in_dim, self.op_embedding_dim))
            self.param_zs_embedder = nn.Sequential(*self.param_zs_embedder)

        self.mask_nonparametrized_ops = mask_nonparametrized_ops

        self.input_node_emb = nn.Embedding(1, self.node_embedding_dim)
        # Maybe separate output node?
        self.other_node_emb = nn.Parameter(
            torch.zeros(1, self.node_embedding_dim), requires_grad = not other_node_zero
        )

        # the last embedding is the output op emb
        self.input_op_emb = nn.Parameter(
            torch.zeros(1, self.op_embedding_dim),
            requires_grad = self.input_op_emb_trainable
        )
        self.op_emb = nn.Embedding(2*self.num_op_choices, self.op_embedding_dim)
        self.output_op_emb = nn.Embedding(1, self.op_embedding_dim)
        if self.use_global_node:
            self.global_op_emb = nn.Parameter(
                torch.zeros((1, self.op_embedding_dim)))
            self.vertices += 1

        self.x_hidden = nn.Linear(self.node_embedding_dim, self.hid_dim)

        if self.share_op_attention:
            assert (
                len(np.unique(self.gcn_out_dims)) == 1
            ), "If share op attention, all the gcn-flow layers should have the same dimension"
            self.op_attention = nn.Linear(
                self.op_embedding_dim, self.gcn_out_dims[0])

        # init graph convolutions
        self.gcns = []
        self.bns = []
        in_dim = self.hid_dim
        for dim in self.gcn_out_dims:
            self.gcns.append(
                DenseGraphFlow(
                    in_dim,
                    dim,
                    self.op_embedding_dim + self.concat_param_zs_as_opemb_dimension \
                    if symmetry_breaking_method == "param_zs" else\
                    (self.op_embedding_dim if not self.share_op_attention else dim),
                    has_attention=not self.share_op_attention,
                    **(gcn_kwargs or {})
                )
            )
            in_dim = dim
            if self.use_bn:
                self.bns.append(nn.BatchNorm1d(self.vertices))
        self.gcns = nn.ModuleList(self.gcns)
        if self.use_bn:
            self.bns = nn.ModuleList(self.bns)
        self.num_gcn_layers = len(self.gcns)
        self.out_dim = in_dim

        # init backward graph convolutions
        self.b_gcns = []
        self.b_bns = []
        if self.concat_arch_zs_as_l_dimension is not None \
           and self.concat_l_layer == len(self.fb_conversion_dims) - 1:
            in_dim = self.fb_conversion_dims[-1] + self.concat_arch_zs_as_l_dimension
        else:
            in_dim = self.fb_conversion_dims[-1]
        for dim in self.backward_gcn_out_dims:
            self.b_gcns.append(
                DenseGraphFlow(
                    in_dim,
                    dim,
                    self.op_embedding_dim + self.concat_param_zs_as_opemb_dimension \
                    if symmetry_breaking_method == "param_zs" else self.op_embedding_dim,
                    has_attention = not self.share_op_attention,
                    reverse = True,
                    **(gcn_kwargs or {})
                )
            )
            in_dim = dim
            if self.b_use_bn:
                self.b_bns.append(nn.BatchNorm1d(self.vertices))
        self.b_gcns = nn.ModuleList(self.b_gcns)
        if self.b_use_bn:
            self.b_bns = nn.ModuleList(self.b_bns)
        self.num_b_gcn_layers = len(self.b_gcns)

        # init the network to convert forward output info into backward input info
        if self.num_time_steps > 1:
            self.fb_conversion_list = []
            dim = self.gcn_out_dims[-1]
            num_fb_layers = len(fb_conversion_dims)
            self._num_before_concat_l = None
            for i_dim, fb_conversion_dim in enumerate(fb_conversion_dims):
                self.fb_conversion_list.append(nn.Linear(dim, fb_conversion_dim))
                if i_dim < num_fb_layers - 1:
                    self.fb_conversion_list.append(nn.ReLU(inplace=False))
                if self.concat_arch_zs_as_l_dimension is not None and \
                   self.concat_l_layer == i_dim:
                    dim = fb_conversion_dim + self.concat_arch_zs_as_l_dimension
                    self._num_before_concat_l = len(self.fb_conversion_list)
                else:
                    dim = fb_conversion_dim
            self.fb_conversion = nn.Sequential(*self.fb_conversion_list)

            # init the network to get delta op_emb
            if self.updateopemb_method == "concat_ofb":
                in_dim = self.gcn_out_dims[-1] + self.backward_gcn_out_dims[-1] \
                         + self.op_embedding_dim
            elif self.updateopemb_method == "concat_fb":
                in_dim = self.gcn_out_dims[-1] + self.backward_gcn_out_dims[-1]
            elif self.updateopemb_method == "concat_b":
                in_dim = self.backward_gcn_out_dims[-1]
            else:
                raise NotImplementedError()

            self.updateop_embedder = []
            for embedder_dim in self.updateopemb_dims:
                self.updateop_embedder.append(nn.Linear(in_dim, embedder_dim))
                self.updateop_embedder.append(nn.ReLU(inplace = False))
                in_dim = embedder_dim
            self.updateop_embedder.append(nn.Linear(in_dim, self.op_embedding_dim))
            self.updateop_embedder = nn.Sequential(*self.updateop_embedder)

    def embed_and_transform_arch(self, archs):
        # import pdb; pdb.set_trace()
        adjs = self.input_op_emb.new([arch.T for arch in archs[0]])
        op_inds = self.input_op_emb.new([arch for arch in archs[1]]).long()
        if self.use_global_node:
            tmp_ones = torch.ones((adjs.shape[0], 1, 1), device = adjs.device)
            tmp_cat = torch.cat(
                (
                    tmp_ones,
                    (op_inds != self.none_op_ind).unsqueeze(1).to(torch.float32),
                    tmp_ones,
                ),
                dim = 2
            )
            adjs = torch.cat(
                (
                    torch.cat((adjs, tmp_cat), dim = 1),
                    torch.zeros(
                        (adjs.shape[0], self.vertices, 1), device = adjs.device),
                ),
                dim = 2
            )
        # (batch_size, vertices - 2, op_emb_dim)
        op_embs = self.op_emb(op_inds)
        b_size = op_embs.shape[0]
        # the input one should not be relevant
        if self.use_global_node:
            op_embs = torch.cat(
                (
                    self.input_op_emb.unsqueeze(0).repeat([b_size, 1, 1]),
                    op_embs,
                    self.output_op_emb.weight.unsqueeze(
                        0).repeat([b_size, 1, 1]),
                    self.global_op_emb.unsqueeze(0).repeat([b_size, 1, 1]),
                ),
                dim = 1
            )
        else:
            op_embs = torch.cat(
                (
                    self.input_op_emb.unsqueeze(0).repeat([b_size, 1, 1]),
                    op_embs,
                    self.output_op_emb.weight.unsqueeze(
                        0).repeat([b_size, 1, 1]),
                ),
                dim = 1
            )
        node_embs = torch.cat(
            (
                self.input_node_emb.weight.unsqueeze(0).repeat([b_size, 1, 1]),
                self.other_node_emb.unsqueeze(0).repeat(
                    [b_size, self.vertices - 1, 1]),
            ),
            dim = 1
        )
        x = self.x_hidden(node_embs)
        # x: (batch_size, vertices, hid_dim)
        return adjs, x, op_embs, op_inds

    def _forward_pass(self, x, adjs, auged_op_emb) -> Tensor:
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

    def _backward_pass(self, y, adjs, zs_as_l, auged_op_emb) -> Tensor:
        # --- backward pass ---
        # get the information of the output node
        # b_info = torch.cat(
        #     (
        #         torch.zeros([y.shape[0], self.vertices - 1, y.shape[-1]], device=y.device),
        #         y[:, -1:, :]
        #     ),
        #     dim=1
        # )
        b_info = y[:, -1:, :]
        if self.detach_vinfo:
            b_info = b_info.detach()
        if self.concat_arch_zs_as_l_dimension:
            # process before concat l
            b_info = self.fb_conversion[:self._num_before_concat_l](b_info)
            # concat l
            b_info = torch.cat((b_info, zs_as_l.unsqueeze(-2)), dim = -1)
            if not self.concat_l_layer == len(self.fb_conversion_list) - 1:
                # process after concat l
                b_info = self.fb_conversion[self._num_before_concat_l:](b_info)
        else:
            b_info = self.fb_conversion(b_info)
        b_info = torch.cat(
            (
                torch.zeros([y.shape[0], self.vertices - 1, b_info.shape[-1]], device = y.device),
                b_info
            ),
            dim = 1
        )

        # start backward flow
        b_adjs = adjs.transpose(1, 2)
        b_y = b_info
        for i_layer, gcn in enumerate(self.b_gcns):
            b_y = gcn(b_y, b_adjs, auged_op_emb)
            if self.b_use_bn:
                shape_y = b_y.shape
                b_y = self.b_bns[i_layer](b_y.reshape(shape_y[0], -1, shape_y[-1]))\
                            .reshape(shape_y)
            if i_layer != self.num_b_gcn_layers - 1:
                b_y = F.relu(b_y)
                b_y = F.dropout(b_y, self.dropout, training = self.training)

        return b_y

    def _update_op_emb(self, y, b_y, op_emb, concat_op_emb_mask) -> Tensor:
        # --- UpdateOpEmb ---
        if self.updateopemb_method == "concat_ofb":
            in_embedding = torch.cat(
                (
                    op_emb.detach() if self.updateopemb_detach_opemb else op_emb,
                    y.detach() if self.updateopemb_detach_finfo else y,
                    b_y
                ),
                dim = -1)
        elif self.updateopemb_method == "concat_fb":
            in_embedding = torch.cat(
                (
                    y.detach() if self.updateopemb_detach_finfo else y,
                    b_y
                ), dim = -1)
        elif self.updateopemb_method == "concat_b":
            in_embedding = b_y
        update = self.updateop_embedder(in_embedding)

        if self.mask_nonparametrized_ops:
            update = update * concat_op_emb_mask

        op_emb = op_emb + self.updateopemb_scale * update
        return op_emb

    def _final_process(self, y: Tensor, op_inds) -> Tensor:
        ## --- output ---
        # y: (batch_size, vertices, gcn_out_dims[-1])
        if self.use_final_only:
            # only use the output node's info embedding as the embedding
            y = y[:, -1, :]
        else:
            y = y[:, 1:, :]  # do not keep the inputs node embedding

            # throw away padded info here
            if self.use_global_node:
                y = torch.cat(
                    (
                        y[:, :-2, :]
                        * (op_inds !=
                           self.none_op_ind)[:, :, None].to(torch.float32),
                        y[:, -2:, :],
                    ),
                    dim = 1
                )
            else:
                y = torch.cat(
                    (
                        y[:, :-1, :]
                        * (op_inds !=
                           self.none_op_ind)[:, :, None].to(torch.float32),
                        y[:, -1:, :],
                    ),
                    dim = 1
                )

            y = torch.mean(y, dim = 1)  # average across nodes (bs, god)

        return y

    def forward(self, x_ops, x_adj, zs_as_p, zcp_):
        # adjs: (batch_size, vertices, vertices)
        # x: (batch_size, vertices, hid_dim)
        # op_emb: (batch_size, vertices, emb_dim)
        # if isinstance(archs, tuple):
        #     if len(archs) == 2:
        #         archs, zs_as_p = archs
        #         zs_as_l = None
        #     elif len(archs) == 3:
        #         archs, zs_as_l, zs_as_p = archs
        #     else:
        #         raise Exception()
        # else:
        #     zs_as_l = zs_as_p = None
        # zip(x_adj, x_ops)
        # import pdb; pdb.set_trace()
        # archs = list(zip([np.asarray(x.cpu()) for x in x_adj], [np.asarray(x.cpu()) for x in x_ops]))
        archs = [[np.asarray(x.cpu()) for x in x_adj], [np.asarray(x.cpu()) for x in x_ops]]
        zs_as_p = zs_as_p.cpu()
        zs_as_l = None
        adjs, x, op_emb, op_inds = self.embed_and_transform_arch(archs)
        if self.concat_arch_zs_as_l_dimension is not None:
            zs_as_l = self.op_emb.weight.new(np.array(zs_as_l))
            assert zs_as_l.shape[-1] == self.concat_arch_zs_as_l_dimension
        # import pdb; pdb.set_trace()
        concat_op_emb_mask = ((op_inds == 0) | (op_inds == 1))
        concat_op_emb_mask = F.pad(concat_op_emb_mask, (1, 1), mode = "constant")
        concat_op_emb_mask = concat_op_emb_mask.unsqueeze(-1).to(torch.float32)

        # symmetry breaking
        if self.symmetry_breaking_method == "random":
            # random, dimension not changed: op_emb + random noise
            noise = torch.zeros_like(op_emb).normal_() * 0.1
            op_emb = op_emb + noise
        elif self.symmetry_breaking_method == "param_zs":
            zs_as_p = self.input_op_emb.new(np.array(zs_as_p))
            assert zs_as_p.shape[-1] == self.concat_param_zs_as_opemb_dimension
        elif self.symmetry_breaking_method == "param_zs_add":
            # param-level zeroshot: op_emb | zeroshot
            # import pdb; pdb.set_trace()
            zs_as_p = self.input_op_emb.new(np.array(zs_as_p))
            zs_as_p = self.param_zs_embedder(zs_as_p)
            # import pdb; pdb.set_trace()
            op_emb = op_emb + zs_as_p * self.concat_param_zs_as_opemb_scale

        if self.share_op_attention:
            op_emb = self.op_attention(op_emb)

        for t in range(self.num_time_steps):
            if self.symmetry_breaking_method == "param_zs":
                # param-level zeroshot: op_emb | zeroshot
                auged_op_emb = torch.cat((op_emb, zs_as_p), dim = -1)
            else:
                auged_op_emb = op_emb

            y = self._forward_pass(x, adjs, auged_op_emb)

            if t == self.num_time_steps - 1:
                break

            b_y = self._backward_pass(y, adjs, zs_as_l, auged_op_emb)
            op_emb = self._update_op_emb(y, b_y, op_emb, concat_op_emb_mask)

        ## --- output ---
        # y: (batch_size, vertices, gcn_out_dims[-1])
        return self.mlp(self._final_process(y, op_inds))
