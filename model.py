# -*- coding:utf-8 -*-
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from einops import *

from torchdiffeq import odeint_adjoint as odeint


class Value_Encoder(nn.Module):
    def __init__(self, output_dim):
        self.output_dim = output_dim
        super(Value_Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(1, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x = rearrange(x, 'b l k -> b l k 1')
        x = self.encoder(x)
        return x


class Time_Encoder(nn.Module):
    def __init__(self, embed_time, var_num):
        super(Time_Encoder, self).__init__()
        self.periodic = nn.Linear(1, embed_time - 1)
        self.var_num = var_num
        self.linear = nn.Linear(1, 1)

    def forward(self, tt):
        if tt.dim() == 3:  # [B,L,K]
            tt = rearrange(tt, 'b l k -> b l k 1')
        else:  # [B,L]
            tt = rearrange(tt, 'b l -> b l 1 1')

        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        out = torch.cat([out1, out2], -1)  # [B,L,1,D]
        # out = repeat(out, 'b l 1 d -> b l v d', v=self.var_num)
        return out


class FFNN(nn.Module):
    def __init__(self, input_dim, hid_units, output_dim):
        self.hid_units = hid_units
        self.output_dim = output_dim
        super(FFNN, self).__init__()

        self.linear = nn.Linear(input_dim, hid_units)
        self.W = nn.Linear(hid_units, output_dim, bias=False)

    def forward(self, x):
        x = self.linear(x)
        x = self.W(torch.tanh(x))
        return x


class MLP_Tau_Encoder(nn.Module):
    def __init__(self, embed_time, hid_dim=16):
        super(MLP_Tau_Encoder, self).__init__()
        self.encoder = FFNN(1, hid_dim, embed_time)

    def forward(self, tt):
        if tt.dim() == 3:  # [B,L,K]
            tt = rearrange(tt, 'b l k -> b l k 1')
        else:  # [B,L]
            tt = rearrange(tt, 'b l -> b l 1 1')
        # out1 = F.gelu(self.linear1(tt))
        tt = self.encoder(tt)
        return tt  # [B,L,K,D]


class MLP_Weight_Pool(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, nodes_num):
        super(MLP_Weight_Pool, self).__init__()
        self.linear_layer = nn.ModuleList([nn.Linear(input_size, output_size) for i in range(nodes_num)])
        self.linear_layer_weight = \
            nn.Parameter(torch.stack([self.linear_layer[i].weight.T for i in range(nodes_num)]))
        self.linear_layer_bias = \
            nn.Parameter(torch.stack([self.linear_layer[i].bias.T for i in range(nodes_num)]))
        self.relu = nn.ReLU()

    def forward(self, x, node_ind):
        x = torch.einsum("ni,nio->no", x, self.linear_layer_weight[node_ind[1]]) \
            + self.linear_layer_bias[node_ind[1]]
        #        x = self.relu(x)
        return x


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x, node_ind=None):
        return self.layers(x)


class MLP_linear(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP_linear, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x, node_ind=None):
        return self.layers(x)
        
class MLP_Param(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, node_nums):
        super(MLP_Param, self).__init__()
        self.W_1 = nn.Parameter(torch.FloatTensor(node_nums, input_size, output_size))
        self.b_1 = nn.Parameter(torch.FloatTensor(node_nums, output_size))
        #        self.W_2 = nn.Parameter(torch.FloatTensor(node_nums, hidden_size, output_size))
        #        self.b_2 = nn.Parameter(torch.FloatTensor(node_nums, output_size))

        nn.init.xavier_uniform_(self.W_1)
        nn.init.xavier_uniform_(self.b_1)

    #        nn.init.xavier_uniform_(self.W_2)
    #        nn.init.xavier_uniform_(self.b_2)

    def forward(self, x, var_vector, node_ind=None):
        W_1 = torch.einsum("nd, dio->nio", var_vector, self.W_1)
        b_1 = torch.einsum("nd, do->no", var_vector, self.b_1)
        x = torch.squeeze(torch.bmm(x.unsqueeze(1), W_1)) + b_1
        return x
        #        x = torch.relu(x)
        #        W_2 = torch.einsum("nd, dio->nio", var_vector, self.W_2)
        #        b_2 = torch.einsum("nd, do->no", var_vector, self.b_2)
        return torch.squeeze(torch.bmm(x.unsqueeze(1), W_2)) + b_2


class Weight_Param(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, node_nums):
        super(Weight_Param, self).__init__()
        self.W_1 = nn.Parameter(torch.FloatTensor(input_size, output_size))
        self.b_1 = nn.Parameter(torch.FloatTensor(output_size))
        nn.init.xavier_uniform_(self.W_1)
        nn.init.uniform(self.b_1)

    #        nn.init.xavier_uniform_(self.W_2)
    #        nn.init.xavier_uniform_(self.b_2)

    def forward(self, x, var_vector, node_ind=None):
        x = torch.matmul(x, self.W_1) + self.b_1
        return x
        #        x = torch.relu(x)
        #        W_2 = torch.einsum("nd, dio->nio", var_vector, self.W_2)
        #        b_2 = torch.einsum("nd, do->no", var_vector, self.b_2)
        return torch.squeeze(torch.bmm(x.unsqueeze(1), W_2)) + b_2


class MLP_Param_Var(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, node_nums):
        super(MLP_Param_Var, self).__init__()
        self.W_1 = nn.Parameter(torch.FloatTensor(node_nums, input_size, output_size))
        self.b_1 = nn.Parameter(torch.FloatTensor(node_nums, output_size))
        nn.init.xavier_uniform_(self.W_1)
        nn.init.xavier_uniform_(self.b_1)

    #        nn.init.xavier_uniform_(self.W_2)
    #        nn.init.xavier_uniform_(self.b_2)

    def forward(self, x, var_vector, node_ind=None):
        # W_1 = torch.einsum("nd, dio->nio", var_vector, self.W_1)
        # b_1 = torch.einsum("nd, do->no", var_vector, self.b_1)
        x = torch.squeeze(torch.bmm(x.unsqueeze(1), self.W_1[node_ind[1]])) + self.b_1[node_ind[1]]
        return x
        #        x = torch.relu(x)
        #        W_2 = torch.einsum("nd, dio->nio", var_vector, self.W_2)
        #        b_2 = torch.einsum("nd, do->no", var_vector, self.b_2)
        return torch.squeeze(torch.bmm(x.unsqueeze(1), W_2)) + b_2
# class AGCRNCellWithMLP(nn.Module):
#    def __init__(self, input_size, mlp_hidden_size, nodes_num):
#        super(AGCRNCellWithMLP, self).__init__()
#        self.update_gate = MLP_Param(2 * input_size + 1, mlp_hidden_size, input_size, nodes_num)
#        self.reset_gate = MLP_Param(2 * input_size + 1, mlp_hidden_size, input_size, nodes_num)
#        self.candidate_gate = MLP_Param(2 * input_size + 1, mlp_hidden_size, input_size, nodes_num)
#
#    def forward(self, x, h, var_vector, adj, nodes_ind):
#        combined = torch.cat([x, h], dim=-1)
#        combined = torch.matmul(adj, combined)
#        r = torch.sigmoid(self.reset_gate(combined[nodes_ind], var_vector, nodes_ind))
#        u = torch.sigmoid(self.update_gate(combined[nodes_ind], var_vector, nodes_ind))
#        combined_new = torch.cat([x[nodes_ind], r * h[nodes_ind]], dim=-1)
#        candidate_h = torch.tanh(self.candidate_gate(combined_new, var_vector, nodes_ind))
#        return (1 - u) * h[nodes_ind] + u * candidate_h


class AGCRNCellWithMLP(nn.Module):
    def __init__(self, input_size, mlp_hidden_size, nodes_num):
        super(AGCRNCellWithMLP, self).__init__()
        self.update_gate = MLP_Param(2 * input_size + 1, mlp_hidden_size, input_size, nodes_num)
        self.reset_gate = MLP_Param(2 * input_size + 1, mlp_hidden_size, input_size, nodes_num)
        self.candidate_gate = MLP_Param(2 * input_size + 1, mlp_hidden_size, input_size, nodes_num)

    def forward(self, x, h, var_vector, adj, nodes_ind):
        combined = torch.cat([x, h], dim=-1)
        combined = torch.matmul(adj, combined)
        r = torch.sigmoid(self.reset_gate(combined[nodes_ind], var_vector, nodes_ind))
        u = torch.sigmoid(self.update_gate(combined[nodes_ind], var_vector, nodes_ind))
        h[nodes_ind] = r * h[nodes_ind]
        combined_new = torch.cat([x, h], dim=-1)
        #        combined_new = torch.matmul(adj, combined_new)
        candidate_h = torch.tanh(self.candidate_gate(combined_new[nodes_ind], var_vector, nodes_ind))
        return (1 - u) * h[nodes_ind] + u * candidate_h

class AGCRNCellWithMLP_Var(nn.Module):
    def __init__(self, input_size, mlp_hidden_size, nodes_num):
        super(AGCRNCellWithMLP_Var, self).__init__()
        self.update_gate = MLP_Param_Var(2 * input_size + 1, mlp_hidden_size, input_size, nodes_num)
        self.reset_gate = MLP_Param_Var(2 * input_size + 1, mlp_hidden_size, input_size, nodes_num)
        self.candidate_gate = MLP_Param_Var(2 * input_size + 1, mlp_hidden_size, input_size, nodes_num)

    def forward(self, x, h, var_vector, adj, nodes_ind):
        combined = torch.cat([x, h], dim=-1)
        combined = torch.matmul(adj, combined)
        r = torch.sigmoid(self.reset_gate(combined[nodes_ind], var_vector, nodes_ind))
        u = torch.sigmoid(self.update_gate(combined[nodes_ind], var_vector, nodes_ind))
        h[nodes_ind] = r * h[nodes_ind]
        combined_new = torch.cat([x, h], dim=-1)
        #        combined_new = torch.matmul(adj, combined_new)
        candidate_h = torch.tanh(self.candidate_gate(combined_new[nodes_ind], var_vector, nodes_ind))
        return (1 - u) * h[nodes_ind] + u * candidate_h

class AGCRNCellWithGenearalMLP(nn.Module):
    def __init__(self, input_size, mlp_hidden_size, nodes_num):
        super(AGCRNCellWithGenearalMLP, self).__init__()
        self.update_gate = Weight_Param(2 * input_size + 1, mlp_hidden_size, input_size, nodes_num)
        self.reset_gate = Weight_Param(2 * input_size + 1, mlp_hidden_size, input_size, nodes_num)
        self.candidate_gate = Weight_Param(2 * input_size + 1, mlp_hidden_size, input_size, nodes_num)

    def forward(self, x, h, var_vector, adj, nodes_ind):
        combined = torch.cat([x, h], dim=-1)
        combined = torch.matmul(adj, combined)
        r = torch.sigmoid(self.reset_gate(combined[nodes_ind], var_vector, nodes_ind))
        u = torch.sigmoid(self.update_gate(combined[nodes_ind], var_vector, nodes_ind))
        h[nodes_ind] = r * h[nodes_ind]
        combined_new = torch.cat([x, h], dim=-1)
        #        combined_new = torch.matmul(adj, combined_new)
        candidate_h = torch.tanh(self.candidate_gate(combined_new[nodes_ind], var_vector, nodes_ind))
        return (1 - u) * h[nodes_ind] + u * candidate_h


class GRUCellWithLinear(nn.Module):
    def __init__(self, input_size, mlp_hidden_size, nodes_num):
        super(GRUCellWithLinear, self).__init__()
        self.update_gate = nn.Linear(2 * input_size + 1, input_size)
        self.reset_gate = nn.Linear(2 * input_size + 1, input_size)
        self.candidate_gate = nn.Linear(2 * input_size + 1, input_size)

    def forward(self, x, h, var_vector, nodes_ind):
        combined = torch.cat([x, h], dim=-1)
        r = torch.sigmoid(self.reset_gate(combined))
        u = torch.sigmoid(self.update_gate(combined))
        combined_new = torch.cat([x, r * h], dim=-1)
        candidate_h = torch.tanh(self.candidate_gate(combined_new))

        return (1 - u) * h + u * candidate_h


class CGRNN_batch_adjIGraph(nn.Module):
    def __init__(self, d_in, d_model, num_nodes, rarity_alpha=0.5, var_emb_dim=5, var_graph_emb_dim=8):
        super(CGRNN_batch_adjIGraph, self).__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.num_nodes = num_nodes
        self.gated_update = AGCRNCellWithMLP(d_model, 2 * d_model, var_emb_dim)
        self.rarity_alpha = rarity_alpha
        self.rarity_W = nn.Parameter(torch.randn(num_nodes, num_nodes))
        self.relu = nn.ReLU()
        self.prior_to_state = MLP(768, 2 * d_model, var_emb_dim)
        self.adjI = nn.Parameter(torch.eye(num_nodes))

    def init_hidden_states(self, x):
        return torch.zeros(size=(x.shape[0], x.shape[2], self.d_model)).to(x.device)

    def forward(self, obs_emb, adj, observed_mask, observed_tp, tp_emb_tensor, lengths, avg_interval,
                var_prior_emb_tensor):
        batch, steps, nodes, features = obs_emb.size()

        device = obs_emb.device

        h = self.init_hidden_states(obs_emb)

        I = repeat(torch.eye(nodes).to(device), 'v x -> b v x', b=batch)

        output = torch.zeros_like(h)

        nodes_initial_mask = torch.zeros(batch, nodes).to(device)

        var_total_obs = torch.sum(observed_mask, dim=1)

        var_prior_emb_tensor = repeat(var_prior_emb_tensor, "n d -> b n d", b=batch)

        var_vector = self.prior_to_state(var_prior_emb_tensor)

        for step in range(int(torch.max(lengths).item())):

            adj_mask = torch.zeros(size=[batch, nodes, nodes]).to(device)

            cur_obs = obs_emb[:, step]

            cur_mask = observed_mask[:, step]

            cur_obs_var = torch.where(cur_mask)

            nodes_initial_mask[cur_obs_var] = 1

            nodes_need_update = cur_obs_var

            cur_avg_interval = avg_interval[:, step]

            rarity_score = self.rarity_alpha * torch.tanh(cur_avg_interval / (var_total_obs + 1))

            rarity_score_matrix_row = repeat(rarity_score, 'b v -> b v x', x=nodes)
            rarity_score_matrix_col = repeat(rarity_score, 'b v -> b x v', x=nodes)
            rarity_score_matrix = -1 * self.rarity_W * (torch.abs(rarity_score_matrix_row - rarity_score_matrix_col))

            if nodes_need_update[0].shape[0] > 0:
                adj_mask[cur_obs_var[0], :, cur_obs_var[1]] = torch.ones(len(cur_obs_var[0]), nodes).to(device)
                wo_observed_nodes = torch.where(cur_mask == 0)
                wo_init_nodes = torch.where(nodes_initial_mask == 0)
                adj_mask[wo_observed_nodes] = torch.zeros(len(wo_observed_nodes[0]), nodes).to(device)

                cur_adj = self.adjI * (1 + rarity_score_matrix) * adj_mask * (1 - I) + I

                h[nodes_need_update] = self.gated_update(
                    torch.cat([cur_obs, rarity_score.unsqueeze(-1)], dim=-1),
                    h, var_vector[nodes_need_update], cur_adj, nodes_need_update)

            end_sample_ind = torch.where(step == (lengths.squeeze(1) - 1))
            output[end_sample_ind[0]] = h[end_sample_ind[0]]
            if step == int(torch.max(lengths).item()) - 1:
                return output

        return output


class CGRNN_batch_IGraph(nn.Module):
    def __init__(self, d_in, d_model, num_nodes, rarity_alpha=0.5, var_emb_dim=5, var_graph_emb_dim=8):
        super(CGRNN_batch_IGraph, self).__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.num_nodes = num_nodes
        self.gated_update = AGCRNCellWithMLP(d_model, 2 * d_model, var_emb_dim)
        self.rarity_alpha = rarity_alpha
        self.relu = nn.ReLU()
        self.prior_to_state = MLP(768, 2 * d_model, var_emb_dim)

    def init_hidden_states(self, x):
        return torch.zeros(size=(x.shape[0], x.shape[2], self.d_model)).to(x.device)

    def forward(self, obs_emb, adj, observed_mask, observed_tp, tp_emb_tensor, lengths, avg_interval,
                var_prior_emb_tensor):
        batch, steps, nodes, features = obs_emb.size()

        device = obs_emb.device

        h = self.init_hidden_states(obs_emb)

        I = repeat(torch.eye(nodes).to(device), 'v x -> b v x', b=batch)

        output = torch.zeros_like(h)

        nodes_initial_mask = torch.zeros(batch, nodes).to(device)

        var_total_obs = torch.sum(observed_mask, dim=1)

        var_prior_emb_tensor = repeat(var_prior_emb_tensor, "n d -> b n d", b=batch)

        var_vector = self.prior_to_state(var_prior_emb_tensor)

        for step in range(int(torch.max(lengths).item())):

            adj_mask = torch.zeros(size=[batch, nodes, nodes]).to(device)

            cur_obs = obs_emb[:, step]

            cur_mask = observed_mask[:, step]

            cur_obs_var = torch.where(cur_mask)

            nodes_initial_mask[cur_obs_var] = 1

            nodes_need_update = cur_obs_var

            cur_avg_interval = avg_interval[:, step]

            rarity_score = self.rarity_alpha * torch.tanh(cur_avg_interval / (var_total_obs + 1))
            if nodes_need_update[0].shape[0] > 0:
                adj_mask[cur_obs_var[0], :, cur_obs_var[1]] = torch.ones(len(cur_obs_var[0]), nodes).to(device)
                wo_observed_nodes = torch.where(cur_mask == 0)
                wo_init_nodes = torch.where(nodes_initial_mask == 0)
                adj_mask[wo_observed_nodes] = torch.zeros(len(wo_observed_nodes[0]), nodes).to(device)
                cur_adj = I
                h[nodes_need_update] = self.gated_update(
                    torch.cat([cur_obs, rarity_score.unsqueeze(-1)], dim=-1),
                    h, var_vector[nodes_need_update], cur_adj, nodes_need_update)

            end_sample_ind = torch.where(step == (lengths.squeeze(1) - 1))
            output[end_sample_ind[0]] = h[end_sample_ind[0]]
            if step == int(torch.max(lengths).item()) - 1:
                return output

        return output


class CGRNN_batch_woSpec(nn.Module):
    def __init__(self, d_in, d_model, num_nodes, rarity_alpha=0.5, var_emb_dim=5, var_graph_emb_dim=8):
        super(CGRNN_batch_woSpec, self).__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.num_nodes = num_nodes
        self.gated_update = AGCRNCellWithGenearalMLP(d_model, 2 * d_model, var_emb_dim)
        self.rarity_alpha = rarity_alpha
        self.rarity_W = nn.Parameter(torch.randn(num_nodes, num_nodes))
        self.relu = nn.ReLU()
        self.prior_to_state = MLP(768, 2 * d_model, var_emb_dim)
        self.prior_to_graph = MLP(768, 2 * d_model, var_graph_emb_dim)

    def init_hidden_states(self, x):
        return torch.zeros(size=(x.shape[0], x.shape[2], self.d_model)).to(x.device)

    def forward(self, obs_emb, adj, observed_mask, observed_tp, tp_emb_tensor, lengths, avg_interval,
                var_prior_emb_tensor):
        batch, steps, nodes, features = obs_emb.size()

        device = obs_emb.device

        h = self.init_hidden_states(obs_emb)

        I = repeat(torch.eye(nodes).to(device), 'v x -> b v x', b=batch)

        output = torch.zeros_like(h)

        nodes_initial_mask = torch.zeros(batch, nodes).to(device)

        var_total_obs = torch.sum(observed_mask, dim=1)

        var_prior_emb_tensor = repeat(var_prior_emb_tensor, "n d -> b n d", b=batch)

        var_vector = self.prior_to_state(var_prior_emb_tensor)

        prior_graph_vector = self.prior_to_graph(var_prior_emb_tensor)

        var_vector_nor = F.normalize(prior_graph_vector, p=2, dim=2)

        adj = torch.softmax(torch.bmm(var_vector_nor, var_vector_nor.permute(0, 2, 1)), dim=-1)

        for step in range(int(torch.max(lengths).item())):

            adj_mask = torch.zeros(size=[batch, nodes, nodes]).to(device)

            cur_obs = obs_emb[:, step]

            cur_mask = observed_mask[:, step]

            cur_obs_var = torch.where(cur_mask)

            nodes_initial_mask[cur_obs_var] = 1

            nodes_need_update = cur_obs_var

            cur_avg_interval = avg_interval[:, step]

            rarity_score = self.rarity_alpha * torch.tanh(cur_avg_interval / (var_total_obs + 1))
            #            rarity_score = self.rarity_alpha * self.rarity_W * torch.tanh(cur_avg_interval / (var_total_obs + 1))
            #            rarity_score_matrix = repeat(rarity_score, 'b v -> b v x', x=nodes)

            #            rarity_score = self.rarity_alpha * torch.tanh(cur_avg_interval / (var_total_obs + 1))
            rarity_score_matrix_row = repeat(rarity_score, 'b v -> b v x', x=nodes)
            rarity_score_matrix_col = repeat(rarity_score, 'b v -> b x v', x=nodes)
            #            rarity_score_matrix = -1* self.rarity_W *(rarity_score_matrix_row + rarity_score_matrix_col)
            #            rarity_score_matrix = rarity_score_matrix_row
            rarity_score_matrix = -1 * self.rarity_W * (torch.abs(rarity_score_matrix_row - rarity_score_matrix_col))
            # rarity_score_matrix = self.rarity_W * (rarity_score_matrix_row + rarity_score_matrix_col)
            if nodes_need_update[0].shape[0] > 0:
                adj_mask[cur_obs_var[0], :, cur_obs_var[1]] = torch.ones(len(cur_obs_var[0]), nodes).to(device)
                wo_observed_nodes = torch.where(cur_mask == 0)
                wo_init_nodes = torch.where(nodes_initial_mask == 0)
                adj_mask[wo_observed_nodes] = torch.zeros(len(wo_observed_nodes[0]), nodes).to(device)
                #                adj_mask[wo_observed_nodes] = torch.zeros(len(wo_observed_nodes[0]), nodes).to(device)

                #                cur_adj = adj * adj_mask * (1 - I) + I
                #                cur_adj = (adj * (1 + rarity_score_matrix) * adj_mask + I) / 2
                #                cur_adj = I
                #                cur_adj = self.adjI * adj_mask * (1 - I) + I
                #                cur_adj = I * adj_mask * (1 + rarity_score_matrix) * (1 - I) + I
                #                cur_adj = adj * (1 + rarity_score_matrix) * adj_mask
                #                cur_adj = adj * adj_mask * (1 - I) + I
                #                cur_adj = adj * adj_mask * (1 - I) + I
                cur_adj = adj * (1 + rarity_score_matrix) * adj_mask * (1 - I) + I
                #                cur_adj = self.adjI * (1 + rarity_score_matrix) * adj_mask * (1 - I) + I
                h[nodes_need_update] = self.gated_update(
                    torch.cat([cur_obs, rarity_score.unsqueeze(-1)], dim=-1),
                    h, var_vector[nodes_need_update], cur_adj, nodes_need_update)

            end_sample_ind = torch.where(step == (lengths.squeeze(1) - 1))
            output[end_sample_ind[0]] = h[end_sample_ind[0]]
            if step == int(torch.max(lengths).item()) - 1:
                return output

        return output


class CGRNN_batch_fixedGraph(nn.Module):
    def __init__(self, d_in, d_model, num_nodes, rarity_alpha=0.5, var_emb_dim=5, var_graph_emb_dim=8):
        super(CGRNN_batch_fixedGraph, self).__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.num_nodes = num_nodes
        self.gated_update = AGCRNCellWithMLP(d_model, 2 * d_model, var_emb_dim)
        self.rarity_alpha = rarity_alpha
        self.rarity_W = nn.Parameter(torch.randn(num_nodes, num_nodes))
        self.relu = nn.ReLU()
        self.prior_to_state = MLP(768, 2 * d_model, var_emb_dim)

    def init_hidden_states(self, x):
        return torch.zeros(size=(x.shape[0], x.shape[2], self.d_model)).to(x.device)

    def forward(self, obs_emb, adj, observed_mask, observed_tp, tp_emb_tensor, lengths, avg_interval,
                var_prior_emb_tensor):
        batch, steps, nodes, features = obs_emb.size()

        device = obs_emb.device

        h = self.init_hidden_states(obs_emb)

        I = repeat(torch.eye(nodes).to(device), 'v x -> b v x', b=batch)

        output = torch.zeros_like(h)

        nodes_initial_mask = torch.zeros(batch, nodes).to(device)

        var_total_obs = torch.sum(observed_mask, dim=1)

        var_prior_emb_tensor = repeat(var_prior_emb_tensor, "n d -> b n d", b=batch)

        var_vector = self.prior_to_state(var_prior_emb_tensor)

        for step in range(int(torch.max(lengths).item())):

            adj_mask = torch.zeros(size=[batch, nodes, nodes]).to(device)

            cur_obs = obs_emb[:, step]

            cur_mask = observed_mask[:, step]

            cur_obs_var = torch.where(cur_mask)

            nodes_initial_mask[cur_obs_var] = 1

            nodes_need_update = cur_obs_var

            cur_avg_interval = avg_interval[:, step]

            rarity_score = self.rarity_alpha * torch.tanh(cur_avg_interval / (var_total_obs + 1))
            #            rarity_score = self.rarity_alpha * self.rarity_W * torch.tanh(cur_avg_interval / (var_total_obs + 1))
            #            rarity_score_matrix = repeat(rarity_score, 'b v -> b v x', x=nodes)

            #            rarity_score = self.rarity_alpha * torch.tanh(cur_avg_interval / (var_total_obs + 1))
            rarity_score_matrix_row = repeat(rarity_score, 'b v -> b v x', x=nodes)
            rarity_score_matrix_col = repeat(rarity_score, 'b v -> b x v', x=nodes)
            #            rarity_score_matrix = -1* self.rarity_W *(rarity_score_matrix_row + rarity_score_matrix_col)
            #            rarity_score_matrix = rarity_score_matrix_row
            rarity_score_matrix = -1 * self.rarity_W * (torch.abs(rarity_score_matrix_row - rarity_score_matrix_col))
            # rarity_score_matrix = self.rarity_W * (rarity_score_matrix_row + rarity_score_matrix_col)
            if nodes_need_update[0].shape[0] > 0:
                adj_mask[cur_obs_var[0], :, cur_obs_var[1]] = torch.ones(len(cur_obs_var[0]), nodes).to(device)
                wo_observed_nodes = torch.where(cur_mask == 0)
                wo_init_nodes = torch.where(nodes_initial_mask == 0)
                adj_mask[wo_observed_nodes] = torch.zeros(len(wo_observed_nodes[0]), nodes).to(device)
                #                adj_mask[wo_observed_nodes] = torch.zeros(len(wo_observed_nodes[0]), nodes).to(device)

                #                cur_adj = adj * adj_mask * (1 - I) + I
                #                cur_adj = (adj * (1 + rarity_score_matrix) * adj_mask + I) / 2
                #                cur_adj = I
                #                cur_adj = self.adjI * adj_mask * (1 - I) + I
                #                cur_adj = I * adj_mask * (1 + rarity_score_matrix) * (1 - I) + I
                #                cur_adj = adj * (1 + rarity_score_matrix) * adj_mask
                #                cur_adj = adj * adj_mask * (1 - I) + I
                #                cur_adj = adj * adj_mask * (1 - I) + I
                cur_adj = adj * (1 + rarity_score_matrix) * adj_mask * (1 - I) + I
                #                cur_adj = self.adjI * (1 + rarity_score_matrix) * adj_mask * (1 - I) + I
                h[nodes_need_update] = self.gated_update(
                    torch.cat([cur_obs, rarity_score.unsqueeze(-1)], dim=-1),
                    h, var_vector[nodes_need_update], cur_adj, nodes_need_update)

            end_sample_ind = torch.where(step == (lengths.squeeze(1) - 1))
            output[end_sample_ind[0]] = h[end_sample_ind[0]]
            if step == int(torch.max(lengths).item()) - 1:
                return output

        return output


class CGRNN_batch_woRare(nn.Module):
    def __init__(self, d_in, d_model, num_nodes, rarity_alpha=0.5, var_emb_dim=5, var_graph_emb_dim=8):
        super(CGRNN_batch_woRare, self).__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.num_nodes = num_nodes
        self.gated_update = AGCRNCellWithMLP(d_model, 2 * d_model, var_emb_dim)
        self.rarity_alpha = rarity_alpha
        self.relu = nn.ReLU()
        self.prior_to_state = MLP(768, 2 * d_model, var_emb_dim)
        self.prior_to_graph = MLP(768, 2 * d_model, var_graph_emb_dim)

    def init_hidden_states(self, x):
        return torch.zeros(size=(x.shape[0], x.shape[2], self.d_model)).to(x.device)

    def forward(self, obs_emb, adj, observed_mask, observed_tp, tp_emb_tensor, lengths, avg_interval,
                var_prior_emb_tensor):
        batch, steps, nodes, features = obs_emb.size()

        device = obs_emb.device

        h = self.init_hidden_states(obs_emb)

        I = repeat(torch.eye(nodes).to(device), 'v x -> b v x', b=batch)

        output = torch.zeros_like(h)

        nodes_initial_mask = torch.zeros(batch, nodes).to(device)

        var_total_obs = torch.sum(observed_mask, dim=1)

        var_prior_emb_tensor = repeat(var_prior_emb_tensor, "n d -> b n d", b=batch)

        var_vector = self.prior_to_state(var_prior_emb_tensor)

        prior_graph_vector = self.prior_to_graph(var_prior_emb_tensor)

        var_vector_nor = F.normalize(prior_graph_vector, p=2, dim=2)

        adj = torch.softmax(torch.bmm(var_vector_nor, var_vector_nor.permute(0, 2, 1)), dim=-1)

        for step in range(int(torch.max(lengths).item())):

            adj_mask = torch.zeros(size=[batch, nodes, nodes]).to(device)

            cur_obs = obs_emb[:, step]

            cur_mask = observed_mask[:, step]

            cur_obs_var = torch.where(cur_mask)

            nodes_initial_mask[cur_obs_var] = 1

            nodes_need_update = cur_obs_var

            cur_avg_interval = avg_interval[:, step]

            rarity_score = self.rarity_alpha * torch.tanh(cur_avg_interval / (var_total_obs + 1))

            if nodes_need_update[0].shape[0] > 0:
                adj_mask[cur_obs_var[0], :, cur_obs_var[1]] = torch.ones(len(cur_obs_var[0]), nodes).to(device)
                wo_observed_nodes = torch.where(cur_mask == 0)
                wo_init_nodes = torch.where(nodes_initial_mask == 0)
                adj_mask[wo_observed_nodes] = torch.zeros(len(wo_observed_nodes[0]), nodes).to(device)
                cur_adj = adj * adj_mask * (1 - I) + I
                #                cur_adj = self.adjI * (1 + rarity_score_matrix) * adj_mask * (1 - I) + I
                h[nodes_need_update] = self.gated_update(
                    torch.cat([cur_obs, rarity_score.unsqueeze(-1)], dim=-1),
                    h, var_vector[nodes_need_update], cur_adj, nodes_need_update)

            end_sample_ind = torch.where(step == (lengths.squeeze(1) - 1))
            output[end_sample_ind[0]] = h[end_sample_ind[0]]
            if step == int(torch.max(lengths).item()) - 1:
                return output

        return output


class CGRNN_batch_woText(nn.Module):
    def __init__(self, d_in, d_model, num_nodes, rarity_alpha=0.5, var_emb_dim=5, var_graph_emb_dim=8):
        super(CGRNN_batch_woText, self).__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.num_nodes = num_nodes
        self.gated_update = AGCRNCellWithMLP(d_model, 2 * d_model, var_emb_dim)
        self.rarity_alpha = rarity_alpha
        self.rarity_W = nn.Parameter(torch.randn(num_nodes, num_nodes))
        self.relu = nn.ReLU()
        self.graph_emb = nn.Embedding(num_nodes, var_graph_emb_dim)
        self.state_emb = nn.Embedding(num_nodes, var_emb_dim)

    def init_hidden_states(self, x):
        return torch.zeros(size=(x.shape[0], x.shape[2], self.d_model)).to(x.device)

    def forward(self, obs_emb, adj, observed_mask, observed_tp, tp_emb_tensor, lengths, avg_interval,
                var_prior_emb_tensor):
        batch, steps, nodes, features = obs_emb.size()

        device = obs_emb.device

        h = self.init_hidden_states(obs_emb)

        I = repeat(torch.eye(nodes).to(device), 'v x -> b v x', b=batch)

        output = torch.zeros_like(h)

        nodes_initial_mask = torch.zeros(batch, nodes).to(device)

        var_total_obs = torch.sum(observed_mask, dim=1)

        var_prior_emb_tensor = repeat(var_prior_emb_tensor, "n d -> b n d", b=batch)

        var_vector = repeat(self.state_emb.weight, "n d -> b n d", b=batch)

        prior_graph_vector = repeat(self.graph_emb.weight, "n d -> b n d", b=batch)

        var_vector_nor = F.normalize(prior_graph_vector, p=2, dim=2)

        adj = torch.softmax(torch.bmm(var_vector_nor, var_vector_nor.permute(0, 2, 1)), dim=-1)

        for step in range(int(torch.max(lengths).item())):

            adj_mask = torch.zeros(size=[batch, nodes, nodes]).to(device)

            cur_obs = obs_emb[:, step]

            cur_mask = observed_mask[:, step]

            cur_obs_var = torch.where(cur_mask)

            nodes_initial_mask[cur_obs_var] = 1

            nodes_need_update = cur_obs_var

            cur_avg_interval = avg_interval[:, step]

            rarity_score = self.rarity_alpha * torch.tanh(cur_avg_interval / (var_total_obs + 1))

            rarity_score_matrix_row = repeat(rarity_score, 'b v -> b v x', x=nodes)
            rarity_score_matrix_col = repeat(rarity_score, 'b v -> b x v', x=nodes)

            rarity_score_matrix = -1 * self.rarity_W * (torch.abs(rarity_score_matrix_row - rarity_score_matrix_col))

            if nodes_need_update[0].shape[0] > 0:
                adj_mask[cur_obs_var[0], :, cur_obs_var[1]] = torch.ones(len(cur_obs_var[0]), nodes).to(device)
                wo_observed_nodes = torch.where(cur_mask == 0)
                wo_init_nodes = torch.where(nodes_initial_mask == 0)
                adj_mask[wo_observed_nodes] = torch.zeros(len(wo_observed_nodes[0]), nodes).to(device)

                cur_adj = adj * (1 + rarity_score_matrix) * adj_mask * (1 - I) + I

                h[nodes_need_update] = self.gated_update(
                    torch.cat([cur_obs, rarity_score.unsqueeze(-1)], dim=-1),
                    h, var_vector[nodes_need_update], cur_adj, nodes_need_update)

            end_sample_ind = torch.where(step == (lengths.squeeze(1) - 1))
            output[end_sample_ind[0]] = h[end_sample_ind[0]]
            if step == int(torch.max(lengths).item()) - 1:
                return output

        return output


class CGRNN_batch_wodyTo(nn.Module):
    def __init__(self, d_in, d_model, num_nodes, rarity_alpha=0.5, var_emb_dim=5, var_graph_emb_dim=8):
        super(CGRNN_batch_wodyTo, self).__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.num_nodes = num_nodes
        self.gated_update = AGCRNCellWithMLP(d_model, 2 * d_model, var_emb_dim)
        self.rarity_alpha = rarity_alpha
        self.rarity_W = nn.Parameter(torch.randn(num_nodes, num_nodes))
        self.relu = nn.ReLU()
        self.prior_to_state = MLP(768, 2 * d_model, var_emb_dim)
        self.prior_to_graph = MLP(768, 2 * d_model, var_graph_emb_dim)

    def init_hidden_states(self, x):
        return torch.zeros(size=(x.shape[0], x.shape[2], self.d_model)).to(x.device)

    def forward(self, obs_emb, adj, observed_mask, observed_tp, tp_emb_tensor, lengths, avg_interval,
                var_prior_emb_tensor):
        batch, steps, nodes, features = obs_emb.size()

        device = obs_emb.device

        h = self.init_hidden_states(obs_emb)

        I = repeat(torch.eye(nodes).to(device), 'v x -> b v x', b=batch)

        output = torch.zeros_like(h)

        nodes_initial_mask = torch.zeros(batch, nodes).to(device)

        var_total_obs = torch.sum(observed_mask, dim=1)

        var_prior_emb_tensor = repeat(var_prior_emb_tensor, "n d -> b n d", b=batch)

        var_vector = self.prior_to_state(var_prior_emb_tensor)

        prior_graph_vector = self.prior_to_graph(var_prior_emb_tensor)

        var_vector_nor = F.normalize(prior_graph_vector, p=2, dim=2)

        adj = torch.softmax(torch.bmm(var_vector_nor, var_vector_nor.permute(0, 2, 1)), dim=-1)

        for step in range(int(torch.max(lengths).item())):

            adj_mask = torch.zeros(size=[batch, nodes, nodes]).to(device)

            cur_obs = obs_emb[:, step]

            cur_mask = observed_mask[:, step]

            cur_obs_var = torch.where(cur_mask)

            nodes_initial_mask[cur_obs_var] = 1

            nodes_need_update = cur_obs_var

            cur_avg_interval = avg_interval[:, step]

            rarity_score = self.rarity_alpha * torch.tanh(cur_avg_interval / (var_total_obs + 1))
            #            rarity_score = self.rarity_alpha * self.rarity_W * torch.tanh(cur_avg_interval / (var_total_obs + 1))
            #            rarity_score_matrix = repeat(rarity_score, 'b v -> b v x', x=nodes)

            #            rarity_score = self.rarity_alpha * torch.tanh(cur_avg_interval / (var_total_obs + 1))
            rarity_score_matrix_row = repeat(rarity_score, 'b v -> b v x', x=nodes)
            rarity_score_matrix_col = repeat(rarity_score, 'b v -> b x v', x=nodes)
            #            rarity_score_matrix = -1* self.rarity_W *(rarity_score_matrix_row + rarity_score_matrix_col)
            #            rarity_score_matrix = rarity_score_matrix_row
            rarity_score_matrix = -1 * self.rarity_W * (torch.abs(rarity_score_matrix_row - rarity_score_matrix_col))
            # rarity_score_matrix = self.rarity_W * (rarity_score_matrix_row + rarity_score_matrix_col)
            if nodes_need_update[0].shape[0] > 0:
                adj_mask[cur_obs_var[0], :, cur_obs_var[1]] = torch.ones(len(cur_obs_var[0]), nodes).to(device)
                wo_observed_nodes = torch.where(cur_mask == 0)
                wo_init_nodes = torch.where(nodes_initial_mask == 0)
                adj_mask[wo_observed_nodes] = torch.zeros(len(wo_observed_nodes[0]), nodes).to(device)
                #                adj_mask[wo_observed_nodes] = torch.zeros(len(wo_observed_nodes[0]), nodes).to(device)

                #                cur_adj = adj * adj_mask * (1 - I) + I
                #                cur_adj = (adj * (1 + rarity_score_matrix) * adj_mask + I) / 2
                #                cur_adj = I
                #                cur_adj = self.adjI * adj_mask * (1 - I) + I
                #                cur_adj = I * adj_mask * (1 + rarity_score_matrix) * (1 - I) + I
                #                cur_adj = adj * (1 + rarity_score_matrix) * adj_mask
                #                cur_adj = adj * adj_mask * (1 - I) + I
                #                cur_adj = adj * adj_mask * (1 - I) + I
                cur_adj = adj * (1 + rarity_score_matrix) * (1 - I) + I
                #                cur_adj = self.adjI * (1 + rarity_score_matrix) * adj_mask * (1 - I) + I
                h[nodes_need_update] = self.gated_update(
                    torch.cat([cur_obs, rarity_score.unsqueeze(-1)], dim=-1),
                    h, var_vector[nodes_need_update], cur_adj, nodes_need_update)

            end_sample_ind = torch.where(step == (lengths.squeeze(1) - 1))
            output[end_sample_ind[0]] = h[end_sample_ind[0]]
            if step == int(torch.max(lengths).item()) - 1:
                return output

        return output


class CGRNN_batch(nn.Module):
    def __init__(self, d_in, d_model, num_nodes, rarity_alpha=0.5, var_emb_dim=5, var_graph_emb_dim=8):
        super(CGRNN_batch, self).__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.num_nodes = num_nodes
        self.gated_update = AGCRNCellWithMLP(d_model, 8 * d_model, var_emb_dim)
        self.rarity_alpha = rarity_alpha
        self.rarity_W = nn.Parameter(torch.randn(num_nodes, num_nodes))
        self.relu = nn.ReLU()
        self.prior_to_state = MLP(768, 2 * d_model, var_emb_dim)
        self.graph_emb = nn.Embedding(num_nodes, var_graph_emb_dim)
#        self.prior_to_graph = MLP(768, 2 * d_model, var_graph_emb_dim)
#        self.graph_emb = nn.Embedding(num_nodes, var_graph_emb_dim)
        
    def init_hidden_states(self, x):
        return torch.zeros(size=(x.shape[0], x.shape[2], self.d_model)).to(x.device)

    def forward(self, obs_emb, adj, observed_mask, observed_tp, tp_emb_tensor, lengths, avg_interval,
                var_prior_emb_tensor):
        batch, steps, nodes, features = obs_emb.size()

        device = obs_emb.device

        h = self.init_hidden_states(obs_emb)

        I = repeat(torch.eye(nodes).to(device), 'v x -> b v x', b=batch)

        output = torch.zeros_like(h)

        nodes_initial_mask = torch.zeros(batch, nodes).to(device)

        var_total_obs = torch.sum(observed_mask, dim=1)

        var_prior_emb_tensor = repeat(var_prior_emb_tensor, "n d -> b n d", b=batch)

        var_vector = self.prior_to_state(var_prior_emb_tensor)
#        var_vector = repeat(self.state_emb.weight, "n d -> b n d", b=batch)
#        var_vector = var_prior_emb_tensor

#        prior_graph_vector = self.prior_to_graph(var_prior_emb_tensor)
        prior_graph_vector = repeat(self.graph_emb.weight, "n d -> b n d", b=batch)

        var_vector_nor = F.normalize(prior_graph_vector, p=2, dim=2)

        adj = torch.softmax(torch.bmm(var_vector_nor, var_vector_nor.permute(0, 2, 1)), dim=-1)

        for step in range(int(torch.max(lengths).item())):

            adj_mask = torch.zeros(size=[batch, nodes, nodes]).to(device)

            cur_obs = obs_emb[:, step]

            cur_mask = observed_mask[:, step]

            cur_obs_var = torch.where(cur_mask)

            nodes_initial_mask[cur_obs_var] = 1

            nodes_need_update = cur_obs_var

            cur_avg_interval = avg_interval[:, step]

            rarity_score = self.rarity_alpha * torch.tanh(cur_avg_interval / (var_total_obs + 1))
            #            rarity_score = self.rarity_alpha * self.rarity_W * torch.tanh(cur_avg_interval / (var_total_obs + 1))
            #            rarity_score_matrix = repeat(rarity_score, 'b v -> b v x', x=nodes)

            #            rarity_score = self.rarity_alpha * torch.tanh(cur_avg_interval / (var_total_obs + 1))
            rarity_score_matrix_row = repeat(rarity_score, 'b v -> b v x', x=nodes)
            rarity_score_matrix_col = repeat(rarity_score, 'b v -> b x v', x=nodes)
            #            rarity_score_matrix = -1* self.rarity_W *(rarity_score_matrix_row + rarity_score_matrix_col)
            #            rarity_score_matrix = rarity_score_matrix_row
            rarity_score_matrix = -1 * self.rarity_W * (torch.abs(rarity_score_matrix_row - rarity_score_matrix_col))
            # rarity_score_matrix = self.rarity_W * (rarity_score_matrix_row + rarity_score_matrix_col)
            if nodes_need_update[0].shape[0] > 0:
                adj_mask[cur_obs_var[0], :, cur_obs_var[1]] = torch.ones(len(cur_obs_var[0]), nodes).to(device)
                wo_observed_nodes = torch.where(cur_mask == 0)
                wo_init_nodes = torch.where(nodes_initial_mask == 0)
                adj_mask[wo_observed_nodes] = torch.zeros(len(wo_observed_nodes[0]), nodes).to(device)
                #                adj_mask[wo_observed_nodes] = torch.zeros(len(wo_observed_nodes[0]), nodes).to(device)

                #                cur_adj = adj * adj_mask * (1 - I) + I
                #                cur_adj = (adj * (1 + rarity_score_matrix) * adj_mask + I) / 2
                #                cur_adj = I
                #                cur_adj = self.adjI * adj_mask * (1 - I) + I
                #                cur_adj = I * adj_mask * (1 + rarity_score_matrix) * (1 - I) + I
                #                cur_adj = adj * (1 + rarity_score_matrix) * adj_mask
                #                cur_adj = adj * adj_mask * (1 - I) + I
                #                cur_adj = adj * adj_mask * (1 - I) + I
                cur_adj = adj * (1 + rarity_score_matrix) * adj_mask * (1 - I) + I
                #                cur_adj = self.adjI * (1 + rarity_score_matrix) * adj_mask * (1 - I) + I
                h[nodes_need_update] = self.gated_update(
                    torch.cat([cur_obs, rarity_score.unsqueeze(-1)], dim=-1),
                    h, var_vector[nodes_need_update], cur_adj, nodes_need_update)

            end_sample_ind = torch.where(step == (lengths.squeeze(1) - 1))
            output[end_sample_ind[0]] = h[end_sample_ind[0]]
            if step == int(torch.max(lengths).item()) - 1:
                return output

        return output

class CGRNN_batch_Var(nn.Module):
    def __init__(self, d_in, d_model, num_nodes, rarity_alpha=0.5, var_emb_dim=5, var_graph_emb_dim=8):
        super(CGRNN_batch_Var, self).__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.num_nodes = num_nodes

        self.prior_to_state = MLP(768, 2 * d_model, var_emb_dim)
        self.prior_to_graph = MLP(768, 2 * d_model, var_graph_emb_dim)
        self.gated_update = AGCRNCellWithMLP_Var(d_model, 2 * d_model, num_nodes)

        self.rarity_alpha = rarity_alpha
        self.rarity_W = nn.Parameter(torch.randn(num_nodes, num_nodes))
        self.relu = nn.ReLU()



    def init_hidden_states(self, x):
        return torch.zeros(size=(x.shape[0], x.shape[2], self.d_model)).to(x.device)

    def forward(self, obs_emb, adj, observed_mask, observed_tp, tp_emb_tensor, lengths, avg_interval,
                var_prior_emb_tensor):
        batch, steps, nodes, features = obs_emb.size()

        device = obs_emb.device

        h = self.init_hidden_states(obs_emb)

        I = repeat(torch.eye(nodes).to(device), 'v x -> b v x', b=batch)

        output = torch.zeros_like(h)

        nodes_initial_mask = torch.zeros(batch, nodes).to(device)

        var_total_obs = torch.sum(observed_mask, dim=1)

        var_prior_emb_tensor = repeat(var_prior_emb_tensor, "n d -> b n d", b=batch)

        prior_graph_vector = self.prior_to_graph(var_prior_emb_tensor)

        var_vector_nor = F.normalize(prior_graph_vector, p=2, dim=2)

        adj = torch.softmax(torch.bmm(var_vector_nor, var_vector_nor.permute(0, 2, 1)), dim=-1)

        for step in range(int(torch.max(lengths).item())):

            adj_mask = torch.zeros(size=[batch, nodes, nodes]).to(device)

            cur_obs = obs_emb[:, step]

            cur_mask = observed_mask[:, step]

            cur_obs_var = torch.where(cur_mask)

            nodes_initial_mask[cur_obs_var] = 1

            nodes_need_update = cur_obs_var

            cur_avg_interval = avg_interval[:, step]

            rarity_score = self.rarity_alpha * torch.tanh(cur_avg_interval / (var_total_obs + 1))
            #            rarity_score = self.rarity_alpha * self.rarity_W * torch.tanh(cur_avg_interval / (var_total_obs + 1))
            #            rarity_score_matrix = repeat(rarity_score, 'b v -> b v x', x=nodes)

            #            rarity_score = self.rarity_alpha * torch.tanh(cur_avg_interval / (var_total_obs + 1))
            rarity_score_matrix_row = repeat(rarity_score, 'b v -> b v x', x=nodes)
            rarity_score_matrix_col = repeat(rarity_score, 'b v -> b x v', x=nodes)
            #            rarity_score_matrix = -1* self.rarity_W *(rarity_score_matrix_row + rarity_score_matrix_col)
            #            rarity_score_matrix = rarity_score_matrix_row
            rarity_score_matrix = -1 * self.rarity_W * (torch.abs(rarity_score_matrix_row - rarity_score_matrix_col))
            # rarity_score_matrix = self.rarity_W * (rarity_score_matrix_row + rarity_score_matrix_col)
            if nodes_need_update[0].shape[0] > 0:
                adj_mask[cur_obs_var[0], :, cur_obs_var[1]] = torch.ones(len(cur_obs_var[0]), nodes).to(device)
                wo_observed_nodes = torch.where(cur_mask == 0)
                wo_init_nodes = torch.where(nodes_initial_mask == 0)
                adj_mask[wo_observed_nodes] = torch.zeros(len(wo_observed_nodes[0]), nodes).to(device)
                #                adj_mask[wo_observed_nodes] = torch.zeros(len(wo_observed_nodes[0]), nodes).to(device)

                #                cur_adj = adj * adj_mask * (1 - I) + I
                #                cur_adj = (adj * (1 + rarity_score_matrix) * adj_mask + I) / 2
                #                cur_adj = I
                #                cur_adj = self.adjI * adj_mask * (1 - I) + I
                #                cur_adj = I * adj_mask * (1 + rarity_score_matrix) * (1 - I) + I
                #                cur_adj = adj * (1 + rarity_score_matrix) * adj_mask
                #                cur_adj = adj * adj_mask * (1 - I) + I
                #                cur_adj = adj * adj_mask * (1 - I) + I
                cur_adj = adj * (1 + rarity_score_matrix) * adj_mask * (1 - I) + I
                #                cur_adj = self.adjI * (1 + rarity_score_matrix) * adj_mask * (1 - I) + I
                h[nodes_need_update] = self.gated_update(
                    torch.cat([cur_obs, rarity_score.unsqueeze(-1)], dim=-1),
                    h, None, cur_adj, nodes_need_update)

            end_sample_ind = torch.where(step == (lengths.squeeze(1) - 1))
            output[end_sample_ind[0]] = h[end_sample_ind[0]]
            if step == int(torch.max(lengths).item()) - 1:
                return output

        return output



class newModel(nn.Module):

    def __init__(self, DEVICE, graph_node_d_model, num_of_vertices, num_of_tp, d_static,
                 n_class, node_enc_layer=2, rarity_alpha=0.5, var_emb_dim=5, var_graph_emb_dim=8):

        super(newModel, self).__init__()
        self.num_of_vertices = num_of_vertices
        self.num_of_tp = num_of_tp
        self.graph_node_d_model = graph_node_d_model
        self.adj = nn.Parameter(torch.ones(size=[num_of_vertices, num_of_vertices]))

#        self.prior_emb = nn.Embedding(num_of_vertices, 768)

        self.value_enc = Value_Encoder(output_dim=graph_node_d_model)

        self.abs_time_enc = Time_Encoder(embed_time=graph_node_d_model, var_num=num_of_vertices)

        self.obs_tp_enc = nn.GRU(input_size=graph_node_d_model, hidden_size=graph_node_d_model,
                                 num_layers=node_enc_layer, batch_first=True, bidirectional=False)
        #        self.prior_to_state = nn.Linear(768, graph_node_d_model)
        self.obs_enc = nn.Sequential(
            nn.Linear(in_features=6 * graph_node_d_model, out_features=graph_node_d_model),
            nn.ReLU()
        )

        self.emb1 = nn.Embedding(num_of_vertices, graph_node_d_model)

        self.GCRNN = CGRNN_batch(d_in=self.graph_node_d_model, d_model=self.graph_node_d_model,
                                 num_nodes=num_of_vertices, rarity_alpha=rarity_alpha,
                                 var_emb_dim=var_emb_dim, var_graph_emb_dim=var_graph_emb_dim)

        self.final_conv = nn.Conv2d(graph_node_d_model, 1, kernel_size=1)
        self.d_static = d_static
        if d_static != 0:
            self.emb = nn.Linear(d_static, num_of_vertices)
            self.classifier = nn.Sequential(
                nn.Linear(num_of_vertices * 2, 200),
                nn.ReLU(),
                nn.Linear(200, n_class)).to(DEVICE)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(num_of_vertices, 200),
                nn.ReLU(),
                nn.Linear(200, n_class))
        self.DEVICE = DEVICE

        self.to(DEVICE)

    def forward(self, P, P_fft, P_static, P_avg_interval, P_length, P_time_pre, P_time, P_var_prior_emb_tensor):
        b, t, v = P.shape
        v = v // 2
        observed_data = P[:, :, :v]
        observed_mask = P[:, :, v:]

        value_emb = self.value_enc(observed_data) * observed_mask.unsqueeze(-1)

        abs_time_emb = self.abs_time_enc(P_time) * observed_mask.unsqueeze(-1)

#        P_var_prior_emb_tensor = self.prior_emb.weight

        E_1 = repeat(self.emb1.weight, 'v d -> b v d', b=b)

        cos_sim = nn.functional.cosine_similarity(P_var_prior_emb_tensor.unsqueeze(1),
                                                  P_var_prior_emb_tensor.unsqueeze(0), dim=2)
        cos_sim = torch.softmax(cos_sim, dim=-1)

        adj = repeat(cos_sim, 'v d -> b v d', b=b)

        P_time = P_time * observed_mask

        obs_emb = (value_emb + abs_time_emb + repeat(E_1, 'b v d -> b t v d', t=t)) * observed_mask.unsqueeze(-1)

        spatial_gcn = self.GCRNN(obs_emb, adj, observed_mask, P_time, None, P_length, P_avg_interval,
                                 P_var_prior_emb_tensor)

        if P_static is not None:
            static_emb = self.emb(P_static)
            return self.classifier(torch.cat([torch.sum(spatial_gcn, dim=-1), static_emb], dim=-1))
        else:
            return self.classifier(torch.sum(spatial_gcn, dim=-1))


