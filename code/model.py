import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from helper import *
from batch_test import *
import random
import numpy as np
from utils import *


class DMCR(nn.Module):
    name = 'DMCR'

    def __init__(self, max_item_len, data_config, args):
        super(DMCR, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.max_item_len = max_item_len
        self.num_users = data_config['n_users']
        self.num_items = data_config['n_items']
        self.total_nodes = self.num_users + self.num_items

        self.adj_matrices = data_config['pre_adjs']
        self.graph_tensors = [self._sp_mat_to_tensor(adj).to(device) for adj in self.adj_matrices]

        self.criteria_names = data_config['cris']
        self.criteria_count = len(self.criteria_names)

        self.alpha = torch.tensor(eval(args.coefficient)).view(1, -1).to(device)

        self.embed_dim = args.embed_size
        self.batch_sz = args.batch_size
        self.layer_dims = eval(args.layer_size)
        self.depth = len(self.layer_dims)
        self.dropout_rates = eval(args.mess_dropout)
        self.heads = args.nhead
        self.attn_dim = args.att_dim

        param_dict = {}
        param_dict['crit_embed'] = Parameter(torch.FloatTensor(self.criteria_count, self.embed_dim))
        param_dict['gcn_weight'] = Parameter(torch.FloatTensor(self.embed_dim, self.embed_dim))
        param_dict['user_embed'] = Parameter(torch.FloatTensor(self.num_users, self.criteria_count, self.embed_dim))
        param_dict['item_embed'] = Parameter(torch.FloatTensor(self.num_items, self.criteria_count, self.embed_dim))

        full_dims = [self.embed_dim] + self.layer_dims
        for l in range(self.depth):
            param_dict[f'gcn_layer_{l}'] = Parameter(torch.FloatTensor(full_dims[l], full_dims[l + 1]))
            param_dict[f'crit_layer_{l}'] = Parameter(torch.FloatTensor(full_dims[l], full_dims[l + 1]))

        param_dict['attn_proj_1'] = Parameter(
            torch.FloatTensor(self.criteria_count, self.embed_dim, self.attn_dim))
        param_dict['attn_proj_2'] = Parameter(
            torch.FloatTensor(self.criteria_count, self.attn_dim, 1))

        self.params = nn.ParameterDict(param_dict)
        self.dropout = nn.Dropout(self.dropout_rates[0], inplace=True)
        self.activation = nn.LeakyReLU(inplace=True)

        self._init_parameters()

    def _init_parameters(self):
        nn.init.xavier_uniform_(self.params['user_embed'])
        nn.init.xavier_uniform_(self.params['item_embed'])
        nn.init.xavier_uniform_(self.params['crit_embed'])
        nn.init.xavier_uniform_(self.params['attn_proj_1'])
        nn.init.xavier_uniform_(self.params['attn_proj_2'])
        nn.init.xavier_uniform_(self.params['gcn_weight'])

        for l in range(self.depth):
            nn.init.xavier_uniform_(self.params[f'gcn_layer_{l}'])
            nn.init.xavier_uniform_(self.params[f'crit_layer_{l}'])

    def _sp_mat_to_tensor(self, matrix):
        coo = matrix.tocoo()
        idx = np.vstack((coo.row, coo.col))
        return torch.sparse.FloatTensor(
            torch.LongTensor(idx),
            torch.FloatTensor(coo.data),
            torch.Size(coo.shape)
        )

    def forward(self, subgraphs, device):
        user_emb = self.params['user_embed']
        item_emb = self.params['item_embed']
        crit_embeds = self.params['crit_embed']
        ego = torch.cat([user_emb, item_emb], dim=0)

        feature_bank = ego
        criteria_embeds = {name: [crit_embeds[i].view(-1, self.embed_dim)] for i, name in enumerate(self.criteria_names)}

        mm_total_time = 0.

        for l in range(self.depth):
            criterion_features = []
            for c_idx in range(self.criteria_count):
                start_time = time()
                graph_feat = torch.matmul(self.graph_tensors[c_idx], ego[:, c_idx, :])
                graph_feat = torch.matmul(graph_feat, self.params['gcn_weight'])
                mm_total_time += time() - start_time

                crit_vec = criteria_embeds[self.criteria_names[c_idx]][l]
                combined = torch.mul(graph_feat, crit_vec)
                out_feat = self.activation(torch.matmul(combined, self.params[f'gcn_layer_{l}']))
                criterion_features.append(out_feat)

            stacked_feat = torch.stack(criterion_features, dim=1)

            attn_outputs, attention_weights = [], []
            for c_idx in range(self.criteria_count):
                raw_scores = torch.matmul(
                    torch.tanh(torch.matmul(stacked_feat, self.params['attn_proj_1'][c_idx])),
                    self.params['attn_proj_2'][c_idx]
                ).squeeze(2)
                attn_weights = F.softmax(raw_scores, dim=1).unsqueeze(1)
                attention_weights.append(attn_weights)
                out_vec = torch.matmul(attn_weights, stacked_feat).squeeze(1)
                attn_outputs.append(out_vec)

            ego = torch.stack(attn_outputs, dim=1)
            combined_attn = torch.cat(attention_weights, dim=1)

            ego = self.dropout(ego)
            feature_bank += ego

            for c_idx in range(self.criteria_count):
                crit_proj = torch.matmul(criteria_embeds[self.criteria_names[c_idx]][l],
                                         self.params[f'crit_layer_{l}'])
                criteria_embeds[self.criteria_names[c_idx]].append(crit_proj)

        feature_bank /= (self.depth + 1)
        user_out, item_out = torch.split(feature_bank, [self.num_users, self.num_items], dim=0)

        dummy_token = torch.zeros([1, self.criteria_count, self.embed_dim], device=device)
        item_out = torch.cat((item_out, dummy_token), dim=0)

        for key in criteria_embeds:
            criteria_embeds[key] = torch.mean(torch.stack(criteria_embeds[key], dim=0), dim=0)

        return user_out, item_out, criteria_embeds


class BprLoss(nn.Module):
    def __init__(self, data_config, args):
        super(BprLoss, self).__init__()
        self.criteria_list = data_config['cris']
        self.criteria_num = len(self.criteria_list)
        self.user_total = data_config['n_users']
        self.item_total = data_config['n_items']
        self.user_history_per_crit = data_config['users_dict']
        self.max_seq_len = data_config['max_item_list']
        self.embedding_size = args.embed_size

    def _draw_positive_items(self, u_dict, u_id, num_pos):
        positives = u_dict[u_id]
        selected = []
        while len(selected) < num_pos:
            chosen = positives[np.random.randint(len(positives))]
            if num_pos > len(positives) or chosen not in selected:
                selected.append(chosen)
        return selected

    def _draw_negative_items(self, u_dict, u_id, num_neg):
        positives = u_dict[u_id]
        negatives = []
        while len(negatives) < num_neg:
            sample = np.random.randint(self.item_total)
            if sample not in positives and sample not in negatives:
                negatives.append(sample)
        return negatives

    def _construct_batch_triplets(self, u_dict, batch_sz):
        user_ids = list(u_dict.keys())
        if batch_sz <= len(user_ids):
            sampled_users = random.sample(user_ids, len(user_ids))
        else:
            sampled_users = [random.choice(user_ids) for _ in range(batch_sz)]

        pos_list, neg_list = [], []
        for uid in sampled_users:
            pos_list += self._draw_positive_items(u_dict, uid, 1)
            neg_list += self._draw_negative_items(u_dict, uid, 1)

        user_tensor = torch.LongTensor(sampled_users)
        pos_tensor = torch.LongTensor(pos_list)
        neg_tensor = torch.LongTensor(neg_list)
        return user_tensor, pos_tensor, neg_tensor

    def _l2_reg(self, tensor):
        return torch.mean(torch.sum(tensor ** 2, dim=1) / 2.)

    def forward(self, users_in_batch, user_embeds, item_embeds):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        user_embed_input = torch.reshape(user_embeds[users_in_batch], (-1, self.criteria_num, self.embedding_size))
        total_loss = 0

        for crit_idx in range(self.criteria_num):
            u_batch, pos_batch, neg_batch = self._construct_batch_triplets(self.user_history_per_crit[crit_idx],
                                                                            len(users_in_batch))
            u_batch, pos_batch, neg_batch = u_batch.to(device), pos_batch.to(device), neg_batch.to(device)

            u_vec = torch.reshape(user_embeds[u_batch], (-1, self.criteria_num, self.embedding_size))[:, crit_idx, :]
            p_vec = item_embeds[:, crit_idx, :][pos_batch]
            n_vec = item_embeds[:, crit_idx, :][neg_batch]

            if crit_idx == self.criteria_num - 1:
                sim_pos = torch.sigmoid(torch.sum(u_vec * p_vec, dim=1))
                sim_neg = torch.sigmoid(torch.sum(u_vec * n_vec, dim=1))

                combined_pos = 0.0
                combined_neg = 0.0
                K = self.criteria_num - 1
                for k in range(K):
                    u_k = torch.nn.functional.normalize(user_embeds[u_batch][:, k, :], p=2, dim=1)
                    i_pos_k = item_embeds[:, k, :][pos_batch]
                    i_neg_k = item_embeds[:, k, :][neg_batch]
                    user_sim = torch.sigmoid(torch.sum(u_vec * u_k, dim=1))
                    p_score_k = torch.sigmoid(torch.sum(u_k * i_pos_k, dim=1))
                    n_score_k = torch.sigmoid(torch.sum(u_k * i_neg_k, dim=1))
                    combined_pos += user_sim * p_score_k
                    combined_neg += user_sim * n_score_k
                combined_pos /= K
                combined_neg /= K
                final_pos_score = combined_pos * sim_pos
                final_neg_score = combined_neg * sim_neg
            else:
                final_pos_score = torch.sigmoid(torch.sum(u_vec * p_vec, dim=1))
                final_neg_score = torch.sigmoid(torch.sum(u_vec * n_vec, dim=1))

            loss_bpr = -F.logsigmoid(final_pos_score - final_neg_score).mean()
            reg_term = self._l2_reg(u_vec) + self._l2_reg(p_vec) + self._l2_reg(n_vec)

            if crit_idx != self.criteria_num - 1:
                loss_bpr *= 1e-2

            total_loss += loss_bpr + reg_term

        return total_loss / 5



class RecLoss(nn.Module):
    def __init__(self, data_config, args):
        super(RecLoss, self).__init__()
        self.criteria_list = data_config['cris']
        self.num_criteria = len(self.criteria_list)
        self.user_total = data_config['n_users']
        self.item_total = data_config['n_items']
        self.embedding_size = args.embed_size

        self.alpha_weights = eval(args.coefficient)
        self.beta_weights = eval(args.wid)

    def forward(self, input_u, label_phs, ua_embeddings, ia_embeddings, crit_embeddings):
        user_embed_slice = ua_embeddings[input_u]
        user_embed_slice = torch.reshape(user_embed_slice, (-1, self.num_criteria, self.embedding_size))

        crit_scores = []
        for idx in range(self.num_criteria):
            crit_key = self.criteria_list[idx]
            selected_items = ia_embeddings[:, idx, :][label_phs[idx]]
            valid_item_mask = torch.ne(label_phs[idx], self.item_total).float()
            masked_items = torch.einsum('ab,abc->abc', valid_item_mask, selected_items)
            interaction_score = torch.einsum('ac,abc->abc', user_embed_slice[:, idx, :], masked_items)
            interaction_score = torch.einsum('ajk,lk->aj', interaction_score, crit_embeddings[crit_key])
            crit_scores.append(interaction_score)

        total_loss = 0.
        for idx in range(self.num_criteria):
            crit_key = self.criteria_list[idx]
            item_sim = torch.einsum('ab,ac->bc', ia_embeddings[:, idx, :], ia_embeddings[:, idx, :])
            user_sim = torch.einsum('ab,ac->bc', user_embed_slice[:, idx, :], user_embed_slice[:, idx, :])
            combined_sim = item_sim * user_sim
            current_loss = self.beta_weights[idx] * torch.sum(
                combined_sim * torch.matmul(crit_embeddings[crit_key].T, crit_embeddings[crit_key])
            )
            current_loss += torch.sum((1.0 - self.beta_weights[idx]) * torch.square(crit_scores[idx]) - 2.0 * crit_scores[idx])
            total_loss += self.alpha_weights[idx] * current_loss

        reg_term = 0.5 * (torch.sum(torch.square(user_embed_slice)) + torch.sum(torch.square(ia_embeddings)))
        reg_loss = args.decay * reg_term

        return total_loss, reg_loss