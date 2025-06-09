import numpy as np
import torch
import time
import random
from sklearn.metrics import roc_auc_score
from batch_test import *
from collections import defaultdict

def model_assessor(user_tensor, item_tensor, crit_tensor, eval_uids, crit_ids, batch_eval=False):
    def sigmoid_fn(matrix):
        return 1. / (1. + np.exp(-matrix))

    def normalize_matrix(mat, axis=1):
        norms = np.linalg.norm(mat, axis=axis, keepdims=True)
        return mat / (norms + 1e-8)

    def aggregate_user_features(user_embed, crit_count):
        aggregate = user_embed[:, 0, :].copy()
        for idx in range(1, crit_count):
            aggregate += user_embed[:, idx, :]
        return aggregate / crit_count

    def compute_user_item_scores(U, I, C, uid_seq, iid_seq):
        crit_len = len(crit_ids)
        score_matrix = 0.0

        user_repr = aggregate_user_features(U[uid_seq], crit_len)
        u_unit = normalize_matrix(user_repr)

        for c_idx in range(crit_len):
            u_slice = U[uid_seq, c_idx, :]
            i_slice = I[iid_seq, c_idx, :]
            crit_vec = C[crit_ids[c_idx]].detach().cpu().numpy()

            weighted_items = i_slice * crit_vec
            dot_prod = np.matmul(u_slice, weighted_items.T)

            user_alignment = np.matmul(user_repr, u_unit.T)
            user_alignment_sig = sigmoid_fn(user_alignment)
            score_sig = sigmoid_fn(dot_prod)
            mix = np.matmul(user_alignment_sig, score_sig)

            modulated = dot_prod * mix
            score_matrix += sigmoid_fn(modulated)

        return score_matrix / crit_len

    # Initialize final results
    eval_result = {
        'precision': np.zeros(len(Ks)),
        'recall': np.zeros(len(Ks)),
        'ndcg': np.zeros(len(Ks)),
        'hit_ratio': np.zeros(len(Ks))
    }

    target_users = eval_uids
    num_targets = len(target_users)

    task_pool = multiprocessing.Pool(cores)

    u_slice_size = BATCH_SIZE
    u_slices = num_targets // u_slice_size + 1
    count_acc = 0

    for slice_id in range(u_slices):
        from_idx = slice_id * u_slice_size
        to_idx = (slice_id + 1) * u_slice_size

        slice_users = target_users[from_idx: to_idx]
        all_items = list(range(ITEM_NUM))

        pred_scores = compute_user_item_scores(user_tensor, item_tensor, crit_tensor, slice_users, all_items)
        zipped_inputs = zip(pred_scores, slice_users)

        eval_batch = task_pool.map(test_one_user, zipped_inputs)
        count_acc += len(eval_batch)

        for eval_metrics in eval_batch:
            for key in eval_result:
                eval_result[key] += eval_metrics[key] / num_targets

    assert count_acc == num_targets
    task_pool.close()
    return eval_result


def form_user_item_pairs(user_batch, criteria_targets, total_items):
    u_collector, i_collector = [], []

    for idx, user_vec in enumerate(user_batch):
        valid_items = criteria_targets[idx][criteria_targets[idx] != total_items]
        uid_scalar = user_vec[0]
        u_collector.extend([uid_scalar] * valid_items.shape[0])
        i_collector.extend(valid_items.tolist())

    user_output = np.asarray(u_collector).reshape(-1)
    item_output = np.asarray(i_collector).reshape(-1)

    return user_output, item_output

def reshape_user_record_dict(record_map, pad_val, top_ratio=0.9999):
    lengths = [len(v) for v in record_map.values()]
    sorted_lengths = sorted(lengths)

    max_len = sorted_lengths[int(len(sorted_lengths) * top_ratio) - 1]

    padded_map = {}
    for uid, items in record_map.items():
        truncated = items[:max_len]
        if len(truncated) < max_len:
            padded = truncated + [pad_val] * (max_len - len(truncated))
        else:
            padded = truncated
        padded_map[uid] = padded

    return max_len, padded_map

def generate_training_patterns(max_item_list, cri_label_list, n_items, n_cris):
    user_train = []
    cri_item_list = [list() for i in range(n_cris)]  #

    for i in cri_label_list[-1].keys():
        user_train.append(i)
        cri_item_list[-1].append(cri_label_list[-1][i])
        for j in range(n_cris - 1):
            if not i in cri_label_list[j].keys():
                cri_item_list[j].append([n_items] * max_item_list[j])
            else:
                cri_item_list[j].append(cri_label_list[j][i])

    user_train = np.array(user_train)
    cri_item_list = [np.array(cri_item) for cri_item in cri_item_list]
    user_train = user_train[:, np.newaxis]
    return user_train, cri_item_list


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
