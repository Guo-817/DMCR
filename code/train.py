import torch
import torch.nn as nn
import numpy as np
import random
import sys
import copy
from time import time
from helper import *
from batch_test import *
from utils import *
from model import *

if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(2025)

    cfg_dict = dict()
    cfg_dict['device'] = device
    cfg_dict['n_users'] = data_generator.n_users
    cfg_dict['n_items'] = data_generator.n_items
    cfg_dict['cris'] = data_generator.cris
    cfg_dict['trn_mat'] = data_generator.trnMats[-1]
    cfg_dict['users_dict'] = data_generator.trnDicts

    adj_collection = data_generator.get_adj_mat()
    cfg_dict['pre_adjs'] = adj_collection
    user_num, item_num = data_generator.n_users, data_generator.n_items
    crit_list = data_generator.cris
    crit_count = data_generator.cri_num

    train_dict = copy.deepcopy(data_generator.trnDicts)
    item_max_group = []
    label_group = []

    for i in range(crit_count):
        max_i, label_set = reshape_user_record_dict(train_dict[i], pad_val=item_num)
        item_max_group.append(max_i)
        label_group.append(label_set)

    t0 = time()
    cfg_dict['max_item_list'] = item_max_group

    model = DMCR(item_max_group, data_config=cfg_dict, args=args).to(device)
    loss_module = RecLoss(data_config=cfg_dict, args=args).to(device)
    bpr_module = BprLoss(data_config=cfg_dict, args=args).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_score = 0.
    run_tag = 1

    loss_log, pre_log, rec_log, ndcg_log, hit_log = [], [], [], [], []

    stop_counter = 0
    early_stop = False

    user_index_set, label_item_group = generate_training_patterns(item_max_group, label_group,
                                                             n_items=item_num, n_cris=crit_count)

    non_share_id = -1

    for ep in range(args.epoch):
        model.train()

        perm_indices = np.random.permutation(np.arange(len(user_index_set)))
        user_index_set = user_index_set[perm_indices]
        label_item_group = [group[perm_indices] for group in label_item_group]

        t1 = time()
        total_loss = total_rec = total_emb = total_ssl = total_kl = total_bpr = 0.
        batch_total = int(len(user_index_set) / args.batch_size)
        subgraph_data = {}

        for b_idx in range(batch_total):
            optimizer.zero_grad()
            st_idx = b_idx * args.batch_size
            ed_idx = min((b_idx + 1) * args.batch_size, len(user_index_set))

            user_sub = user_index_set[st_idx:ed_idx]
            label_sub = [group[st_idx:ed_idx] for group in label_item_group]

            u_ids, i_ids = form_user_item_pairs(user_batch=user_sub,
                                           criteria_targets=label_sub[-1],
                                           total_items=item_num)

            user_tensor = torch.from_numpy(user_sub).to(device)

            label_tensor_list = [torch.from_numpy(arr).to(device) for arr in label_sub]
            u_ids_tensor = torch.from_numpy(u_ids).to(device)
            i_ids_tensor = torch.from_numpy(i_ids).to(device)

            ua_emb, ia_emb, crit_emb = model(subgraph_data, device)

            rec_l, emb_l = loss_module(user_tensor, label_tensor_list, ua_emb, ia_emb, crit_emb)
            bpr_l = bpr_module(user_tensor, ua_emb, ia_emb)

            total_batch_loss = rec_l + emb_l + bpr_l
            total_batch_loss.backward()
            optimizer.step()

            total_loss += total_batch_loss.item() / batch_total
            total_rec += rec_l.item() / batch_total
            total_emb += emb_l.item() / batch_total
            total_bpr += bpr_l.item() / batch_total

        torch.cuda.empty_cache()

        if np.isnan(total_loss):
            print('ERROR: loss is nan.')
            sys.exit()

        t2 = time()
        model.eval()
        with torch.no_grad():
            ua_emb, ia_emb, crit_emb = model(subgraph_data, device)
            test_users = list(data_generator.test_set.keys())
            eval_result = model_assessor(ua_emb.cpu().numpy(),
                                    ia_emb.cpu().numpy(),
                                    crit_emb, test_users, crit_list)

        t3 = time()

        loss_log.append(total_loss)
        rec_log.append(eval_result['recall'])
        pre_log.append(eval_result['precision'])
        ndcg_log.append(eval_result['ndcg'])
        hit_log.append(eval_result['hit_ratio'])

        if args.verbose > 0:
            print('Epoch %d:  Precision=[%.4f, %.4f], Recall=[%.4f, %.4f], Hit=[%.4f, %.4f], NDCG=[%.4f, %.4f]' %
                  (ep, eval_result['precision'][0], eval_result['precision'][1],
                   eval_result['recall'][0], eval_result['recall'][1],
                   eval_result['hit_ratio'][0], eval_result['hit_ratio'][1],
                   eval_result['ndcg'][0], eval_result['ndcg'][1]))

        best_score, stop_counter, early_stop, flag = early_stopping_new(
            eval_result['recall'][0], best_score, stop_counter,
            expected_order='acc', flag_step=10)

        if early_stop:
            break

    rec_arr = np.array(rec_log)
    pre_arr = np.array(pre_log)
    ndcg_arr = np.array(ndcg_log)
    hit_arr = np.array(hit_log)

    best_idx = np.argmax(rec_arr[:, 0])
    best_output = "Best Iter=[%d]\trecall=[%s], ndcg=[%s]" % (
        best_idx,
        '\t'.join(['%.4f' % v for v in rec_arr[best_idx]]),
        '\t'.join(['%.4f' % v for v in ndcg_arr[best_idx]])
    )
    print(best_output)
