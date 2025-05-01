import tensorflow as tf
import numpy as np
import sys
import copy
import random
import time
import multiprocessing
from helper import *
from batch_test import *
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class DMCR(tf.keras.Model):
    name = 'DMCR'

    def __init__(self, max_item_list, data_config, args):
        super(DMCR, self).__init__()
        self.max_item_list = max_item_list
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.num_nodes = self.n_users + self.n_items
        self.pre_adjs = data_config['pre_adjs']
        self.pre_adjs_tensor = [self._convert_sp_mat_to_sp_tensor(adj) for adj in self.pre_adjs]
        self.cris = data_config['cris']
        self.n_criterion = len(self.cris)

        self.coefficient = tf.constant(eval(args.coefficient), shape=(1, -1), dtype=tf.float32)
        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size
        self.weight_size = eval(args.layer_size)
        self.n_layers = len(self.weight_size)
        self.mess_dropout = eval(args.mess_dropout)
        self.nhead = args.nhead
        self.att_dim = args.att_dim

        self.criterion_embedding = tf.Variable(
            tf.random.normal([self.n_criterion, self.emb_dim], stddev=0.01))
        self.w_gcn = tf.Variable(
            tf.random.normal([self.emb_dim, self.emb_dim], stddev=0.01))
        
        self.user_embedding = tf.Variable(
            tf.random.normal([self.n_users, self.n_criterion, self.emb_dim], stddev=0.01))
        self.item_embedding = tf.Variable(
            tf.random.normal([self.n_items, self.n_criterion, self.emb_dim], stddev=0.01))

        self.weight_size_list = [self.emb_dim] + self.weight_size

        self.W_gc = []
        self.W_rel = []
        for k in range(self.n_layers):
            self.W_gc.append(tf.Variable(
                tf.random.normal([self.weight_size_list[k], self.weight_size_list[k+1]], stddev=0.01)))
            self.W_rel.append(tf.Variable(
                tf.random.normal([self.weight_size_list[k], self.weight_size_list[k+1]], stddev=0.01)))

        self.trans_weights_s1 = tf.Variable(
            tf.random.normal([self.n_criterion, self.emb_dim, self.att_dim], stddev=0.01))
        self.trans_weights_s2 = tf.Variable(
            tf.random.normal([self.n_criterion, self.att_dim, 1], stddev=0.01))
        
        self.dropout = tf.keras.layers.Dropout(self.mess_dropout[0])
        self.leaky_relu = tf.keras.layers.LeakyReLU()

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def call(self, sub_mats, training=False):
        pre_embeddings = tf.concat([self.user_embedding, self.item_embedding], axis=0)
        users_items_embeddings = pre_embeddings
        all_cri_embeddings_ = {}
        for i in range(self.n_criterion):
            beh = self.cris[i]
            cris_emb = tf.reshape(self.criterion_embedding[i], (-1, self.emb_dim))
            all_cri_embeddings_[beh] = [cris_emb]

        total_mm_time = 0.
        for k in range(self.n_layers):
            embeddings_list = []
            for i in range(self.n_criterion):
                st = time.time()
                
                embeddings_ = tf.sparse.sparse_dense_matmul(self.pre_adjs_tensor[i], pre_embeddings[:, i, :])
                embeddings_ = tf.matmul(embeddings_, self.w_gcn)
                
                total_mm_time += time.time() - st
                cris_emb = all_cri_embeddings_[self.cris[i]][k]
                embeddings_ = self.leaky_relu(
                    tf.matmul(tf.multiply(embeddings_, cris_emb), self.W_gc[k]))
                embeddings_list.append(embeddings_)

            embeddings_st = tf.stack(embeddings_list, axis=1)
            embeddings_list = []
            attention_list = []
            for i in range(self.n_criterion):
                attention = tf.nn.softmax(
                    tf.squeeze(
                        tf.matmul(
                            tf.tanh(tf.matmul(embeddings_st, self.trans_weights_s1[i])),
                            self.trans_weights_s2[i]
                        ),
                        axis=2
                    ),
                    axis=1
                )
                attention = tf.expand_dims(attention, axis=1)
                attention_list.append(attention)
                embs_cur_cris = tf.squeeze(tf.matmul(attention, embeddings_st), axis=1)
                embs_cur_cris = self.leaky_relu(embs_cur_cris)
                embeddings_list.append(embs_cur_cris)

            pre_embeddings = tf.stack(embeddings_list, axis=1)
            attn = tf.concat(attention_list, axis=1)
            pre_embeddings = self.dropout(pre_embeddings, training=training)
            users_items_embeddings = users_items_embeddings + pre_embeddings

            for i in range(self.n_criterion):
                cris_emb = tf.matmul(all_cri_embeddings_[self.cris[i]][k], self.W_rel[k])
                cris_emb = self.leaky_relu(cris_emb)
                all_cri_embeddings_[self.cris[i]].append(cris_emb)

        users_items_embeddings = users_items_embeddings / (self.n_layers + 1)
        users_embedding_, items_embedding_ = tf.split(users_items_embeddings, [self.n_users, self.n_items], axis=0)
        token_embedding = tf.zeros([1, self.n_criterion, self.emb_dim], dtype=tf.float32)
        items_embedding_ = tf.concat([items_embedding_, token_embedding], axis=0)

        for i in range(self.n_criterion):
            all_cri_embeddings_[self.cris[i]] = tf.reduce_mean(tf.stack(all_cri_embeddings_[self.cris[i]], axis=0), axis=0)

        return users_embedding_, items_embedding_, all_cri_embeddings_


class BprLoss(tf.keras.layers.Layer):
    def __init__(self, data_config, args):
        super(BprLoss, self).__init__()
        self.cris = data_config['cris']
        self.n_criterion = len(self.cris)
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.users_dict_list = data_config['users_dict']
        self.max_item_list = config['max_item_list']
        self.emb_dim = args.embed_size

    def sample_pos_items_for_u(self, user_dict, user_id, n_sample_pos_items):
        pos_items = user_dict[user_id]
        n_pos_items = len(pos_items)

        sample_pos_items = []
        while True:
            if len(sample_pos_items) == n_sample_pos_items:
                break

            pos_item_idx = np.random.randint(low=0, high=n_pos_items, size=1)[0]
            pos_item_id = pos_items[pos_item_idx]
            if n_pos_items < n_sample_pos_items:
                sample_pos_items.append(pos_item_id)
            else:
                if pos_item_id not in sample_pos_items:
                    sample_pos_items.append(pos_item_id)
        return sample_pos_items

    def sample_neg_items_for_u(self, user_dict, user_id, n_sample_neg_items):
        pos_items = user_dict[user_id]

        sample_neg_items = []
        while True:
            if len(sample_neg_items) == n_sample_neg_items:
                break

            neg_item_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
            if neg_item_id not in pos_items and neg_item_id not in sample_neg_items:
                sample_neg_items.append(neg_item_id)
        return sample_neg_items

    def generate_cf_batch(self, user_dict, batch_size):
        exist_users = list(user_dict.keys())
        if batch_size <= len(exist_users):
            batch_user = random.sample(exist_users, len(exist_users))
        else:
            batch_user = [random.choice(exist_users) for _ in range(batch_size)]

        batch_pos_item, batch_neg_item = [], []
        for u in batch_user:
            batch_pos_item += self.sample_pos_items_for_u(user_dict, u, 1)
            batch_neg_item += self.sample_neg_items_for_u(user_dict, u, 1)

        batch_user = [x for i in range(1) for x in batch_user]
        batch_user = tf.convert_to_tensor(batch_user, dtype=tf.int32)
        batch_pos_item = tf.convert_to_tensor(batch_pos_item, dtype=tf.int32)
        batch_neg_item = tf.convert_to_tensor(batch_neg_item, dtype=tf.int32)
        return batch_user, batch_pos_item, batch_neg_item

    def L2_loss(self, x):
        return tf.reduce_mean(tf.reduce_sum(tf.pow(x, 2), axis=1, keepdims=False) / 2.)

    def call(self, input_u, ua_embeddings, ia_embeddings):
        uid = tf.gather(ua_embeddings, input_u)
        uid = tf.reshape(uid, (-1, self.n_criterion, self.emb_dim))
        bpr_loss = 0
        for i in range(self.n_criterion):
            batch_user, batch_pos_item, bath_neg_item = self.generate_cf_batch(
                self.users_dict_list[i], len(input_u))
            user_embed = tf.gather(ua_embeddings, batch_user)
            user_embed = tf.reshape(user_embed, (-1, self.n_criterion, self.emb_dim))[:, i, :]
            item_pos_embed = tf.gather(ia_embeddings[:, i, :], batch_pos_item)
            item_neg_embed = tf.gather(ia_embeddings[:, i, :], bath_neg_item)

            if i == (self.n_criterion - 1):
                g2_pos = tf.sigmoid(tf.reduce_sum(user_embed * item_pos_embed, axis=1))
                g2_neg = tf.sigmoid(tf.reduce_sum(user_embed * item_neg_embed, axis=1))

                K = self.n_criterion - 1
                g1_pos = 0.0
                g1_neg = 0.0

                for k in range(K):
                    user_embed_k = tf.gather(ua_embeddings, batch_user)[:, k, :]
                    user_embed_k_ = tf.math.l2_normalize(user_embed_k, axis=1)

                    item_pos_embed_k = tf.gather(ia_embeddings[:, k, :], batch_pos_item)
                    item_neg_embed_k = tf.gather(ia_embeddings[:, k, :], bath_neg_item)

                    sim_user = tf.sigmoid(tf.reduce_sum(user_embed * user_embed_k, axis=1))
                    m_pos_k = tf.sigmoid(tf.reduce_sum(user_embed_k * item_pos_embed_k, axis=1))
                    m_neg_k = tf.sigmoid(tf.reduce_sum(user_embed_k * item_neg_embed_k, axis=1))

                    g1_pos += sim_user * m_pos_k
                    g1_neg += sim_user * m_neg_k

                g1_pos /= K
                g1_neg /= K

                pos_score = g1_pos * g2_pos
                neg_score = g1_neg * g2_neg
            else:
                pos_score = tf.sigmoid(tf.reduce_sum(user_embed * item_pos_embed, axis=1))
                neg_score = tf.sigmoid(tf.reduce_sum(user_embed * item_neg_embed, axis=1))

            cf_loss = (-1.0) * tf.math.log_sigmoid(pos_score - neg_score)
            cf_loss = tf.reduce_mean(cf_loss)

            l2_loss = self.L2_loss(user_embed) + self.L2_loss(item_pos_embed) + self.L2_loss(item_neg_embed)
            if i != self.n_criterion - 1:
                cf_loss = cf_loss * 1e-2
            bpr_loss = bpr_loss + cf_loss + l2_loss

        bpr_loss = bpr_loss / 5
        return bpr_loss


class RecLoss(tf.keras.layers.Layer):
    def __init__(self, data_config, args):
        super(RecLoss, self).__init__()
        self.cris = data_config['cris']
        self.n_criterion = len(self.cris)
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.emb_dim = args.embed_size

        self.coefficient = eval(args.coefficient)
        self.wid = eval(args.wid)

    def call(self, input_u, label_phs, ua_embeddings, ia_embeddings, cri_embeddings):
        uid = tf.gather(ua_embeddings, input_u)
        uid = tf.reshape(uid, (-1, self.n_criterion, self.emb_dim))
        pos_r_list = []
        for i in range(self.n_criterion):
            beh = self.cris[i]
            pos_beh = tf.gather(ia_embeddings[:, i, :], label_phs[i])
            pos_num_beh = tf.cast(tf.not_equal(label_phs[i], self.n_items), tf.float32)
            pos_beh = tf.einsum('ab,abc->abc', pos_num_beh, pos_beh)
            pos_r = tf.einsum('ac,abc->abc', uid[:, i, :], pos_beh)
            pos_r = tf.einsum('ajk,lk->aj', pos_r, cri_embeddings[beh])
            pos_r_list.append(pos_r)

        loss = 0.
        for i in range(self.n_criterion):
            beh = self.cris[i]
            temp = tf.einsum('ab,ac->bc', ia_embeddings[:, i, :], ia_embeddings[:, i, :]) \
                   * tf.einsum('ab,ac->bc', uid[:, i, :], uid[:, i, :])
            tmp_loss = self.wid[i] * tf.reduce_sum(
                temp * tf.matmul(tf.transpose(cri_embeddings[beh]), cri_embeddings[beh]))
            tmp_loss += tf.reduce_sum((1.0 - self.wid[i]) * tf.square(pos_r_list[i]) - 2.0 * pos_r_list[i])

            loss += self.coefficient[i] * tmp_loss

        regularizer = tf.reduce_sum(tf.square(uid)) * 0.5 + tf.reduce_sum(tf.square(ia_embeddings)) * 0.5
        emb_loss = args.decay * regularizer

        return loss, emb_loss


def get_lables(temp_set, k=0.9999):
    max_item = 0
    item_lenth = []
    for i in temp_set:
        item_lenth.append(len(temp_set[i]))
        if len(temp_set[i]) > max_item:
            max_item = len(temp_set[i])
    item_lenth.sort()

    max_item = item_lenth[int(len(item_lenth) * k) - 1]

    print(max_item)
    for i in temp_set:
        if len(temp_set[i]) > max_item:
            temp_set[i] = temp_set[i][0:max_item]
        while len(temp_set[i]) < max_item:
            temp_set[i].append(n_items)
    return max_item, temp_set


def get_train_instances1(max_item_list, beh_label_list):
    user_train = []
    beh_item_list = [list() for i in range(n_cris)]

    for i in beh_label_list[-1].keys():
        user_train.append(i)
        beh_item_list[-1].append(beh_label_list[-1][i])
        for j in range(n_cris - 1):
            if i not in beh_label_list[j].keys():
                beh_item_list[j].append([n_items] * max_item_list[j])
            else:
                beh_item_list[j].append(beh_label_list[j][i])

    user_train = np.array(user_train)
    beh_item_list = [np.array(beh_item) for beh_item in beh_item_list]
    user_train = user_train[:, np.newaxis]
    return user_train, beh_item_list


def get_train_pairs(user_train_batch, beh_item_tgt_batch):
    input_u_list, input_i_list = [], []
    for i in range(len(user_train_batch)):
        pos_items = beh_item_tgt_batch[i][np.where(beh_item_tgt_batch[i] != n_items)]
        uid = user_train_batch[i][0]
        input_u_list += [uid] * len(pos_items)
        input_i_list += pos_items.tolist()

    return np.array(input_u_list).reshape([-1]), np.array(input_i_list).reshape([-1])

def test_DMCR(ua_embeddings, ia_embeddings, cri_embedding, users_to_test, cris, batch_test_flag=False):
    def get_score_np(ua_embeddings, ia_embeddings, cri_embedding, users, items):
        n_criterion = len(cris)
        batch_ratings = 0.0

        for i in range(n_criterion):
            ug_embeddings = ua_embeddings[users, i, :]
            pos_ig_embeddings = ia_embeddings[items, i, :]
            rel_embed = cri_embedding[cris[i]].numpy()

            dot = pos_ig_embeddings * rel_embed
            ratings_i = np.matmul(ug_embeddings, dot.T)

            user_norms = np.linalg.norm(ug_embeddings, axis=1, keepdims=True)
            sigma_norm_u = 1 / (1 + np.exp(-user_norms))

            user_item_dot = np.matmul(ug_embeddings, pos_ig_embeddings.T)
            sigma_dot = 1 / (1 + np.exp(-user_item_dot))

            g1_pos = sigma_norm_u * sigma_dot
            ratings_i *= g1_pos
            batch_ratings += ratings_i

        batch_ratings = batch_ratings / n_criterion
        return batch_ratings

    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)), 'auc': 0.}

    test_users = users_to_test
    n_test_users = len(test_users)

    pool = multiprocessing.Pool(cores)

    u_batch_size = BATCH_SIZE
    n_user_batchs = n_test_users // u_batch_size + 1

    count = 0
    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_batch = test_users[start: end]

        item_batch = range(ITEM_NUM)
        rate_batch = get_score_np(ua_embeddings, ia_embeddings, cri_embedding, user_batch, item_batch)

        user_batch_rating_uid = zip(rate_batch, user_batch)
        batch_result = pool.map(test_one_user, user_batch_rating_uid)
        count += len(batch_result)

        for re in batch_result:
            result['precision'] += re['precision'] / n_test_users
            result['recall'] += re['recall'] / n_test_users
            result['ndcg'] += re['ndcg'] / n_test_users
            result['hit_ratio'] += re['hit_ratio'] / n_test_users
            result['auc'] += re['auc'] / n_test_users
    assert count == n_test_users

    pool.close()
    return result


def preprocess_sim(args, config):
    topk1_user = args.topk1_user
    topk1_item = args.topk1_item

    input_u_sim = tf.constant(config['user_sim'])
    user_topk_values1 = tf.math.top_k(input_u_sim, k=min(topk1_user, config['n_users'])).values[:, -1:]
    user_indices_remove = input_u_sim > user_topk_values1

    input_i_sim = tf.constant(config['item_sim'])
    item_topk_values1 = tf.math.top_k(input_i_sim, k=min(topk1_item, config['n_items'])).values[:, -1:]
    item_indices_remove = input_i_sim > item_topk_values1
    item_indices_token = tf.constant([False] * item_indices_remove.shape[0], dtype=tf.bool)[:, tf.newaxis]
    item_indices_remove = tf.concat([item_indices_remove, item_indices_token], axis=1)

    return user_indices_remove, item_indices_remove


def set_seed(seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


if __name__ == '__main__':

    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print('-----------------------')
    print(len(physical_devices))
    print('-----------------------')

    set_seed(30)

    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items
    config['cris'] = data_generator.cris
    config['trn_mat'] = data_generator.trnMats[-1]
    config['users_dict'] = data_generator.trnDicts

    pre_adj_list = data_generator.get_adj_mat()
    config['pre_adjs'] = pre_adj_list
    n_users, n_items = data_generator.n_users, data_generator.n_items
    cris = data_generator.cris
    n_cris = data_generator.beh_num

    trnDicts = copy.deepcopy(data_generator.trnDicts)
    max_item_list = []
    beh_label_list = []
    for i in range(n_cris):
        max_item, beh_label = get_lables(trnDicts[i])
        max_item_list.append(max_item)
        beh_label_list.append(beh_label)

    t0 = time.time()
    config['max_item_list'] = max_item_list

    model = DMCR(max_item_list, data_config=config, args=args)
    recloss = RecLoss(data_config=config, args=args)
    bprloss = BprLoss(data_config=config, args=args)

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
        args.lr, decay_steps=args.lr_decay_step, decay_rate=args.lr_gamma, staircase=True)
    
    cur_best_pre_0 = 0.

    run_time = 1

    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []

    stopping_step = 0
    should_stop = False

    user_train1, beh_item_list = get_train_instances1(max_item_list, beh_label_list)

    nonshared_idx = -1

    for epoch in range(args.epoch):
        shuffle_indices = np.random.permutation(np.arange(len(user_train1)))
        user_train1 = user_train1[shuffle_indices]
        beh_item_list = [beh_item[shuffle_indices] for beh_item in beh_item_list]

        t1 = time.time()
        loss, rec_loss, emb_loss, bpr_loss = 0., 0., 0., 0.
        n_batch = int(len(user_train1) / args.batch_size)

        for idx in range(n_batch):
            start_index = idx * args.batch_size
            end_index = min((idx + 1) * args.batch_size, len(user_train1))

            u_batch = user_train1[start_index:end_index]
            beh_batch = [beh_item[start_index:end_index] for beh_item in beh_item_list]

            u_batch_list, i_batch_list = get_train_pairs(
                user_train_batch=u_batch, beh_item_tgt_batch=beh_batch[-1])

            u_batch = tf.convert_to_tensor(u_batch, dtype=tf.int32)
            beh_batch = [tf.convert_to_tensor(beh_item, dtype=tf.int32) for beh_item in beh_batch]
            u_batch_list = tf.convert_to_tensor(u_batch_list, dtype=tf.int32)
            i_batch_list = tf.convert_to_tensor(i_batch_list, dtype=tf.int32)

            with tf.GradientTape() as tape:
                ua_embeddings, ia_embeddings, cri_embeddings = model(None, training=True)
                
                batch_rec_loss, batch_emb_loss = recloss(
                    u_batch, beh_batch, ua_embeddings, ia_embeddings, cri_embeddings)
                
                batch_bpr_loss = bprloss(u_batch_list, ua_embeddings, ia_embeddings)
                
                batch_loss = batch_rec_loss + batch_emb_loss + batch_bpr_loss

            gradients = tape.gradient(batch_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            loss += batch_loss.numpy() / n_batch
            rec_loss += batch_rec_loss.numpy() / n_batch
            emb_loss += batch_emb_loss.numpy() / n_batch
            bpr_loss += batch_bpr_loss.numpy() / n_batch

        if np.isnan(loss):
            print('ERROR: loss is nan.')
            sys.exit()

        if (epoch + 1) % args.test_epoch != 0:
            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f + %.5f + %.5f]' % (
                    epoch, time.time() - t1, loss, rec_loss, emb_loss)
                print(perf_str)
            continue

        t2 = time.time()
        ua_embeddings, ia_embeddings, cri_embeddings = model(None, training=False)
        
        users_to_test = list(data_generator.test_set.keys())
        
        ret = test_DMCR(ua_embeddings.numpy(), ia_embeddings.numpy(),
                       cri_embeddings, users_to_test, cris)

        t3 = time.time()

        loss_loger.append(loss)
        rec_loger.append(ret['recall'])
        pre_loger.append(ret['precision'])
        ndcg_loger.append(ret['ndcg'])
        hit_loger.append(ret['hit_ratio'])

        if args.verbose > 0:
            perf_str = 'Epoch %d [%.1fs + %.1fs]:,  ' \
                      'precision=[%.5f, %.5f], recall=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                      (epoch, t2 - t1, t3 - t2,
                       ret['precision'][0], ret['precision'][1],
                       ret['recall'][0], ret['recall'][1],
                       ret['hit_ratio'][0], ret['hit_ratio'][1],
                       ret['ndcg'][0], ret['ndcg'][1])
            print(perf_str)

        cur_best_pre_0, stopping_step, should_stop, flag = early_stopping_new(
            ret['recall'][0], cur_best_pre_0, stopping_step, expected_order='acc', flag_step=10)

        if should_stop:
            break

    recs = np.array(rec_loger)
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)
    hit = np.array(hit_loger)

    best_rec_0 = max(recs[:, 0])
    idx = list(recs[:, 0]).index(best_rec_0)

    final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], ndcg=[%s]" % \
                (idx, time.time() - t0, '\t'.join(['%.4f' % r for r in recs[idx]]),
                 '\t'.join(['%.4f' % r for r in ndcgs[idx]]))
    print(final_perf)
