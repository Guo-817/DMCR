import metrics as metrics
from parser import parse_args
from load_data import *
import multiprocessing
import heapq
import numpy as np

cores = multiprocessing.cpu_count() // 2

args = parse_args()
Ks = eval(args.Ks)

data_generator = DataHandler(dataset=args.dataset, batch_size=args.batch_size)
USR_NUM, ITEM_NUM = data_generator.n_users, data_generator.n_items
N_TRAIN, N_TEST = data_generator.n_train, data_generator.n_test
BATCH_SIZE = args.batch_size // 4 if args.dataset == 'RB-Extended' else args.batch_size // 2


def ranklist_by_heapq(user_pos_test, test_items, rating, Ks):
    item_score = {i: rating[i] for i in test_items}
    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = [1 if i in user_pos_test else 0 for i in K_max_item_score]
    auc = 0.
    return r, auc


def get_auc(item_score, user_pos_test):
    item_score = sorted(item_score.items(), key=lambda kv: kv[1], reverse=True)
    item_sort = [x[0] for x in item_score]
    posterior = [x[1] for x in item_score]
    r = [1 if i in user_pos_test else 0 for i in item_sort]
    auc = metrics.auc(ground_truth=r, prediction=posterior)
    return auc


def ranklist_by_sorted(user_pos_test, test_items, rating, Ks):
    item_score = {i: rating[i] for i in test_items}
    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = [1 if i in user_pos_test else 0 for i in K_max_item_score]
    auc = get_auc(item_score, user_pos_test)
    return r, auc, item_score


def get_performance(user_pos_test, r, auc, Ks):
    precision, recall, ndcg, hit_ratio = [], [], [], []

    for K in Ks:
        precision.append(metrics.precision_at_k(r, K))
        recall.append(metrics.recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(metrics.ndcg_at_k(r, K))
        hit_ratio.append(metrics.hit_at_k(r, K))

    map_score = metrics.mean_average_precision([r])
    mrr_score = metrics.mean_reciprocal_rank([r])

    return {
        'recall': np.array(recall),
        'precision': np.array(precision),
        'ndcg': np.array(ndcg),
        'hit_ratio': np.array(hit_ratio),
    }

def test_one_user(x):
    rating = x[0]
    u = x[1]

    try:
        training_items = data_generator.train_items[u]
    except Exception:
        training_items = []

    user_pos_test = data_generator.test_set[u]
    all_items = set(range(ITEM_NUM))
    test_items = list(all_items - set(training_items))

    if args.test_flag == 'part':
        r, auc = ranklist_by_heapq(user_pos_test, test_items, rating, Ks)
        item_score_dict = {i: rating[i] for i in test_items}
    else:
        r, auc, item_score_dict = ranklist_by_sorted(user_pos_test, test_items, rating, Ks)

    perf = get_performance(user_pos_test, r, auc, Ks)

    real_scores = [1.0 for _ in user_pos_test]
    pred_scores = [item_score_dict[i] for i in user_pos_test if i in item_score_dict]

    perf['real_scores'] = real_scores
    perf['pred_scores'] = pred_scores

    return perf


def test_one_user_train(x):
    rating = x[0]
    u = x[1]
    training_items = []
    user_pos_test = data_generator.train_items[u]
    all_items = set(range(ITEM_NUM))
    test_items = list(all_items - set(training_items))

    if args.test_flag == 'part':
        r, auc = ranklist_by_heapq(user_pos_test, test_items, rating, Ks)
        item_score_dict = {i: rating[i] for i in test_items}
    else:
        r, auc, item_score_dict = ranklist_by_sorted(user_pos_test, test_items, rating, Ks)

    perf = get_performance(user_pos_test, r, auc, Ks)
    real_scores = [1.0 for _ in user_pos_test]
    pred_scores = [item_score_dict[i] for i in user_pos_test if i in item_score_dict]
    perf['real_scores'] = real_scores
    perf['pred_scores'] = pred_scores

    return perf


def test(sess, model, users_to_test, drop_flag=False, batch_test_flag=False, train_set_flag=0):
    result = {
        'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)),
        'ndcg': np.zeros(len(Ks)), 'hit_ratio': np.zeros(len(Ks))
    }

    pool = multiprocessing.Pool(cores)

    u_batch_size = BATCH_SIZE
    i_batch_size = BATCH_SIZE

    test_users = users_to_test
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1

    all_real, all_pred = [], []

    count = 0

    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size
        user_batch = test_users[start: end]

        if batch_test_flag:
            rate_batch = np.zeros((len(user_batch), ITEM_NUM))
            for i_batch_id in range(ITEM_NUM // i_batch_size + 1):
                i_start = i_batch_id * i_batch_size
                i_end = min((i_batch_id + 1) * i_batch_size, ITEM_NUM)
                item_batch = range(i_start, i_end)
                feed_dict = {model.users: user_batch, model.pos_items: item_batch}
                if drop_flag:
                    feed_dict.update({
                        model.node_dropout: [0.] * len(eval(args.layer_size)),
                        model.mess_dropout: [0.] * len(eval(args.layer_size))
                    })
                i_rate_batch = sess.run(model.batch_ratings, feed_dict)
                rate_batch[:, i_start:i_end] = i_rate_batch
        else:
            item_batch = range(ITEM_NUM)
            feed_dict = {model.users: user_batch, model.pos_items: item_batch}
            if drop_flag:
                feed_dict.update({
                    model.node_dropout: [0.] * len(eval(args.layer_size)),
                    model.mess_dropout: [0.] * len(eval(args.layer_size))
                })
            rate_batch = sess.run(model.batch_ratings, feed_dict)

        user_batch_rating_uid = zip(rate_batch, user_batch)
        if train_set_flag == 0:
            batch_result = pool.map(test_one_user, user_batch_rating_uid)
        else:
            batch_result = pool.map(test_one_user_train, user_batch_rating_uid)
        count += len(batch_result)

        for re in batch_result:
            result['precision'] += re['precision'] / n_test_users
            result['recall'] += re['recall'] / n_test_users
            result['ndcg'] += re['ndcg'] / n_test_users
            result['hit_ratio'] += re['hit_ratio'] / n_test_users
            all_real.extend(re['real_scores'])
            all_pred.extend(re['pred_scores'])

    assert count == n_test_users
    pool.close()
    return result
