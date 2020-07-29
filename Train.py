import random
import os
import time
import argparse
import numpy as np
import tensorflow as tf
import MIFN as MIFN
import pickle as pk
from scipy import sparse
import gc

random.seed(331)
np.random.seed(331)


def load_batch(block_data, batch_size, pad_int, args):
    random.shuffle(block_data)
    for batch_i in range(0, len(block_data) // batch_size + 1):
        start_i = batch_i * batch_size
        batch = block_data[start_i:start_i + batch_size]
        yield get_session(batch, pad_int, args)


def get_session(batch, pad_int, args):
    seq_A, seq_B = [], []
    len_A, len_B = [], []
    len_all = []
    pos_A, pos_B = [], []
    target_A, target_B = [], []
    index1, index2 = [], []
    adj_1, adj_2, adj_3, adj_4, adj_5 = [], [], [], [], []
    neighbors = []
    nei_mask1, nei_mask2 = [], []
    nei_mask_L_A, nei_mask_L_B = [], []
    nei_index1, nei_index2 = [], []
    Isina, Isinb = [], []
    tarinA, tarinB = [], []
    for session in batch:
        len_A.append(session[4])
        len_B.append(session[5])
        len_all.append(len(session[8]))
    maxlen_A = max(len_A)
    maxlen_B = max(len_B)
    i = 0
    for session in batch:
        seq_A.append(session[0] + [pad_int] * (maxlen_A - len_A[i]))
        seq_B.append(session[1] + [pad_int] * (maxlen_B - len_B[i]))
        pos_A.append(session[2] + [pad_int] * (maxlen_A - len_A[i]))
        pos_B.append(session[3] + [pad_int] * (maxlen_B - len_B[i]))
        target_A.append(session[6])
        target_B.append(session[7])
        index1.append(session[9] + [pad_int] * (maxlen_A - len_A[i]))
        index2.append(session[10] + [pad_int] * (maxlen_B - len_B[i]))
        adj_1.append(session[11])
        adj_2.append(session[12])
        adj_3.append(session[13])
        adj_4.append(session[14])
        adj_5.append(session[15])

        neighbors.append(list(session[16].keys()))
        nei_index1.append(session[17])
        nei_index2.append(session[18])
        nei_mask1.append(session[19])
        nei_mask2.append(session[20])
        pad_len = maxlen_A + maxlen_B - len_all[i]
        t1 = np.concatenate((session[21], np.zeros((pad_len, args.nei_num))), axis=0)
        t2 = np.concatenate((session[22], np.zeros((pad_len, args.nei_num))), axis=0)
        nei_mask_L_A.append(t1)
        nei_mask_L_B.append(t2)
        Isina.append(session[23])
        Isinb.append(session[24])
        tarinA.append(session[25])
        tarinB.append(session[26])
        i += 1
    index = np.arange(len(batch))
    index = np.expand_dims(index, axis=-1)
    index_p = np.repeat(index, maxlen_A, axis=1)
    pos_A = np.stack([index_p, np.array(pos_A)], axis=-1)
    index1 = np.stack([index_p, np.array(index1)], axis=-1)
    index_p = np.repeat(index, maxlen_B, axis=1)
    pos_B = np.stack([index_p, np.array(pos_B)], axis=-1)
    index2 = np.stack([index_p, np.array(index2)], axis=-1)

    index_p = np.repeat(index, args.nei_num, axis=1)
    nei_index1 = np.stack([index_p, np.array(nei_index1)], axis=-1)
    nei_index2 = np.stack([index_p, np.array(nei_index2)], axis=-1)

    return np.array(adj_1, dtype=int), np.array(adj_2, dtype=int), np.array(adj_3, dtype=int), \
           np.array(adj_4, dtype=int), np.array(adj_5, dtype=int), \
           np.array(seq_A), np.array(seq_B), pos_A, pos_B, index1, index2, \
           np.array(len_A), np.array(len_B), np.array(target_A), np.array(target_B), \
           np.array(neighbors), np.array(nei_index1), np.array(nei_index2), \
           np.array(Isina), np.array(Isinb), \
           np.array(nei_mask_L_A), np.array(nei_mask_L_B), \
           np.array(nei_mask1), np.array(nei_mask2), np.array(tarinA), np.array(tarinB)


def feed_dict(model, data, isTrain):
    adj1, adj2, adj3, adj4, adj5 = data[0], data[1], data[2], data[3], data[4],
    seq_A, seq_B, pos_A, pos_B, index_A, index_B = data[5], data[6], data[7], data[8], data[9], data[10],
    len_A, len_B, target_A, target_B, neighbors = data[11], data[12], data[13], data[14], data[15],
    nei_index_A, nei_index_B, = data[16], data[17],
    Isina, Isinb = data[18], data[19],
    nei_mask_L_a, nei_mask_L_b = data[20], data[21],
    nei_maska, nei_maskb = data[22], data[23],
    tarinA, tarinB = data[24], data[25],
    if isTrain:
        feed_dict = {model.seq_A: seq_A, model.seq_B: seq_B, model.pos_A: pos_A, model.pos_B: pos_B,
                     model.len_A: len_A, model.len_B: len_B, model.target_A: target_A, model.target_B: target_B,
                     model.tar_in_A: tarinA, model.tar_in_B: tarinB,
                     model.adj_1: adj1, model.adj_2: adj2, model.adj_3: adj3, model.adj_4: adj4, model.adj_5: adj5,
                     model.neighbors: neighbors,
                     model.index_A: index_A, model.index_B: index_B,
                     model.nei_index_A: nei_index_A, model.nei_index_B: nei_index_B,
                     model.nei_A_mask: nei_maska, model.nei_B_mask: nei_maskb,
                     model.nei_L_A_mask: nei_mask_L_a, model.nei_L_T_mask: nei_mask_L_b,
                     model.IsinnumA: Isina, model.IsinnumB: Isinb, }
    else:
        feed_dict = {model.seq_A: seq_A, model.seq_B: seq_B, model.pos_A: pos_A, model.pos_B: pos_B,
                     model.len_A: len_A, model.len_B: len_B,
                     model.adj_1: adj1, model.adj_2: adj2, model.adj_3: adj3, model.adj_4: adj4, model.adj_5: adj5,
                     model.neighbors: neighbors,
                     model.index_A: index_A, model.index_B: index_B,
                     model.nei_index_A: nei_index_A, model.nei_index_B: nei_index_B,
                     model.nei_A_mask: nei_maska, model.nei_B_mask: nei_maskb,
                     model.nei_L_A_mask: nei_mask_L_a, model.nei_L_T_mask: nei_mask_L_b,
                     model.IsinnumA: Isina, model.IsinnumB: Isinb, }
    return feed_dict


def train(args):
    model = MIFN.AINet_all(num_items_A=29207, num_items_B=34886,
                            num_entity_A=50273, num_entity_B=82552, num_cate=32,
                            neighbor_num=args.nei_num, batch_size=args.batch_size, gpu=args.gpu,
                            hidden_size=args.hidden_size, embedding_size=args.embedding_size,
                            lr=args.lr, keep_prob=args.keep_prob)
    print(time.localtime())
    checkpoint = ''
    with tf.Session(graph=model.graph, config=model.config) as sess:
        writer = tf.summary.FileWriter('', sess.graph)
        saver = tf.train.Saver(max_to_keep=args.epochs)
        sess.run(tf.global_variables_initializer())
        for epoch in range(args.epochs):
            loss = 0
            step = 0
            filelist = os.listdir(args.train_path)
            for file in filelist:
                with open(args.train_path + file, 'rb') as f:
                    block_data = pk.load(f)
                for k, (epoch_data) in enumerate(load_batch(block_data, args.batch_size, args.pad_int, args)):
                    _, l = sess.run([model.train_op, model.loss], feed_dict(model, epoch_data, isTrain=True))
                    loss += l
                    step += 1
                gc.collect()
            print('Epoch {}/{} - Training Loss: {:.3f}'.format(epoch + 1, args.epochs, loss / step))
            saver.save(sess, checkpoint, global_step=epoch + 1)
            evaluation(model, sess, args.valid_path, epoch, loss, step)


def evaluation(model, sess, validpath, epoch, loss, step):
    print(time.localtime())
    validlen = 7650
    r5_a, r10_a, r20_a = 0, 0, 0
    m5_a, m10_a, m20_a = 0, 0, 0
    r5_b, r20_b, r10_b = 0, 0, 0
    m5_b, m10_b, m20_b = 0, 0, 0

    filelist = os.listdir(validpath)
    for file in filelist:
        with open(validpath + file, 'rb') as f:
            block_data = pk.load(f)
        for _, (epoch_data) in enumerate(load_batch(block_data, args.batch_size, args.pad_int, args)):
            pa, pb = sess.run([model.pred_A, model.pred_B], feed_dict(model, epoch_data, isTrain=False))
            target_A, target_B = epoch_data[13], epoch_data[14]
            recall, mrr = get_eval(pa, target_A, [5, 10, 20])
            r5_a += recall[0]
            m5_a += mrr[0]
            r10_a += recall[1]
            m10_a += mrr[1]
            r20_a += recall[2]
            m20_a += mrr[2]
            recall, mrr = get_eval(pb, target_B, [5, 10, 20])
            r5_b += recall[0]
            m5_b += mrr[0]
            r10_b += recall[1]
            m10_b += mrr[1]
            r20_b += recall[2]
            m20_b += mrr[2]
        gc.collect()
    print('Recall5: {:.5f}; Mrr5: {:.5f}'.format(r5_a / validlen, m5_a / validlen))
    print('Recall10: {:.5f}; Mrr10: {:.5f}'.format(r10_a / validlen, m10_a / validlen))
    print('Recall20: {:.5f}; Mrr20: {:.5f}'.format(r20_a / validlen, m20_a / validlen))
    print('Recall5: {:.5f}; Mrr5: {:.5f}'.format(r5_b / validlen, m5_b / validlen))
    print('Recall10: {:.5f}; Mrr10: {:.5f}'.format(r10_b / validlen, m10_b / validlen))
    print('Recall20: {:.5f}; Mrr20: {:.5f}'.format(r20_b / validlen, m20_b / validlen))
    print(time.localtime())

    with open('', 'a+') as f:
        f.write('epoch: ' + str(epoch + 1) + '\t' + str(loss / step) + '\n')
        f.write('recall-A @5|10|20: ' + str(r5_a / validlen) + '\t' + str(r10_a / validlen) + '\t' + str(
            r20_a / validlen) + '\t')
        f.write('mrr-A @5|10|20: ' + str(m5_a / validlen) + '\t' + str(m10_a / validlen) + '\t' + str(
            m20_a / validlen) + '\n')
        f.write('recall-B @5|10|20: ' + str(r5_b / validlen) + '\t' + str(r10_b / validlen) + '\t' + str(
            r20_b / validlen) + '\t')
        f.write('mrr-B @5|10|20: ' + str(m5_b / validlen) + '\t' + str(m10_b / validlen) + '\t' + str(
            m20_b / validlen) + '\n')


def get_eval(predlist, truelist, klist):  # return recall@k and mrr@k
    recall = []
    mrr = []
    predlist = predlist.argsort()
    for k in klist:
        recall.append(0)
        mrr.append(0)
        templist = predlist[:, -k:]  # the result of argsort is in ascending
        i = 0
        while i < len(truelist):
            pos = np.argwhere(templist[i] == truelist[i])  # pos is a list of positions whose values are all truelist[i]
            if len(pos) > 0:
                recall[-1] += 1
                mrr[-1] += 1 / (k - pos[0][0])
            else:
                recall[-1] += 0
                mrr[-1] += 0
            i += 1
    return recall, mrr  # they are sum instead of mean

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--epochs', type=int, default=100, help='the number of epochs')
    parser.add_argument('--keep_prob', type=float, default=0.8, help='keep prob of hidden unit')
    parser.add_argument('--pad_int', type=int, default=0,
                        help='padding on the session')
    parser.add_argument('--train_path', type=str, default='',
                        help='train data path')
    parser.add_argument('--valid_path', type=str, default='',
                        help='valid data path')
    parser.add_argument('--test_path', type=str, default='',
                        help='test data path')
    parser.add_argument('--n_iter', type=int, default=1, help='the number of h hop')
    parser.add_argument('--embedding_size', type=int, default=256, help='embedding size of vector')
    parser.add_argument('--hidden_size', type=int, default=256, help='hidden size')
    parser.add_argument('--num_layers', type=int, default=1, help='num_layers')
    parser.add_argument('--nei_num', type=int, default=200, help='num_neighbours')
    parser.add_argument('--gpu', type=str, default='3', help='use of gpu')
    args = parser.parse_args()

    train(args)