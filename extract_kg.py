import os
import random
import numpy as np
import pickle as pk
from scipy import sparse
import collections
import linecache

def read_all_responding_index():

    with open(ent2ind_a_p, 'r') as f:
        for line in f.readlines():
            line = line.strip().split('\t')
            entity_id2index_A[int(line[0])] = int(line[1]) #[oldid, index]
    with open(ent2ind_b_p, 'r') as f:
        for line in f.readlines():
            line = line.strip().split('\t')
            entity_id2index_B[int(line[0])] = int(line[1])

    with open(item2ind_a_p, 'r') as f:
        for line in f.readlines():
            line = line.strip().split('\t')
            item_index_old2new_A[int(line[0])] = int(line[1])
    with open(item2ind_b_p, 'r') as f:
        for line in f.readlines():
            line = line.strip().split('\t')
            item_index_old2new_B[int(line[0])] = int(line[1])
def getdata(datapath):
    # read sequnece data，each line represents a sequence
    with open(datapath, 'r') as f:
        sessions = []
        for line in f.readlines():
            session = []
            line = line.strip().split('\t')
            for item in line[1:]:
                item = int(item)
                session.append(item)
            sessions.append(session)
    return sessions
def gen_mask(index_dict,sequence):
    # generate nei_mask
    nei = list(index_dict.keys())
    ini_A = np.ones(max_Nei)
    ini_A[np.where(np.array(nei) > entity_border)] = 0
    ini_B = np.ones(max_Nei)
    ini_B[np.where(np.array(nei) <= entity_border)] = 0
    temp_a, temp_t = [], []
    for item in sequence:
        if item <= entity_border:
            temp_a.append(ini_A)
            temp_t.append(ini_B)
        else:
            temp_a.append(ini_B)
            temp_t.append(ini_A)
    return ini_A, ini_B, temp_a,temp_t
def gen_index(index_dict):
    itemB_border = 85159
    itemAset = set(range(item_border+1))
    itemBset = set(range(entity_border+1, itemB_border))
    IsinnumA,IsinnumB = [],[]
    for ii in index_dict.keys():
        if ii in itemAset:
            IsinnumA.append(1)
            IsinnumB.append(0)
        elif ii in itemBset:
            IsinnumA.append(0)
            IsinnumB.append(1)
        else:
            IsinnumA.append(0)
            IsinnumB.append(0)
    nei_index_A = np.zeros(max_Nei)
    ind1 = np.where(np.array(list(index_dict.keys())) <= item_border)[0]
    ent_id = np.array(list(index_dict.keys()))[ind1]
    ite_id = []
    for ii in ent_id:
        ite_id.append(ii)
    nei_index_A[ind1] = ite_id

    nei_index_B = np.zeros(max_Nei)
    ind1 = np.where((np.array(list(index_dict.keys())) > entity_border) &
                    (np.array(list(index_dict.keys())) <= itemB_border))[0]
    ent_id = np.array(list(index_dict.keys()))[ind1]
    ite_id = []
    for ii in ent_id:
        ite_id.append(ii-entity_border)
    nei_index_B[ind1] = ite_id

    return nei_index_A, nei_index_B, IsinnumA, IsinnumB

def processdata(dataset,filepath):
    sessions = []
    all_count = 0
    for session in dataset:  # 1->A，2->B
        all_count += 1
        temp = []
        seq1,seq2 = [],[]
        pos1,pos2 = [],[]
        index1,index2 = [],[]
        len1,len2 = 0,0
        ind = 0
        sequence,allseq = [],[]
        tar_in_A, tar_in_B = [], []
        for item in session[:-2]:  # get all the items except targetA and targetB
            if item <= item_border: # domain A
                item = item_index_old2new_A[item]
                seq1.append(item)
                pos1.append(len2)
                len1 += 1
                index1.append(ind)
            else: # domain B
                item = item_index_old2new_B[item-item_border-1] + len(entity_id2index_A)
                seq2.append(item)
                pos2.append(len1)
                len2 += 1
                index2.append(ind)
            sequence.append(item)
            allseq.append(item)
            ind += 1
        tarA = item_index_old2new_A[session[-2]]
        tarB = item_index_old2new_B[session[-1]-item_border-1]

        adj_1, adj_2, adj_3, adj_4, adj_5, index_dict = get_subgraph3(allseq)

        if tarA in index_dict.keys():
            tar_in_A.append(1)
        else:
            tar_in_A.append(0)
        if tarB in index_dict.keys():
            tar_in_B.append(1)
        else:
            tar_in_B.append(0)
        nei_mask_A, nei_mask_B,nei_mask_L_A,nei_mask_L_T = gen_mask(index_dict,sequence)
        nei_indexA, nei_indexB,IsinnumA, IsinnumB = gen_index(index_dict)
        temp.append(seq1)
        temp.append(seq2)
        temp.append(pos1)
        temp.append(pos2)
        temp.append(len1)
        temp.append(len2)
        temp.append(tarA)
        temp.append(tarB)
        temp.append(seq1 + seq2)
        temp.append(index1)
        temp.append(index2)
        temp.append(adj_1.toarray())
        temp.append(adj_2.toarray())
        temp.append(adj_3.toarray())
        temp.append(adj_4.toarray())
        temp.append(adj_5.toarray())
        temp.append(index_dict)
        temp.append(nei_indexA)
        temp.append(nei_indexB)
        temp.append(nei_mask_A)
        temp.append(nei_mask_B)
        temp.append(nei_mask_L_A)
        temp.append(nei_mask_L_T)
        temp.append(IsinnumA)
        temp.append(IsinnumB)
        temp.append(tar_in_A)
        temp.append(tar_in_B)
        pk.dump(temp, open(filepath + str(all_count) + '.pkl', 'wb+'), -1)
        sessions.append(temp)

    return sessions

def initial_adj():
    adj_1 = sparse.dok_matrix((max_Nei, max_Nei), dtype=np.float32)
    adj_2 = sparse.dok_matrix((max_Nei, max_Nei), dtype=np.float32)
    adj_3 = sparse.dok_matrix((max_Nei, max_Nei), dtype=np.float32)
    adj_4 = sparse.dok_matrix((max_Nei, max_Nei), dtype=np.float32)
    adj_5 = sparse.dok_matrix((max_Nei, max_Nei), dtype=np.float32)
    for i in range(max_Nei):
        adj_1[i, i] = 1
        adj_2[i, i] = 1
        adj_3[i, i] = 1
        adj_4[i, i] = 1
        adj_5[i, i] = 1
    return adj_1,adj_2,adj_3,adj_4,adj_5,

def get_subgraph3(session):

    node = np.unique(session)
    node_a = []
    node_b = []
    for i in node:
        if i <= entity_border:
            node_a.append(i)
        else:
            node_b.append(i)

    adj_1, adj_2, adj_3, adj_4, adj_5, = initial_adj()
    a_set = set(node_a)
    b_set = set(node_b)
    Nei_a_dic, Nei_b_dic = {},{}
    nei_dic_a, nei_dic_b = {},{}
    bNei_a_dic, bNei_b_dic = {},{}
    temp_node_a = node_a
    temp_node_b = node_b
    for hop in range(max_H):
        Nei_a_dic[hop],Nei_b_dic[hop] = {},{}
        nei_dic_a[hop],nei_dic_b[hop] = set(), set()
        bNei_a_dic[hop], bNei_b_dic[hop] = {}, {}
        nei_set_a, nei_set_b = extract_h_graph(temp_node_a,temp_node_b,hop,
                                               nei_dic_a,nei_dic_b,
                                               Nei_a_dic,Nei_b_dic,
                                               bNei_a_dic, bNei_b_dic) # within 1-hop neighbor
        temp_node_a = nei_set_a - a_set - b_set
        temp_node_b = nei_set_b - b_set - a_set
        connected,X,Y = IsConnect3(nei_set_a, nei_set_b, node_a, node_b)
        a_set.update(nei_set_a)
        b_set.update(nei_set_b)
        if connected or hop == max_H-1:
            new_node_a, new_node_b, \
            new_dict_a, new_dict_b = pruning_nei(node_a,node_b,a_set,b_set,
                                                 nei_dic_a, nei_dic_b,Nei_a_dic, Nei_b_dic,
                                                bNei_a_dic, bNei_b_dic, hop, X,Y)
            index_dict = build_adj(session, new_node_a, new_node_b, new_dict_a, new_dict_b,
                                   adj_1, adj_2, adj_3, adj_4, adj_5,)
            return adj_1, adj_2, adj_3, adj_4, adj_5, index_dict

def build_adj(session,a_set,b_set, new_dict_a, new_dict_b,
              adj_1, adj_2, adj_3, adj_4, adj_5,):

    a_set = a_set.union(b_set)
    if len(a_set) < max_Nei:
        pool = set(kg.keys()).difference(a_set)
        rest = list(random.sample(list(pool), max_Nei - len(a_set)))
        a_set.update(rest)
        index_dict = dict(zip(list(a_set)[:max_Nei], range(max_Nei)))
    else:
        index_dict = dict(zip(list(a_set)[:max_Nei], range(max_Nei)))
    all_dic = {}
    all_dic.update(new_dict_a)
    all_dic.update(new_dict_b)
    for k in all_dic.keys():
        if k not in index_dict.keys():
            break
        else:
            u = index_dict[k]
        for it in all_dic[k]:
            if it[0] not in index_dict.keys():
                break
            else:
                v = index_dict[it[0]]
                if it[1] == 0:  # also-buy
                    adj_1[u, v] = 1
                elif it[1] == 1:  # also-view
                    adj_2[u, v] = 1
                elif it[1] == 2:  # buy-after-view
                    adj_3[u, v] = 1
                elif it[1] == 3:  # buy-together
                    adj_4[u, v] = 1
                elif it[1] == 4:  # category
                    adj_5[u, v] = 1

    for ind in range(len(session)):
        item = session[ind]
        if item not in index_dict.keys():
            continue
        else:
            u = index_dict[item]
            if ind < len(session)-1:
                next_item = session[ind+1]
                v = index_dict[next_item]
                adj_1[u, v] = 1
                adj_2[u, v] = 1
                adj_3[u, v] = 1
                adj_4[u, v] = 1
                adj_5[u, v] = 1
            else:
                break

    return index_dict

def extract_h_graph(temp_node_a, temp_node_b, hop, nei_dic_a, nei_dic_b, Nei_a_dic, Nei_b_dic,
                    bNei_a_dic, bNei_b_dic):
    a_set = set(temp_node_a)
    b_set = set(temp_node_b)
    orig_set = a_set.union(b_set)
    all_count = len(orig_set)

    for item in a_set:
        if item in kg.keys():
            neighbors = kg[item]
        else:
            neighbors = []
        if item not in Nei_a_dic[hop].keys() and neighbors:
            Nei_a_dic[hop][item] = []
            Nei_a_dic[hop][item] += neighbors
            for pair in neighbors:
                if pair[0] not in bNei_a_dic[hop].keys():
                    bNei_a_dic[hop][pair[0]] = []
                bNei_a_dic[hop][pair[0]].append((item,pair[1]))
    nei_set_a = set()
    for nn in a_set:
        if nn in Nei_a_dic[hop].keys():
            li = set([nei_pair[0] for nei_pair in Nei_a_dic[hop][nn]])
            nei_dic_a[hop].update(li)
            nei_set_a.update(li)
    temp_a_count = len(nei_set_a)

    for item in b_set:
        if item in kg.keys():
            neighbors = kg[item]
        else:
            neighbors = []
        if item not in Nei_b_dic[hop].keys() and neighbors:
            Nei_b_dic[hop][item] = []
            Nei_b_dic[hop][item] += neighbors
            for pair in neighbors:
                if pair[0] not in bNei_b_dic[hop].keys():
                    bNei_b_dic[hop][pair[0]] = []
                bNei_b_dic[hop][pair[0]].append((item,pair[1]))
    nei_set_b = set()
    for nn in b_set:
        if nn in Nei_b_dic[hop].keys():
            li = set([nei_pair[0] for nei_pair in Nei_b_dic[hop][nn]])
            nei_dic_b[hop].update(li)
            nei_set_b.update(li)
    temp_b_count = len(nei_set_b)

    return nei_set_a, nei_set_b

def pruning_nei(node_a,node_b,a_set,b_set,nei_dic_a, nei_dic_b,Nei_a_dic, Nei_b_dic,
                bNei_a_dic, bNei_b_dic, hop,x,y):

    orig_a = set(node_a)
    orig_b = set(node_b)
    orig_all = len(orig_a) + len(orig_b)
    current_set = a_set.union(b_set)
    current_num = len(current_set)
    nei_set_a = a_set - orig_a
    nei_set_b = b_set - orig_b
    temp_a_count = len(nei_set_a)
    temp_b_count = len(nei_set_b)
    if current_num < max_Nei:  # kg-extra-sample
        rest_count = max_Nei - current_num
        if temp_a_count > 0 and temp_b_count > 0:
            sam_a_count = int(np.ceil(rest_count * (len(a_set) / current_num)))
            sam_b_count = rest_count - sam_a_count
        else:
            sam_a_count = int(np.ceil(rest_count * (len(orig_a) / orig_all)))
            sam_b_count = rest_count - sam_a_count
        return sample_kg(sam_a_count,sam_b_count,a_set,b_set,hop,Nei_a_dic, Nei_b_dic)
    elif current_num == max_Nei: # self-sample
        return sample_self(a_set,b_set,hop,Nei_a_dic, Nei_b_dic)
    else:
        if x:  # a-->b connected
            find_path(x, hop, orig_a, node_a, bNei_a_dic,temp_a_count)
        if y:  # b-->a connected
            find_path(y, hop, orig_b, node_b, bNei_b_dic,temp_b_count)
        rest_count = max_Nei - len(orig_a.union(orig_b))
        sam_a_count = int(np.ceil(rest_count * (len(a_set) / current_num)))
        sam_b_count = rest_count - sam_a_count
        if temp_a_count < sam_a_count:  # a不够
            delta = sam_a_count - temp_a_count
            sam_a_count = temp_a_count
            sam_b_count = sam_b_count + delta
            return sample_frequency(node_a,node_b,sam_a_count,sam_b_count,a_set,b_set,nei_dic_a, nei_dic_b,hop,Nei_a_dic, Nei_b_dic,1,bNei_a_dic, bNei_b_dic)
        elif temp_b_count < sam_b_count:  # b不够
            delta = sam_b_count - temp_b_count
            sam_b_count = temp_b_count
            sam_a_count = sam_a_count + delta
            return sample_frequency(node_a,node_b,sam_a_count,sam_b_count,a_set,b_set,nei_dic_a, nei_dic_b,hop,Nei_a_dic, Nei_b_dic,2,bNei_a_dic, bNei_b_dic)
        else:
            return sample_frequency(node_a,node_b,sam_a_count,sam_b_count,a_set,b_set,nei_dic_a, nei_dic_b,hop,Nei_a_dic, Nei_b_dic,3,bNei_a_dic, bNei_b_dic)

def find_path(Z,hop,origset,orignode,bNeidict,tempcount):
    pathnode = set()
    pathnode.update(Z)
    for i in range(hop+1):
        j = hop - i
        for item in Z:
            if item in bNeidict[j].keys():
                backnode = set([pair[0] for pair in bNeidict[j][item]])
                pathnode.update(backnode)
        Z = pathnode - Z
    mid = pathnode-Z
    origset.update(mid)
    for i in mid:
        orignode.append(i)
        tempcount -= 1

def sample_kg(sam_a_count,sam_b_count,a_set,b_set,hop,Nei_a_dic, Nei_b_dic):
    new_dict_a, new_dict_b = {},{}
    c = a_set.union(b_set)
    pool = set(kg.keys()) - c
    extra_a_list = np.random.choice(list(pool), size=sam_a_count, replace=False)
    new_node_a = a_set.union(set(extra_a_list))
    pool = pool - set(extra_a_list)
    extra_dic_a = {}
    if not Nei_a_dic[0].keys():
        ind = np.random.choice(list(Nei_b_dic[0].keys()), size=1, replace=False)[0]
    else:
        ind = np.random.choice(list(Nei_a_dic[0].keys()), size=1, replace=False)[0]
    extra_dic_a[ind] = []
    for item in extra_a_list:
        extra_dic_a[ind] += [(item, -1)]
    for h in range(hop+1):
        for i in Nei_a_dic[h].keys():
            if i not in new_dict_a.keys():
                new_dict_a[i] = []
            new_dict_a[i] += Nei_a_dic[h][i]
    if ind not in new_dict_a.keys():
        new_dict_a[ind] = []
    new_dict_a[ind] += extra_dic_a[ind]

    extra_b_list = np.random.choice(list(pool), size=sam_b_count, replace=False)
    new_node_b = b_set.union(extra_b_list)
    extra_dic_b = {}
    if not Nei_b_dic[0].keys():
        ind = np.random.choice(list(Nei_a_dic[0].keys()), size=1, replace=False)[0]
    else:
        ind = np.random.choice(list(Nei_b_dic[0].keys()), size=1, replace=False)[0]
    extra_dic_b[ind] = []
    for item in extra_b_list:
        extra_dic_b[ind] += [(item, -1)]
    for h in range(hop+1):
        for i in Nei_b_dic[h].keys():
            if i not in new_dict_b.keys():
                new_dict_b[i] = []
            new_dict_b[i] += Nei_b_dic[h][i]
    if ind not in new_dict_b.keys():
        new_dict_b[ind] = []
    new_dict_b[ind] += extra_dic_b[ind]
    return new_node_a, new_node_b, new_dict_a, new_dict_b

def sample_self(a_set,b_set,hop,Nei_a_dic, Nei_b_dic):
    new_dict_a, new_dict_b = {}, {}
    for h in range(hop+1):
        for i in Nei_a_dic[h].keys():
            if i not in new_dict_a.keys():
                new_dict_a[i] = []
            new_dict_a[i] += Nei_a_dic[h][i]
    for h in range(hop+1):
        for i in Nei_b_dic[h].keys():
            if i not in new_dict_b.keys():
                new_dict_b[i] = []
            new_dict_b[i] += Nei_b_dic[h][i]
    return a_set,b_set, new_dict_a, new_dict_b

def sample_frequency(node_a,node_b,sam_a_count,sam_b_count,a_set,b_set,
                     nei_dic_a,nei_dic_b,hop,Nei_a_dic, Nei_b_dic,k,
                     bNei_a_dic, bNei_b_dic):
    new_node_a, new_node_b = set(), set()
    new_dict_a, new_dict_b = {}, {}
    if k == 1:
        new_node_a = a_set
        new_dict_a = {}
        for h in range(hop+1):
            for i in Nei_a_dic[h].keys():
                if i not in new_dict_a.keys():
                    new_dict_a[i] = []
                new_dict_a[i] += Nei_a_dic[h][i]
        new_node_b = set(node_b)
        new_dict_b = {}
        for h in range(hop+1):
            temp_b = nei_dic_b[h]-(new_node_a-set(node_a))
            temp_b_rel = Nei_b_dic[h]
            need_num = sam_b_count - len(new_node_b - set(node_b) - set(new_node_a))
            if len(temp_b) <= need_num:
                new_node_b.update(temp_b)
                for item in temp_b_rel.keys():
                    if item not in new_dict_b.keys():
                        new_dict_b[item] = []
                    new_dict_b[item] += temp_b_rel[item]
            else:
                c = new_node_b.union(new_node_a)
                temp_fre = read_embsim_temp(temp_b, need_num, c, bNei_b_dic, h)
                new_node_b.update(temp_fre)
                new_dict_b = add_rel(temp_fre, temp_b_rel, new_dict_b)  # 增加对应的rel
                break

    if k == 2: # a采样，b不变
        new_node_b = b_set
        new_dict_b = {}
        for h in range(hop+1):  # 把每一层的关系都加入
            for i in Nei_b_dic[h].keys():
                if i not in new_dict_b.keys():
                    new_dict_b[i] = []
                new_dict_b[i] += Nei_b_dic[h][i]
        new_node_a = set(node_a) #具有原始的node-a
        new_dict_a = {}
        for h in range(hop+1):
            temp_a = nei_dic_a[h]-(new_node_b-set(node_b))  # 拿到当前hop的nei
            temp_a_rel = Nei_a_dic[h]  # 拿到当前hop的nei-rel
            need_num = sam_a_count - len(new_node_a - set(node_a) - set(new_node_b))
            if len(temp_a) <= need_num:  # 把该hop的都加入
                new_node_a.update(temp_a)
                for item in temp_a_rel.keys():
                    if item not in new_dict_a.keys():
                        new_dict_a[item] = []
                    new_dict_a[item] += temp_a_rel[item]
            else:  # 到该层stop,从该层中按照fre-sam
                c = new_node_a.union(set(node_b))
                # temp_fre = read_fre_temp(temp_a,need_num,c)
                temp_fre = read_embsim_temp(temp_a, need_num, c, bNei_a_dic, h)
                # print('top-k-a:', temp_fre[:need_num])
                # x = set([ent[0] for ent in temp_fre[:need_num]])
                new_node_a.update(temp_fre)
                new_dict_a = add_rel(temp_fre, temp_a_rel, new_dict_a)  # 增加对应的rel
                # if len(new_node_a - set(node_a) - new_node_b) < need_num:
                #     print('having repeat sample in node-a!!!')
                #     e_num = need_num - len(new_node_a - set(node_a) - new_node_b)
                #     x = set([ent[0] for ent in temp_fre[need_num:need_num + e_num]])
                #     new_node_a.update(x)
                #     new_dict_a = add_rel(x, temp_a_rel, new_dict_a)  # 增加对应的rel
                #     # need_num += e_num
                break

    if k == 3: # a,b均采样
        ########### 对a采样 ###################
        print('k3--sample--a')
        print('for A sampling............')
        # print(node_a)
        new_node_a = set(node_a)  # 具有原始的node-a加上路径上的mid-node
        new_dict_a = {}
        for h in range(hop+1):
            temp_a = nei_dic_a[h]  # 拿到当前hop的nei
            temp_a_rel = Nei_a_dic[h]  # 拿到当前hop的nei-rel
            need_num = sam_a_count - len(new_node_a-set(node_a)-set(node_b))
            print('need-num:',need_num)
            print(len(temp_a))
            if len(temp_a) <= need_num:  # 把该hop的都加入
                new_node_a.update(temp_a)
                for item in temp_a_rel.keys():
                    if item not in new_dict_a.keys():
                        new_dict_a[item] = []
                    new_dict_a[item] += temp_a_rel[item]
            else:  # 到该层stop,从该层中按照fre-sam
                c = new_node_a.union(set(node_b))
                # temp_fre = read_fre_temp(temp_a,need_num,c) # 按照fre采样
                temp_fre = read_embsim_temp(temp_a, need_num, c,bNei_a_dic,h) # 按照emb-sim采样
                print('top-k-a:',len(temp_fre))
                new_node_a.update(temp_fre)
                new_dict_a = add_rel(temp_fre, temp_a_rel, new_dict_a)  # 增加对应的rel
                break
        ########### 对b采样 ###################
        # print('k3--sample--b')
        print('for B sampling............')
        # print('new-node-a:', new_node_a)
        # print(node_b)
        new_node_b = set(node_b)  # 具有原始的node-b
        new_dict_b = {}
        for h in range(hop+1):
            temp_b = nei_dic_b[h]-(new_node_a-set(node_a))  # 拿到当前hop的nei
            temp_b_rel = Nei_b_dic[h]  # 拿到当前hop的nei-rel
            need_num = sam_b_count - len(new_node_b - set(node_b) - set(new_node_a))
            print('need-num:', need_num)
            print(len(temp_b))
            if len(temp_b) <= need_num:  # 把该hop的都加入
                new_node_b.update(temp_b)
                for item in temp_b_rel.keys():
                    if item not in new_dict_b.keys():
                        new_dict_b[item] = []
                    new_dict_b[item] += temp_b_rel[item]
            else:  # 到该层stop,从该层中按照fre-sam
                c = new_node_b.union(new_node_a)
                # temp_fre = read_fre_temp(temp_b,need_num,c)
                temp_fre = read_embsim_temp(temp_b, need_num, c, bNei_b_dic, h)
                print('top-k-b:', len(temp_fre))
                # x = set([ent[0] for ent in temp_fre[:need_num]])
                new_node_b.update(temp_fre)
                new_dict_b = add_rel(temp_fre, temp_b_rel, new_dict_b)  # 增加对应的rel
                # if len(new_node_b - set(node_b) - new_node_a) < need_num:
                #     # print('having repeat sample in node-b!!!')
                #     e_num = need_num - len(new_node_b - set(node_b) - new_node_a)
                #     x = set([ent[0] for ent in temp_fre[need_num:need_num+e_num]])
                #     new_node_b.update(x)
                #     new_dict_b = add_rel(x, temp_b_rel, new_dict_b)  # 增加对应的rel
                #     # need_num += e_num
                # print('new-node-b:',new_node_b)
                break
    return new_node_a, new_node_b, new_dict_a, new_dict_b

def read_embsim_temp(nei_set_temp,need_num,nodeset,bNeidict,hop):
    score = []
    node_temp = []
    for item in nei_set_temp:
        if item in bNeidict[hop].keys():
            backnode = set([pair[0] for pair in bNeidict[hop][item]])
            for s in cal_sim_p(item, backnode):
                score.append(s)
            for p in backnode:
                node_temp.append(item)
    ind = np.argsort(score).tolist()
    ind.reverse()
    a = [node_temp[ii] for ii in ind]
    get_ent_fre = set(a)
    fre_set = set()
    count = 0
    for item in get_ent_fre:
        if count < need_num:
            if item not in nodeset:
                fre_set.add(item)
                count += 1
        else:
            break
    return fre_set

def cal_sim_p(item,backnode):
    item_vec = get_line_context(pretrain_emb_path,item+1)[1:]
    backnode_vec = [get_line_context(pretrain_emb_path,i+1)[1:] for i in backnode]
    score = []
    for vec in backnode_vec:
        vec_score = cos_sim(item_vec,vec)
        score.append(vec_score)
    return score

def get_line_context(file_path, line_number):
    line = linecache.getline(file_path, line_number).strip().split(' ')
    fltline = list(map(float, line))
    return np.array(fltline)

def add_rel(node, rel_dic,new_dic):
    for ii in rel_dic.keys():
        if not rel_dic[ii]:
            continue
        nodeli = list(np.array(rel_dic[ii])[:, 0])
        relli = list(np.array(rel_dic[ii])[:, 1])
        for jj in range(len(nodeli)):
            if nodeli[jj] in node:
                r = relli[jj]
                if ii not in new_dic.keys():
                    new_dic[ii] = []
                new_dic[ii] += [(nodeli[jj],r)]
    return new_dic

def IsConnect3(nei_set_a, nei_set_b, node_a, node_b):
    orig_a = set(node_a)
    orig_b = set(node_b)
    flag = False
    x = nei_set_a.intersection(orig_b)
    y = nei_set_b.intersection(orig_a)
    if x or y:
        flag = True
    return flag,x,y

def load_kg():
    if os.path.exists(kg_file + '.npy'):
        kg_np = np.load(kg_file + '.npy')
    else:
        kg_np = np.loadtxt(kg_file + '.txt', dtype=np.int64, delimiter='\t')
        np.save(kg_file + '.npy', kg_np)
    kg = collections.defaultdict(list)
    for head, relation, tail in kg_np:
        kg[head].append((tail, relation))
    return kg

def cos_sim(vector_a, vector_b):
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim

if __name__ == '__main__':
    train_path = ''
    valid_path = ''
    test_path = ''
    ent2ind_a_p = ''
    ent2ind_b_p = ''
    item2ind_a_p = ''
    item2ind_b_p = ''
    entity_id2index_A = dict()
    entity_id2index_B = dict()
    item_index_old2new_A = dict()
    item_index_old2new_B = dict()
    read_all_responding_index()
    entity_border = len(entity_id2index_A) - 1
    item_border = len(item_index_old2new_A) - 1
    print('entity-border:', entity_border)
    print('item-border:', item_border)

    kg_file = ''
    train_file = ''
    valid_file = ''
    test_file = ''
    max_H = 2
    max_Nei = 200

    kg = load_kg()
    print('kg load done...')

    pretrain_emb_path = ''

    traindata = getdata(train_path)
    processdata(traindata, train_file)

