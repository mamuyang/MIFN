import os
import tensorflow as tf
import math
from keras.utils.np_utils import to_categorical
from scipy import sparse


class FilterCell(tf.contrib.rnn.RNNCell):
    def __init__(self, num_units, activation=None, reuse=None, kernel_initializer=None,
                 bias_initializer=None):
        super(FilterCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._activation = activation or tf.tanh
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def call(self, inputs, state):
        inputs_A, inputs_T = tf.split(inputs, num_or_size_splits=2, axis=1)
        if self._kernel_initializer is None:
            self._kernel_initializer = tf.contrib.layers.xavier_initializer(uniform=False)
        if self._bias_initializer is None:
            self._bias_initializer = tf.constant_initializer(1.0)
        with tf.variable_scope('gate'):  # sigmoid([i_A|i_T|s_(t-1)]*[W_fA;W_fT;U_f]+b_f)
            self.W_f = tf.get_variable(dtype=tf.float32, name='W_f',
                                       shape=[inputs.get_shape()[-1].value + state.get_shape()[-1].value, self._num_units],
                                       initializer=self._kernel_initializer)
            self.b_f = tf.get_variable(dtype=tf.float32, name='b_f', shape=[self._num_units, ],
                                       initializer=self._bias_initializer)
            f = tf.concat([inputs, state], axis=-1)  # f=[batch_size, hidden_size+hidden_size+self._num_units]
            f = tf.matmul(f, self.W_f)  # f=[batch_size,self._num_units]
            f = f + self.b_f  # f=[batch_size, self._num_units]
            f = tf.sigmoid(f) # f=[batch_size, self._num_units]

        with tf.variable_scope('candidate'):  # tanh([i_A|s_(t-1)]*[W_s;U_s]+b_s)
            self.W_s = tf.get_variable(dtype=tf.float32, name='W_s',
                                       shape=[inputs_A.get_shape()[-1].value + state.get_shape()[-1].value,
                                              self._num_units], initializer=self._kernel_initializer)
            self.b_s = tf.get_variable(dtype=tf.float32, name='b_s', shape=[self._num_units, ],
                                       initializer=self._bias_initializer)
            _s = tf.concat([inputs_A, state], axis=-1)  # _s=[batch_size, hidden_size+self._num_units]
            _s = tf.matmul(_s, self.W_s)  # _s=[batch_size,self._num_units]
            _s = _s + self.b_s  # _s=[batch_size,self._num_units]
            _s = self._activation(_s)

        new_s = f * _s + (1 - f) * state  # new_s=[batch_size, self._num_units]
        return new_s, new_s

class MIFN():

    def __init__(self, num_items_A, num_items_B, num_entity_A,num_entity_B,num_cate,batch_size, neighbor_num, gpu,
                 embedding_size=256, hidden_size=256, num_layers=1,
                 lr=0.01,keep_prob=0.8,n_iter=1,
                 training_steps_per_epoch=5,lr_decay_factor=0.9,min_lr=0.00001):
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.num_items_A = num_items_A
        self.num_items_B = num_items_B
        self.n_items = num_entity_A + num_entity_B + num_cate
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.neighbors_num = neighbor_num
        self.n_iter = n_iter
        self.keep_prob = keep_prob
        self.graph = tf.Graph()

        with self.graph.as_default():
            tf.set_random_seed(1)
            with tf.name_scope('inputs'):
                self.get_inputs()
            with tf.name_scope('all_encoder'):
                self.all_encoder()
            with tf.name_scope('encoder_A'):
                encoder_output_A, encoder_state_A = self.encoder_A()
            with tf.name_scope('encoder_B'):
                encoder_output_B, encoder_state_B = self.encoder_B()
            with tf.name_scope('sequence_transfer_A'):
                filter_output_A, filter_state_A = self.filter_A(encoder_output_A, encoder_output_B,)
                transfer_output_A, transfer_state_A = self.transfer_A(filter_output_A,)
            with tf.name_scope('sequence_transfer_B'):
                filter_output_B, filter_state_B = self.filter_B(encoder_output_B, encoder_output_A,)
                transfer_output_B, transfer_state_B = self.transfer_B(filter_output_B,)

            with tf.name_scope('graph_transfer'):
                entity_emb = self.graph_gnn(encoder_output_A, encoder_output_B,
                                            transfer_output_A, transfer_output_B)

            with tf.name_scope('prediction_A'):
                self.PG_A, self.PS_A = self.switch_A(encoder_state_A, transfer_state_B, entity_emb,self.nei_A_mask,)
                s_pred_A = self.s_decoder_A(self.num_items_A, encoder_state_A, transfer_state_B, self.keep_prob)
                g_pred_A,g_att_A = self.g_decoder_A(encoder_state_A,entity_emb,self.num_items_A,self.nei_A_mask,
                                                    self.nei_index_A, self.IsinnumA)
                self.pred_A = self.final_pred_A(self.PG_A, self.PS_A, s_pred_A, g_pred_A)

            with tf.name_scope('prediction_B'):
                self.PG_B, self.PS_B = self.switch_B(encoder_state_B, transfer_state_A, entity_emb,self.nei_B_mask,)
                s_pred_B = self.s_decoder_B(self.num_items_B, encoder_state_B, transfer_state_A, self.keep_prob)
                g_pred_B,g_att_B = self.g_decoder_B(encoder_state_B, entity_emb, self.num_items_B,self.nei_B_mask,
                                                    self.nei_index_B, self.IsinnumB)
                self.pred_B = self.final_pred_B(self.PG_B, self.PS_B, s_pred_B, g_pred_B)

            with tf.name_scope('loss'):
                self.loss = self.cal_loss(self.target_A, self.pred_A, self.target_B, self.pred_B,)

            with tf.name_scope('optimizer'):
                self.train_op,self.grad = self.optimizer(lr,training_steps_per_epoch,lr_decay_factor,min_lr)

    def get_inputs(self):
        self.seq_A = tf.placeholder(dtype=tf.int32, shape=[None, None], name='seq_A')
        self.seq_B = tf.placeholder(dtype=tf.int32, shape=[None, None], name='seq_B')
        self.len_A = tf.placeholder(dtype=tf.int32, shape=[None, ], name='len_A')
        self.len_B = tf.placeholder(dtype=tf.int32, shape=[None, ], name='len_B')
        self.pos_A = tf.placeholder(dtype=tf.int32, shape=[None, None, 2], name='pos_A')
        self.pos_B = tf.placeholder(dtype=tf.int32, shape=[None, None, 2], name='pos_B')
        self.index_A = tf.placeholder(dtype=tf.int32, shape=[None, None, 2], name='index_A')
        self.index_B = tf.placeholder(dtype=tf.int32, shape=[None, None, 2], name='index_B')
        self.target_A = tf.placeholder(dtype=tf.int32,shape=[None,],name='target_A')
        self.target_B = tf.placeholder(dtype=tf.int32,shape=[None,],name='target_B')
        self.tar_in_A = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='tar_in_A')
        self.tar_in_B = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='tar_in_B')
        self.adj_1 = tf.placeholder(dtype=tf.float32, shape=[None,self.neighbors_num,self.neighbors_num],name='adj_alb')
        self.adj_2 = tf.placeholder(dtype=tf.float32, shape=[None,self.neighbors_num,self.neighbors_num],name='adj_alv')
        self.adj_3 = tf.placeholder(dtype=tf.float32, shape=[None,self.neighbors_num,self.neighbors_num],name='adj_bav')
        self.adj_4 = tf.placeholder(dtype=tf.float32, shape=[None,self.neighbors_num,self.neighbors_num],name='adj_bt')
        self.adj_5 = tf.placeholder(dtype=tf.float32, shape=[None,self.neighbors_num,self.neighbors_num],name='adj_ada')
        self.neighbors = tf.placeholder(dtype=tf.int64, shape=[None, self.neighbors_num],name='neighbors')
        self.nei_index_A = tf.placeholder(dtype=tf.int64, shape=[None, self.neighbors_num, 2],name='nei_index_A')
        self.nei_index_B = tf.placeholder(dtype=tf.int64, shape=[None, self.neighbors_num, 2], name='nei_index_B')
        self.nei_A_mask = tf.placeholder(dtype=tf.float32, shape=[None, self.neighbors_num], name='nei_A_mask')
        self.nei_B_mask = tf.placeholder(dtype=tf.float32, shape=[None, self.neighbors_num], name='nei_B_mask')
        self.IsinnumA = tf.placeholder(dtype=tf.float32, shape=[None, self.neighbors_num], name='IsinnumA')
        self.IsinnumB = tf.placeholder(dtype=tf.float32, shape=[None, self.neighbors_num], name='IsinnumB')

        self.nei_L_A_mask = tf.placeholder(dtype=tf.float32, shape=[None, None, self.neighbors_num],
                                           name='nei_L_A_mask')
        self.nei_L_T_mask = tf.placeholder(dtype=tf.float32, shape=[None, None, self.neighbors_num],
                                           name='nei_L_T_mask')

    def get_gru_cell(self,hidden_size,keep_prob):
        gru_cell = tf.contrib.rnn.GRUCell(hidden_size, kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        gru_cell = tf.contrib.rnn.DropoutWrapper(gru_cell, input_keep_prob=keep_prob,
                                                 output_keep_prob=keep_prob,
                                                 state_keep_prob=keep_prob)
        return gru_cell

    def get_filter_cell(self, hidden_size, keep_prob):
        filter_cell = FilterCell(hidden_size)
        filter_cell = tf.contrib.rnn.DropoutWrapper(filter_cell, input_keep_prob=keep_prob, output_keep_prob=keep_prob,
                                                    state_keep_prob=keep_prob)
        return filter_cell

    def all_encoder(self,):
        with tf.variable_scope('all_encoder'):
            self.all_emb_matrix = tf.get_variable(shape=[self.n_items, self.embedding_size], name='item_emb_matrix',
                                                  initializer=tf.contrib.layers.xavier_initializer(uniform=False))

    def encoder_A(self,):
        with tf.variable_scope('encoder_A'):
            embedd_seq_A = tf.nn.embedding_lookup(self.all_emb_matrix, self.seq_A)
            print(embedd_seq_A)
            encoder_cell_A = tf.nn.rnn_cell.MultiRNNCell([self.get_gru_cell(self.hidden_size,self.keep_prob) for _ in range(self.num_layers)])
            encoder_output_A, encoder_state_A = tf.nn.dynamic_rnn(encoder_cell_A, embedd_seq_A, sequence_length=self.len_A, dtype=tf.float32)
        return encoder_output_A, encoder_state_A,

    def encoder_B(self,):
        with tf.variable_scope('encoder_B'):
            embedd_seq_B = tf.nn.embedding_lookup(self.all_emb_matrix, self.seq_B)
            print(embedd_seq_B)
            encoder_cell_B = tf.nn.rnn_cell.MultiRNNCell([self.get_gru_cell(self.hidden_size,self.keep_prob) for _ in range(self.num_layers)])
            encoder_output_B, encoder_state_B = tf.nn.dynamic_rnn(encoder_cell_B, embedd_seq_B, sequence_length=self.len_B, dtype=tf.float32)
        return encoder_output_B, encoder_state_B

    def filter_A(self, encoder_output_A, encoder_output_B,):
        with tf.variable_scope('filter_A'):
            zero_state = tf.zeros(dtype=tf.float32, shape=(tf.shape(encoder_output_A)[0], 1, tf.shape(encoder_output_A)[-1]))
            encoder_output = tf.concat([zero_state, encoder_output_B], axis=1)
            # print(encoder_output) #encoder_output=[batch_size,timestamp_B+1,hidden_size]
            select_output_B = tf.gather_nd(encoder_output,self.pos_A) # 挑出A-output之前的B-item, len还是timestep_A
            # print(select_output_B) #select_output_A=[batch_size,timestamp_A,hidden_size]
            filter_input_A = tf.concat([encoder_output_A, select_output_B], axis=-1)  # filter_input_A=[b,tA,2*h]
            # att = tf.layers.dense(tf.concat([encoder_output_A, select_output_B], axis=-1), units=hidden_size,activation=tf.nn.sigmoid)
            # combined_output_A = att * tf.nn.tanh(encoder_output_A) + (1 - att) * select_output_B
            # print(combined_output_A) #[batch_size,timestamp_A,hidden_size]
            filter_cell_A = tf.nn.rnn_cell.MultiRNNCell(
                [self.get_filter_cell(self.hidden_size, self.keep_prob) for _ in range(self.num_layers)])
            filter_output_A, filter_state_A = tf.nn.dynamic_rnn(filter_cell_A, filter_input_A, sequence_length=self.len_A,
                                                                dtype=tf.float32)
            # print(filter_output_A)  # filter_output_A=[batch_size,timestamp_A,hidden_size]，
            # print(filter_state_A)  # filter_state_A=[batch_size,hidden_size]
        return filter_output_A, filter_state_A

    def transfer_A(self, filter_output_A,):

        with tf.variable_scope('transfer_A'):
            transfer_cell_A = tf.nn.rnn_cell.MultiRNNCell(
                [self.get_gru_cell(self.hidden_size, self.keep_prob) for _ in range(self.num_layers)])
            transfer_output_A, transfer_state_A = tf.nn.dynamic_rnn(transfer_cell_A, filter_output_A,
                                                                    sequence_length=self.len_A,dtype=tf.float32)
            # print(transfer_output_A) # transfer_output_A=[batch_size,timestamp_A,hidden_size],
            # print(transfer_state_A)  # transfer_state_A=([batch_size,hidden_size]*num_layers)
        return transfer_output_A, transfer_state_A

    def filter_B(self, encoder_output_B, encoder_output_A, ):
        with tf.variable_scope('filter_B'):
            zero_state = tf.zeros(dtype=tf.float32,
                                  shape=(tf.shape(encoder_output_B)[0], 1, tf.shape(encoder_output_B)[-1]))
            # print(zero_state)  # zero_state=[batch_size,1,hidden_size]
            encoder_output = tf.concat([zero_state, encoder_output_A], axis=1)
            # print(encoder_output)  # encoder_output=[batch_size,timestamp_B+1,hidden_size]
            select_output_A = tf.gather_nd(encoder_output, self.pos_B)  # 挑出B-output之前的A-item
            # print(select_output_A)  # select_output_B=[batch_size,timestamp_B,hidden_size]
            filter_input_B = tf.concat([encoder_output_B, select_output_A], axis=-1)
            # print(filter_input_B)  # filter_input_B=[batch_size,timestamp_B,hidden_size+hidden_size]

            # att = tf.layers.dense(tf.concat([encoder_output_B, select_output_A], axis=-1), units=hidden_size,
            #                       activation=tf.nn.sigmoid)
            # combined_output_B = att * tf.nn.tanh(encoder_output_B) + (1 - att) * select_output_A
            # print(combined_output_B)  # [batch_size,timestamp_B,hidden_size]
            filter_cell_B = tf.nn.rnn_cell.MultiRNNCell(
                [self.get_filter_cell(self.hidden_size, self.keep_prob) for _ in range(self.num_layers)])
            filter_output_B, filter_state_B = tf.nn.dynamic_rnn(filter_cell_B, filter_input_B, sequence_length=self.len_B,
                                                                dtype=tf.float32)
            # print(filter_output_B)  # filter_output_B=[batch_size,timestamp_B,hidden_size]，
            # print(filter_state_B)  # filter_state_B=[batch_size,hidden_size]
        return filter_output_B, filter_state_B

    def transfer_B(self, filter_output_B,):
        with tf.variable_scope('transfer_B'):
            transfer_cell_B = tf.nn.rnn_cell.MultiRNNCell(
                [self.get_gru_cell(self.hidden_size, self.keep_prob) for _ in range(self.num_layers)])
            transfer_output_B, transfer_state_B = tf.nn.dynamic_rnn(transfer_cell_B, filter_output_B,
                                                                    sequence_length=self.len_B,
                                                                    dtype=tf.float32)
            # print(transfer_output_B)  # transfer_output_B=[batch_size,timestamp_B,hidden_size],
            # print(transfer_state_B)   # transfer_state_B=([batch_size,hidden_size]*num_layers)
        return transfer_output_B, transfer_state_B

    def graph_gnn(self, encoder_output_A, encoder_output_B,
                    transfer_output_A, transfer_output_B,):
        with tf.variable_scope('graph_gnn'):
            self.entity_emb = tf.nn.embedding_lookup(self.all_emb_matrix, self.neighbors) # [b,N,e]
            # ----------------- in-domain adj parameter ------------------
            self.W_alb1 = random_weight(self.hidden_size, self.hidden_size, name='W_alb1')
            self.b11 = random_bias(self.hidden_size, name='b11')
            self.W_alv1 = random_weight(self.hidden_size, self.hidden_size, name='W_alv1')
            self.b21 = random_bias(self.hidden_size, name='b21')
            self.W_bav1 = random_weight(self.hidden_size, self.hidden_size, name='W_bav1')
            self.b31 = random_bias(self.hidden_size, name='b31')
            self.W_bt1 = random_weight(self.hidden_size, self.hidden_size, name='W_bt1')
            self.b41 = random_bias(self.hidden_size, name='b41')
            self.W_ada1 = random_weight(self.hidden_size, self.hidden_size, name='W_ada1')
            self.b51 = random_bias(self.hidden_size, name='b51')

            # ------------------- cross-domain adj parameter ------------------
            self.W_alb2 = random_weight(self.hidden_size, self.hidden_size, name='W_alb2')
            self.b12 = random_bias(self.hidden_size, name='b12')
            self.W_alv2 = random_weight(self.hidden_size, self.hidden_size, name='W_alv2')
            self.b22 = random_bias(self.hidden_size, name='b22')
            self.W_bav2 = random_weight(self.hidden_size, self.hidden_size, name='W_bav2')
            self.b32 = random_bias(self.hidden_size, name='b32')
            self.W_bt2 = random_weight(self.hidden_size, self.hidden_size, name='W_bt2')
            self.b42 = random_bias(self.hidden_size, name='b42')
            self.W_ada2 = random_weight(self.hidden_size, self.hidden_size, name='W_ada2')
            self.b52 = random_bias(self.hidden_size, name='b52')

            inputs_A, inputs_A2T,\
            nei_mask_A, nei_mask_T = self.get_ht(encoder_output_A, encoder_output_B,
                                                transfer_output_A, transfer_output_B,
                                                 self.len_A, self.len_B, self.index_A, self.index_B,
                                                 self.nei_L_A_mask, self.nei_L_T_mask)
            inputs_A = tf.tile(tf.expand_dims(inputs_A, axis=1), [1, self.neighbors_num, 1])
            print('input-A:', inputs_A)  # [b, N, h]
            inputs_A2T = tf.tile(tf.expand_dims(inputs_A2T, axis=1), [1, self.neighbors_num, 1])
            print('input-A2T:', inputs_A2T)  # [b, N, h]

            cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)

            # --------------  softmask-A attention weight --------------
            self.W_s_in = random_weight(self.hidden_size, self.hidden_size, name='W_s_in')
            self.W_emb_in = random_weight(self.hidden_size, self.hidden_size, name='W_emb_in')
            self.W_v_in = random_weight(self.hidden_size, 1, name='W_v_in')
            # -------------- softmask-A2T attention weight -------------
            self.W_s_cross = random_weight(self.hidden_size, self.hidden_size, name='W_s_cross')
            self.W_emb_cross = random_weight(self.hidden_size, self.hidden_size, name='W_emb_cross')
            self.W_v_cross = random_weight(self.hidden_size, 1, name='W_v_cross')

            for i in range(self.n_iter):
                softmask_A = self.indomain_attention(inputs_A, nei_mask_A)
                softmask_A2T = self.crossdomain_attention(inputs_A2T, nei_mask_T)
                gcn_emb = self.get_neigh_rep(inputs_A, inputs_A2T,softmask_A, softmask_A2T, nei_mask_A,nei_mask_T)
                print(gcn_emb) # [b, N, 10*h]
                # self.entity_emb = tf.layers.dense(gcn_emb, self.hidden_size,
                #                                   activation=None,
                #                                   kernel_initializer=tf.contrib.layers.xavier_initializer(
                #                                       uniform=False))  # [b, N, h]
                self.entity_emb = tf.reshape(self.entity_emb, [-1, self.hidden_size])  # [b*N, h]
                graph_output, self.entity_emb = \
                    tf.nn.dynamic_rnn(cell, tf.expand_dims(tf.reshape(gcn_emb, [-1, 12 * self.hidden_size]), axis=1),
                                      initial_state=self.entity_emb)
                print(graph_output)  # graph_output=[b*N, h]，
                print(self.entity_emb)  # graph_state=[b*N, h]*num_layers

            self.entity_emb = tf.reshape(self.entity_emb, [-1, self.neighbors_num, self.hidden_size])
            print('after gnn:', self.entity_emb)
        return self.entity_emb

    def get_neigh_rep(self,inputs_A, inputs_A2T,softmask_A, softmask_A2T, nei_mask_A,nei_mask_T):
        with tf.variable_scope('cdgcn', reuse=tf.AUTO_REUSE):

            self.W_c = random_weight(3 * self.hidden_size, self.hidden_size,
                                     name='W_f')
            self.b_c = random_bias(self.hidden_size, name='b_f')
            inputs = tf.concat([inputs_A, inputs_A2T], axis=-1)  # [b, N, 2h]
            f = tf.concat([inputs, self.entity_emb], axis=-1)  # [b, N, 3h]
            f = tf.matmul(f, self.W_c) + self.b_c
            f = tf.sigmoid(f)  # [b, N, h]
            print('cross-gate:', f)  # f=[b, N, h]
            softmask_A = f * softmask_A
            softmask_A2T = (1 - f) * softmask_A2T

            #--------------- in-domain neighbor ------------------
            fin_state_A = tf.reshape(softmask_A, [-1, self.hidden_size])
            self.W1 = random_weight(self.hidden_size, self.hidden_size, name='w1')
            self.b1 = random_bias(self.hidden_size, name='b1')
            fin_state_A = tf.matmul(fin_state_A, self.W1) + self.b1
            print(fin_state_A)  # [b*N, s]

            #---------------- cross-domain neighbor ---------------
            fin_state_A2T = tf.reshape(softmask_A2T, [-1, self.hidden_size])
            self.W2 = random_weight(self.hidden_size, self.hidden_size, name='w2')
            self.b2 = random_bias(self.hidden_size, name='b2')
            fin_state_A2T = tf.matmul(fin_state_A2T, self.W2) + self.b2
            print(fin_state_A2T)  # [b*N, s]

            nei_mask_A = tf.expand_dims(nei_mask_A, -1)  # [b,N,1]
            nei_mask_T = tf.expand_dims(nei_mask_T, -1)  # [b,N,1]
            mask_emb_A = nei_mask_A * self.entity_emb # [b,N,h]
            mask_emb_T = nei_mask_T * self.entity_emb # [b,N,h]
            att_emb_T = tf.reshape(self.mutual_att(mask_emb_T, mask_emb_A), [-1, self.hidden_size])  # [b*N,s]
            fin_state_A2T = tf.add(fin_state_A2T, att_emb_T)
            print(fin_state_A2T)  # [b*N, s]

            # ------------------ in-domain representation --------------
            fin_state_1a = matrix_mutliply(fin_state_A, self.W_alb1, self.b11, self.neighbors_num, self.hidden_size)
            fin_state_2a = matrix_mutliply(fin_state_A, self.W_alv1, self.b21, self.neighbors_num, self.hidden_size)
            fin_state_3a = matrix_mutliply(fin_state_A, self.W_bav1, self.b31, self.neighbors_num, self.hidden_size)
            fin_state_4a = matrix_mutliply(fin_state_A, self.W_bt1, self.b41, self.neighbors_num, self.hidden_size)
            fin_state_5a = matrix_mutliply(fin_state_A, self.W_ada1, self.b51, self.neighbors_num, self.hidden_size)
            all_nei_A = tf.nn.relu(tf.concat([
                                    tf.matmul(self.adj_1, fin_state_1a),
                                    tf.matmul(self.adj_2, fin_state_2a),
                                    tf.matmul(self.adj_3, fin_state_3a),
                                    tf.matmul(self.adj_4, fin_state_4a),
                                    tf.matmul(self.adj_5, fin_state_5a),], axis=-1))
            print(all_nei_A)  # all_nei=[b, N, 5*h]

            ###################### cross-domain representation #################
            fin_state_1t = matrix_mutliply(fin_state_A2T, self.W_alb2, self.b12, self.neighbors_num, self.hidden_size)
            fin_state_2t = matrix_mutliply(fin_state_A2T, self.W_alv2, self.b22, self.neighbors_num, self.hidden_size)
            fin_state_3t = matrix_mutliply(fin_state_A2T, self.W_bav2, self.b32, self.neighbors_num, self.hidden_size)
            fin_state_4t = matrix_mutliply(fin_state_A2T, self.W_bt2, self.b42, self.neighbors_num, self.hidden_size)
            fin_state_5t = matrix_mutliply(fin_state_A2T, self.W_ada2, self.b52, self.neighbors_num, self.hidden_size)
            all_nei_A2T = tf.nn.relu(tf.concat([
                tf.matmul(self.adj_1, fin_state_1t),
                tf.matmul(self.adj_2, fin_state_2t),
                tf.matmul(self.adj_3, fin_state_3t),
                tf.matmul(self.adj_4, fin_state_4t),
                tf.matmul(self.adj_5, fin_state_5t), ], axis=-1))
            print(all_nei_A2T)  # all_nei=[b, N, s*5]
            all_nei = tf.concat([all_nei_A, all_nei_A2T], axis=-1)
            print(all_nei)  # [b, N, 10*s]

        return all_nei

    def get_ht(self,encoder_output_A, encoder_output_B,transfer_output_A, transfer_output_B,
               len_A, len_B, index_A, index_B, nei_L_A_mask, nei_L_T_mask):
        all_len = tf.add(len_A, len_B)
        ########## get hAi from encoder ##########
        e1 = tf.scatter_nd(index_A, encoder_output_A, [tf.shape(encoder_output_A)[0],
                                                       tf.shape(encoder_output_A)[1] + tf.shape(encoder_output_B)[1],
                                                       self.hidden_size])
        print(e1)
        e2 = tf.scatter_nd(index_B, encoder_output_B, [tf.shape(encoder_output_A)[0],
                                                       tf.shape(encoder_output_A)[1] + tf.shape(encoder_output_B)[1],
                                                       self.hidden_size])
        print(e2)
        seq_L = e1 + e2  # [b, time_A+time_B, h]
        hA = tf.gather_nd(seq_L, tf.stack([tf.range(tf.shape(encoder_output_A)[0]), all_len-1], axis=1))  # 拿到了最后一步的输入
        print('inputsA:',hA)
        ########## get h(A->B)i from transfer ##########
        e3 = tf.scatter_nd(index_A, transfer_output_A, [tf.shape(transfer_output_A)[0],
                                                        tf.shape(transfer_output_A)[1] + tf.shape(transfer_output_B)[1],
                                                        self.hidden_size])
        print(e3)
        e4 = tf.scatter_nd(index_B, transfer_output_B, [tf.shape(transfer_output_A)[0],
                                                        tf.shape(transfer_output_A)[1] + tf.shape(transfer_output_B)[1],
                                                        self.hidden_size])
        print(e4)
        trans_L = e3 + e4  # [b, time_A+time_B, h]
        hA2T = tf.gather_nd(trans_L, tf.stack([tf.range(tf.shape(encoder_output_A)[0]), all_len-1], axis=1))  # 拿到了最后一步的输入
        print('inputsA2T:', hA2T)
        ########## get mask ###############
        print(tf.shape(nei_L_A_mask)[1])
        nei_A_mask = tf.gather_nd(nei_L_A_mask, tf.stack([tf.range(tf.shape(encoder_output_A)[0]), all_len-1], axis=1))
        print('neiA_mask:', nei_A_mask) # [b, N]
        nei_T_mask = tf.gather_nd(nei_L_T_mask, tf.stack([tf.range(tf.shape(encoder_output_A)[0]), all_len-1], axis=1))
        print('neiA2T_mask:', nei_T_mask) # [b, N]

        return hA, hA2T,nei_A_mask, nei_T_mask

    def indomain_attention(self, item_state, nei_mask):

        with tf.variable_scope('softmask_att_in', reuse=tf.AUTO_REUSE):
            S_it = tf.matmul(item_state, self.W_s_in)
            print('S_it:', S_it)  # [b, N, h]
            S_emb = tf.matmul(self.entity_emb, self.W_emb_in)
            print('S_emb:', S_emb)  # [b, N, h]
            tanh = tf.tanh(S_it + S_emb)
            print("tanh:", tanh)  # [b, N, h]
            s = tf.squeeze(tf.matmul(tanh, self.W_v_in))
            print("s:", s)  # [b, N]
            s_inf_mask = self.mask_softmax(nei_mask, s)
            print(s_inf_mask)  # [b, N]
            score = self.normalize_softmax(s_inf_mask)  # [b, N]
            score = tf.expand_dims(score, axis=-1)
            print('score:', score)  # [b, N, 1]
            softmask = score * self.entity_emb  # [b, N, e]
        return softmask

    def crossdomain_attention(self, item_state, nei_mask):
        with tf.variable_scope('softmask_att_cross', reuse=tf.AUTO_REUSE):
            S_it = tf.matmul(item_state, self.W_s_cross)
            print('S_it:', S_it)  # [b, N, h]
            S_emb = tf.matmul(self.entity_emb, self.W_emb_cross)
            print('S_emb:', S_emb)  # [b, N, h]
            tanh = tf.tanh(S_it + S_emb)
            print("tanh:", tanh)  # [b, N, h]
            s = tf.squeeze(tf.matmul(tanh, self.W_v_cross))
            print("s:", s)  # [b, N]
            s_inf_mask = self.mask_softmax(nei_mask, s)
            print(s_inf_mask)  # [b, N]
            score = self.normalize_softmax(s_inf_mask)  # [b, N]
            score = tf.expand_dims(score, axis=-1)
            print('score:', score)  # [b, N, 1]
            softmask = score * self.entity_emb  # [b, N, e]
        return softmask

    def mask_softmax(self, seq_mask, scores):
        '''
        to do softmax, assign -inf value for the logits of padding tokens
        '''
        seq_mask = tf.cast(seq_mask, tf.bool)
        score_mask_values = -1e10 * tf.ones_like(scores, dtype=tf.float32)
        return tf.where(seq_mask, scores, score_mask_values)

    def normalize_softmax(self,x):
        max_value = tf.reshape(tf.reduce_max(x, -1), [-1, 1])
        each_ = tf.exp(x - max_value)
        all_ = tf.reshape(tf.reduce_sum(each_, -1), [-1, 1])
        score = each_ / all_
        return score

    def mutual_att(self,hb, hA,):
        hb_ext = tf.expand_dims(hb, axis=2)  # hb_ext=[b,N1,1,h]
        hb_ext = tf.tile(hb_ext, [1, 1, tf.shape(hA)[1], 1])  # hb_ext=[b,N1,N2,h]
        hA_ext = tf.expand_dims(hA, axis=1)  # hA_ext=[b,1,N2,h]
        hA_ext = tf.tile(hA_ext, [1, tf.shape(hb)[1], 1, 1])  # hA_ext=[b,N1,N2,h]
        dot = hb_ext * hA_ext
        # dot = tf.concat([hb_ext, hA_ext, hb_ext * hA_ext], axis=-1)  # dot=[b,N1,N2,h]
        dot = tf.layers.dense(dot, 1, activation=None, use_bias=False)  # dot=[b,N1,N2,1]
        dot = tf.squeeze(dot)  # dot=[b,N1,N2]
        # sum_row = tf.reduce_sum(dot, axis=-1, keep_dims=True)  # sum_row=[b,N1,1]
        # att_hb = sum_row * hb
        # print(att_hb) # [b, N1, h]
        att_hb = tf.matmul(dot, hA)  # [b,N1,h]
        return att_hb

    def switch_A(self, encoder_state_A, transfer_state_B, graph_state, nei_mask):
        with tf.variable_scope('switch_A'):
            graph_rep = tf.reshape(graph_state, [-1, self.neighbors_num, self.hidden_size])
            nei_mask = tf.expand_dims(nei_mask, -1)
            graph_rep = nei_mask * graph_rep
            graph_rep = tf.reduce_sum(graph_rep, axis=1)
            concat_output = tf.concat([encoder_state_A[-1], transfer_state_B[-1], graph_rep], axis=-1)
            linear_switch = tf.layers.Dense(1, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
            switch_matrix = linear_switch(concat_output)  # Tensor shape (b, 1)
            PG_A = tf.sigmoid(switch_matrix)
            PS_A = 1 - PG_A
            print('PSA:',PS_A)
            print('PGA:',PG_A)
        return PG_A, PS_A

    def s_decoder_A(self, num_items_A, encoder_state_A, transfer_state_B, keep_prob):
        with tf.variable_scope('s_predict_A'):
            concat_output = tf.concat([encoder_state_A[-1],transfer_state_B[-1]],axis=-1)
            concat_output = tf.nn.dropout(concat_output, keep_prob)
            pred_A = tf.layers.dense(concat_output, num_items_A,
                                     activation=None,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            # pred_A = self.normalize_softmax(pred_A)
            pred_A = tf.nn.softmax(pred_A)
            # print(pred_A) # pred_A=[b, num_items_A]
        return pred_A

    def g_decoder_A(self, ht, graph_state, num_items_A, nei_mask, nei_index_A, IsinnumA):

        with tf.variable_scope('g_predict_A'):
            self.W_h_a = random_weight(self.hidden_size, self.hidden_size, name='W_h_a')
            self.W_emb_a = random_weight(self.hidden_size, self.hidden_size, name='W_emb_a')
            self.W_v_a = random_weight(self.hidden_size, 1, name='W_v_a')
            graph_state = tf.reshape(graph_state, [-1, self.neighbors_num, self.hidden_size])  # [b, N, h]
            nei_mask = tf.expand_dims(nei_mask, -1)
            graph_state = nei_mask * graph_state
            att = self.g_decode_attention_A(ht[-1], graph_state, IsinnumA)
            print(att)  # [b, N]
            g_pred_A = tf.scatter_nd(nei_index_A, att, [tf.shape(graph_state)[0], num_items_A])
            print(g_pred_A)  # [b, num_item_A]
        return g_pred_A, att
    def g_decode_attention_A(self, ht, repre, mask):
        S_h = tf.matmul(ht, self.W_h_a)  # [b, h]
        S_h = tf.expand_dims(S_h, 1)
        print('S_it:', S_h)  # [b, 1, h]
        S_emb = tf.reshape(tf.matmul(tf.reshape(repre, [-1, self.hidden_size]), self.W_emb_a),
                           [-1, self.neighbors_num, self.hidden_size])  # [b, N, h]
        print('S_emb:', S_emb)
        tanh = tf.tanh(S_h + S_emb)  # [b, N, h]
        print("tanh:", tanh)
        s = tf.reshape(tf.squeeze(tf.matmul(tf.reshape(tanh, [-1, self.hidden_size]), self.W_v_a)),
                       [-1, self.neighbors_num])  # [b, N]
        print("s:", s)  # [b, N]
        s_inf_mask = self.mask_softmax(mask, s)
        print(s_inf_mask) # [b, N]
        score = self.normalize_softmax(s_inf_mask)  # [b, N]
        print('score:', score)
        return score

    def final_pred_A(self, PG_A, PS_A, s_pred_A, g_pred_A):
        with tf.variable_scope('final_predict_A'):
            pred_A = PG_A * g_pred_A + PS_A * s_pred_A
            print(pred_A)  # [b, num_items_A]
        return pred_A

    def switch_B(self, encoder_state_B, transfer_state_A, graph_state, nei_mask):
        with tf.variable_scope('switch_B'):
            graph_rep = tf.reshape(graph_state, [-1, self.neighbors_num, self.hidden_size])
            nei_mask = tf.expand_dims(nei_mask, -1)
            graph_rep = nei_mask * graph_rep
            graph_rep = tf.reduce_sum(graph_rep, axis=1)
            concat_output = tf.concat([encoder_state_B[-1], transfer_state_A[-1], graph_rep], axis=-1)
            # print(concat_output)  # [batch_size, 3*hidden_size]
            linear_switch = tf.layers.Dense(1, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
            switch_matrix = linear_switch(concat_output)  # Tensor shape (b, 1)
            PG_B = tf.sigmoid(switch_matrix)
            PS_B = 1 - PG_B
            # PG_B = tf.expand_dims(PG_B, 1)  # [batch,1]
            # PS_B = tf.expand_dims(PS_B, 1)
        return PG_B, PS_B

    def s_decoder_B(self, num_items_B, encoder_state_B, transfer_state_A, keep_prob):
        with tf.variable_scope('s_predict_B'):
            concat_output = tf.concat([encoder_state_B[-1],transfer_state_A[-1]],axis=-1)
            concat_output = tf.nn.dropout(concat_output, keep_prob)
            pred_B = tf.layers.dense(concat_output, num_items_B,
                                     activation=None,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            pred_B = tf.nn.softmax(pred_B)
            # pred_B = self.normalize_softmax(pred_B)
            print(pred_B) # [b, num_B]
        return pred_B

    def g_decoder_B(self, ht, graph_state, num_items_B, nei_mask, nei_index_B, IsinnumB):
        with tf.variable_scope('g_predict_B'):
            self.W_h_b = random_weight(self.hidden_size, self.hidden_size, name='W_h_b')
            self.W_emb_b = random_weight(self.hidden_size, self.hidden_size, name='W_emb_b')
            self.W_v_b = random_weight(self.hidden_size, 1, name='W_v_b')
            graph_state = tf.reshape(graph_state, [-1, self.neighbors_num, self.hidden_size])  # [b, N, h]
            nei_mask = tf.expand_dims(nei_mask, -1)
            graph_state = nei_mask * graph_state
            att = self.g_decode_attention_B(ht[-1], graph_state, IsinnumB)  # [b, N]
            g_pred_B = tf.scatter_nd(nei_index_B, att, [tf.shape(graph_state)[0], num_items_B])
            print(g_pred_B)  # [b, num_item_B]
        return g_pred_B, att
    def g_decode_attention_B(self, ht, repre, mask):
        S_h = tf.matmul(ht, self.W_h_b)  # [b, h]
        S_h = tf.expand_dims(S_h, 1)
        print('S_it:', S_h)  # [b, 1, h]
        S_emb = tf.reshape(tf.matmul(tf.reshape(repre, [-1, self.hidden_size]), self.W_emb_b),
                           [-1, self.neighbors_num, self.hidden_size])  # [b, N, h]
        print('S_emb:', S_emb)
        tanh = tf.tanh(S_h + S_emb)  # [b, N, h]
        print("tanh:", tanh)
        s = tf.reshape(tf.squeeze(tf.matmul(tf.reshape(tanh, [-1, self.hidden_size]), self.W_v_b)),
                       [-1, self.neighbors_num])  # [b, N]
        print("s:", s)  # [b, N]
        s_inf_mask = self.mask_softmax(mask, s)
        print(s_inf_mask) # [b, N]
        score = self.normalize_softmax(s_inf_mask)  # [b, N]
        print('score:', score)
        return score

    def final_pred_B(self, PG_B, PS_B, s_pred_B, g_pred_B):
        with tf.variable_scope('final_predict_B'):
            pred_B = PG_B * g_pred_B + PS_B * s_pred_B
            print(pred_B)
        return pred_B

    def cal_loss(self, target_A, pred_A, target_B, pred_B,):

        # loss_A = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_A, logits=pred_A)
        loss_A = tf.contrib.keras.losses.sparse_categorical_crossentropy(target_A,pred_A)
        self.loss_A = tf.reduce_mean(loss_A, name='loss_A')
        # loss_B = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_B, logits=pred_B)
        loss_B = tf.contrib.keras.losses.sparse_categorical_crossentropy(target_B,pred_B)
        self.loss_B = tf.reduce_mean(loss_B, name='loss_B')
        loss = self.loss_A + self.loss_B
        # return loss
        loss_m_A = -(1 - tf.sign(self.tar_in_A)) * tf.log(self.PS_A + 0.0001)
        self.loss_m_A = tf.reduce_mean(loss_m_A, name='loss_m_A')
        loss_m_B = -(1 - tf.sign(self.tar_in_B)) * tf.log(self.PS_B + 0.0001)
        self.loss_m_B = tf.reduce_mean(loss_m_B, name='loss_m_B')
        loss_m = self.loss_m_A + self.loss_m_B
        loss_all = loss + loss_m

        return loss_all

    def optimizer(self, lr, training_steps_per_epoch, lr_decay_factor, min_lr):
        optimizer = tf.train.AdamOptimizer(lr)
        gradients = optimizer.compute_gradients(self.loss)
        capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)
        return train_op, gradients


def random_weight(dim_in, dim_out, name=None):
    return tf.get_variable(dtype=tf.float32, name=name, shape=[dim_in, dim_out], initializer=tf.contrib.layers.xavier_initializer(uniform=False))

def random_bias(dim, name=None):
    return tf.get_variable(dtype=tf.float32, name=name, shape=[dim], initializer=tf.constant_initializer(1.0))

def matrix_mutliply(finstate, W,b,N,h):
    fin_state_new = tf.reshape(tf.matmul(finstate, W) + b, [-1, N, h])
    return fin_state_new

# def get_neighbours1(sessions,lenA,lenB,seqA,seqB,tarA,tarB,neinum):
#     # this function is used to run as a sample to the model.
#     newseqB = []
#     for session in seqB:
#         temp = []
#         for i in session:
#             if i > 0:
#                 temp.append(i + 500)
#             else:
#                 temp.append(0)
#         newseqB.append(temp)
#
#     sa = []
#     for session in seqA:
#         seqa = session[np.where(session > 0)]
#         sa.append(seqa)
#     sb = []
#     for session in seqB:
#         seqb = session[np.where(session > 0)]
#         sb.append([i+500 for i in seqb])
#
#     nei_all = []
#     index_A,index_B = [],[]
#     IsinnumA,IsinnumB = [],[]
#     tarinid = []
#     nei_mask_A,nei_mask_B = [],[]
#     nei_mask_L_A, nei_mask_L_T = [], []
#     tarinA,tarinB = [],[]
#     for i in range(len(sessions)):
#         node = set()
#         itemA = set(sa[i])
#         node.update(itemA)
#         itemB = set(sb[i])
#         node.update(itemB)
#
#         ran = random.randint(0,9)
#         if ran <= 1:
#             node.add(tarA[i])
#             node.add(tarB[i] + 500)
#             tarinid.append(i+10)
#             tarinA.append(1)
#             tarinB.append(1)
#         elif ran >=2 and ran <=4:
#             node.add(tarA[i])
#             tarinid.append(i + 20)
#             tarinA.append(1)
#             tarinB.append(0)
#         elif ran >=5 and ran <=7:
#             node.add(tarB[i]+500)
#             tarinid.append(i+30)
#             tarinA.append(0)
#             tarinB.append(1)
#         else:
#             tarinid.append(i+40)
#             tarinA.append(0)
#             tarinB.append(0)
#
#         pool = set(range(1000)).difference(node)
#         rest = list(random.sample(list(pool),neinum-len(node)))
#         # print(set(rest))
#         node.update(set(rest))
#         # print(node)
#         indx = dict(zip(list(node), range(len(node))))
#         nei_all.append(list(node))
#
#         tempinaset, tempinbset = [], []
#         itemAset = set(range(100))
#         itemBset = set(range(500,600))
#         for ii in indx.keys():
#             if ii in itemAset:
#                 tempinaset.append(1)
#                 tempinbset.append(0)
#             elif ii in itemBset:
#                 tempinaset.append(0)
#                 tempinbset.append(1)
#             else:
#                 tempinaset.append(0)
#                 tempinbset.append(0)
#         IsinnumA.append(tempinaset)
#         IsinnumB.append(tempinbset)
#
#         ini_A = np.ones(neinum)
#         ini_A[np.where(np.array(list(node)) >= 500)] = 0
#         ini_B = np.ones(neinum)
#         ini_B[np.where(np.array(list(node)) < 500)] = 0
#         nei_mask_A.append(ini_A)
#         nei_mask_B.append(ini_B)
#
#         temp_a, temp_t = [],[]
#         for item in sessions[i]:
#             if item < 500:
#                 temp_a.append(ini_A)
#                 temp_t.append(ini_B)
#             else:
#                 temp_a.append(ini_B)
#                 temp_t.append(ini_A)
#         nei_mask_L_A.append(temp_a)
#         nei_mask_L_T.append(temp_t)
#
#
#         zero1 = np.zeros(neinum)
#         ind1 = np.where(np.array(list(node)) <= 99)[0]
#         ent_id = np.array(list(node))[ind1]
#         ite_id = []
#         for ii in ent_id:
#             ite_id.append(ii)
#         zero1[ind1] = ite_id
#         index_A.append(zero1)
#
#         zero2 = np.zeros(neinum)
#         ind2 = []
#         for lll in node:
#             if lll >= 500 and lll < 600:
#                 ind2.append(indx[lll])
#         ent_id = np.array(list(node))[ind2]
#         ite_id = []
#         for ii in ent_id:
#             ite_id.append(ii-500)
#         zero2[ind2] = ite_id
#         index_B.append(zero2)
#
#     nei_all = np.array(nei_all)
#     index = np.arange(len(sessions))
#     index = np.expand_dims(index, axis=-1)
#     index_p = np.repeat(index, neinum, axis=1)
#     index1 = np.stack([index_p, np.array(index_A)], axis=-1)
#     index2 = np.stack([index_p, np.array(index_B)], axis=-1)
#     nei_mask_A = np.array(nei_mask_A)
#     nei_mask_B = np.array(nei_mask_B)
#     nei_mask_L_A = np.array(nei_mask_L_A)
#     nei_mask_L_T = np.array(nei_mask_L_T)
#
#     return np.array(newseqB), nei_all,index1,index2,IsinnumA,IsinnumB,tarinid,nei_mask_A,nei_mask_B,nei_mask_L_A,nei_mask_L_T,tarinA,tarinB

# batch_size = 4
# model = MIFN(num_items_A=100, num_items_B=100,num_entity_A=500,num_entity_B=500,num_cate=100,neighbor_num=100,
#                   batch_size=batch_size, gpu='1') #decay=lr_dc_step * train_data_len / batch_size
# '''
# 这里给出的混合序列是
# A36,B8,A55,B9,A2,B3,B77,(A89),(B16)、
# B19,A1,A45,(A90),(B45)、
# B23,A70,B54,B67,(B56),(A31)，括号里的是预测目标
# 我们需要将该个混合序列里的A和B分开，然后还需要记录A和B里的每个元素的位置
# 以B8的位置为[0,1]为例，0表示它在该batch的第一个样例里（0是该样例的index），1表示它的前面有一个A元素即A36（由于我们会给A序列的开头都添加一个时间步即zero_state,所以1其实就是A36的index）
# 而以A36的位置为[0,0]为例，0表示它在该batch的第一个样例里（0是该样例的index），0表示它的前面没有B元素
# 对于B序列里的padding的元素，我们记录它们前面的A元素数也是0（因为我们想给它们用zero_state）,对于A序列里的padding的元素，我们记录它们前面的B元素数也是0（因为我们想给它们用zero_state）
# '''
# # seq_A = np.array([[36,55,2],[1,45,0],[70,0,0]])
# # seq_B = np.array([[8,9,3,77],[19,0,0,0],[23,54,67,0]])
# # len_A = np.array([3,2,1])
# # len_B = np.array([4,1,3])
# # # pos_A = np.array([[[0,0],[0,1],[0,2]],[[1,1],[1,1],[1,0]],[[2,1],[2,0],[2,0]]])
# # # pos_B = np.array([[[0,1],[0,2],[0,3],[0,3]],[[1,0],[1,0],[1,0],[1,0]],[[2,0],[2,1],[2,1],[2,0]]])
# # index_A = np.array([[[0,0],[0,2],[0,4]],[[1,1],[1,2],[1,0]],[[2,1],[2,0],[2,0]]])
# # index_B = np.array([[[0,1],[0,3],[0,5],[0,6]],[[1,0],[1,0],[1,0],[1,0]],[[2,0],[2,2],[2,3],[2,0]]])
# # target_A = np.array([89,90,31])
# # target_B = np.array([16,45,56])
# # # sequence = np.array([[36,8,55,9,2,3,77],[19,1,45],[23,70,54,67]])
# # sequence = np.array([[36,55,2,8,9,3,77],[1,45,19,0,0,0,0],[70,23,54,67,0,0,0]])
# '''
# 这里给出的混合序列是
# A88,B16,A99,B2,A67,B45,(B44),(A56)、
# B17,B91,A14,A43,(A90),(B73)、
# B44,A34,B87,B72,A11,(A8),(B90)、
# A21,A35,B56,A78,A79,(B11),(A62)，括号里的是预测目标
# '''
# seq_A = np.array([[88,99,67,0],[14,43,0,0],[34,11,0,0],[21,35,78,79]])
# seq_B = np.array([[16,2,45],[17,91,0],[44,87,72],[56,0,0]])
# len_A = np.array([3,2,2,4])
# len_B = np.array([3,2,3,1])
# len_all = np.array([6,4,5,5])
# pos_A = np.array([[[0,0],[0,1],[0,2],[0,2]],[[1,2],[1,2],[1,0],[1,0]],[[2,1],[2,3],[2,0],[2,0]],[[3,0],[3,0],[3,1],[3,1]]])
# pos_B = np.array([[[0,1],[0,2],[0,3]],[[1,0],[1,0],[1,0]],[[2,0],[2,1],[2,1]],[[3,2],[3,0],[3,0]]])
# index_A = np.array([[[0,0],[0,2],[0,4],[0,0]],[[1,2],[1,3],[1,0],[1,0]],[[2,1],[2,4],[2,0],[2,0]],[[3,0],[3,1],[3,3],[3,4]]])
# index_B = np.array([[[0,1],[0,3],[0,5]],[[1,0],[1,1],[1,0]],[[2,0],[2,2],[2,3]],[[3,2],[3,0],[3,0]]])
# target_A = np.array([56,90,8,62])
# target_B = np.array([44,73,90,11])
# sequence = np.array([[88,99,67,16,2,45],[14,43,17,91,0,0],[34,11,44,87,72,0],[21,35,78,79,56,0]])
# adj1_1 = batch_size*[np.random.randint(0,2,(100,100))]
# adj2_1 = batch_size*[np.random.randint(0,2,(100,100))]
# adj3_1 = batch_size*[np.random.randint(0,2,(100,100))]
# adj4_1 = batch_size*[np.random.randint(0,2,(100,100))]
# adj5_1 = batch_size*[np.random.randint(0,2,(100,100))]
# # adj6_1 = batch_size*[np.random.randint(0,2,(100,100))]
# newseqb, neighbors_1,nei_index_A1,nei_index_B1,IsinnumA,IsinnumB,tarinid,\
# nei_mask_A,nei_mask_B,nei_mask_L_A,nei_mask_L_T,tarinA,tarinB = get_neighbours1(sequence,len_A,len_B,seq_A,seq_B,target_A,target_B,100)
# tarinA = np.expand_dims(tarinA,axis=1)
# tarinB = np.expand_dims(tarinB,axis=1)
# print(tarinA)
# print(tarinB)
# print('************************ start training...******************************')
#
# with tf.Session(graph=model.graph,config=model.config) as sess:
#     sess.run(tf.global_variables_initializer())
#     i = 0
#     while i < 100:
#         _, _, l, pa, pb,psa,psb,pga,pgb = sess.run([model.train_op, model.grad, model.loss, model.pred_A, model.pred_B,
#                                         model.PS_A, model.PS_B, model.PG_A, model.PG_B,],
#                                 {model.seq_A:seq_A, model.seq_B:newseqb, model.len_A:len_A, model.len_B:len_B,
#                                  model.target_A:target_A, model.target_B:target_B,
#                                  model.pos_A:pos_A, model.pos_B:pos_B, model.index_A:index_A,model.index_B:index_B,
#                                  model.adj_1:adj1_1,model.adj_2:adj2_1,model.adj_3:adj3_1,model.adj_4:adj4_1,model.adj_5:adj5_1,
#                                  # model.adj_6:adj6_1,
#                                  model.neighbors:neighbors_1, model.nei_index_A:nei_index_A1, model.nei_index_B:nei_index_B1,
#                                  model.IsinnumA:IsinnumA, model.IsinnumB:IsinnumB,
#                                  model.nei_A_mask: nei_mask_A, model.nei_B_mask: nei_mask_B,
#                                  model.nei_L_A_mask:nei_mask_L_A, model.nei_L_T_mask:nei_mask_L_T,
#                                  model.tar_in_A: tarinA, model.tar_in_B: tarinB,})
#
#         print('loss:', l)
#         i += 1