from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from model import loss_ops

_START = 0

class CaptionGenerator(object):
    def __init__(self, num_classes, dim_embed=512, dim_hidden=1024, n_time_step=5, prev2out=True,
                 ctx2out=True, dropout=False):
        """
        Args:
            word_to_idx: word-to-index mapping dictionary.
            dim_embed: (optional) Dimension of word embedding.
            dim_hidden: (optional) Dimension of all hidden state.
            n_time_step: (optional) Time step size of LSTM.
            prev2out: (optional) previously generated word to hidden state. (see Eq (7) for explanation)
            ctx2out: (optional) context to hidden state (see Eq (7) for explanation)
            alpha_c: (optional) Doubly stochastic regularization coefficient. (see Section (4.2.1) for explanation)
            selector: (optional) gating scalar for context vector. (see Section (4.2.1) for explanation)
            dropout: (optional) If true then dropout layer is added.
        """
        self.prev2out = prev2out
        self.ctx2out = ctx2out
        self.dropout = dropout
        self.V = num_classes + 1
        self.M = dim_embed
        self.H = dim_hidden
        self.T = n_time_step
        self.C = num_classes
        self._start = _START
        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer(0.0)
        self.emb_initializer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)
        self.scope = 'cnn_rnn_att'

    def _get_initial_lstm(self, init_prob):
        with tf.variable_scope('initial_lstm'):

            w_ph = tf.get_variable('w_ph', [self.C, self.H], initializer=self.weight_initializer)
            b_ph = tf.get_variable('b_ph', [self.H], initializer=self.const_initializer)
            h = tf.matmul(init_prob, w_ph) + b_ph

            w_pc = tf.get_variable('w_pc', [self.C, self.H], initializer=self.weight_initializer)
            b_pc = tf.get_variable('b_pc', [self.H], initializer=self.const_initializer)
            c = tf.matmul(init_prob, w_pc) + b_pc

            h = tf.nn.tanh(h)
            c = tf.nn.tanh(c)
            return c, h

    def _word_embedding(self, inputs, reuse=False):
        with tf.variable_scope('word_embedding', reuse=reuse):
            w = tf.get_variable('w', [self.V, self.M], initializer=self.emb_initializer)
            x = tf.nn.embedding_lookup(w, inputs, name='word_vector')  # (N, T, M) or (N, M)
            return x

    def _decode_lstm(self, x, h, dropout=False, reuse=False):
        with tf.variable_scope('logits', reuse=reuse):
            w_h = tf.get_variable('w_h', [self.H, self.M], initializer=self.weight_initializer)
            b_h = tf.get_variable('b_h', [self.M], initializer=self.const_initializer)
            w_out = tf.get_variable('w_out', [self.M, self.C], initializer=self.weight_initializer)
            b_out = tf.get_variable('b_out', [self.C], initializer=self.const_initializer)

            if dropout:
                h = tf.nn.dropout(h, 0.8)
            h_logits = tf.matmul(h, w_h) + b_h

            if self.prev2out:
                h_logits += x
            h_logits = tf.nn.tanh(h_logits)

            if dropout:
                h_logits = tf.nn.dropout(h_logits, 0.8)
            out_logits = tf.matmul(h_logits, w_out) + b_out
            return out_logits

    def _batch_norm(self, x, mode=True, name=None):
        return tf.contrib.layers.batch_norm(inputs=x,
                                            decay=0.95,
                                            center=True,
                                            scale=True,
                                            is_training=mode,
                                            #updates_collections=None,
                                            scope=(name + '_batch_norm'))

    def build_model(self, init_prob, labels):

        with tf.variable_scope(self.scope, [init_prob, labels]):
            batch_size = tf.shape(init_prob)[0]

            # batch normalize feature vectors
            init_prob = self._batch_norm(init_prob, mode=True, name='init_prob')
            c, h = self._get_initial_lstm(init_prob=init_prob)

            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.H)
            time_reg = 0.0
            prob_reg = 0.0
            time_reg_mask = tf.ones_like(labels)
            caption_len = tf.cast(tf.reduce_sum(labels, 1), tf.int32)

            for t in range(self.T):
                mask = tf.expand_dims(tf.cast(tf.less(t, caption_len), tf.float32),1)

                if t == 0:
                    x = self._word_embedding(inputs=tf.fill([batch_size], self._start))
                else:
                    x = self._word_embedding(inputs=sampled_word + 1, reuse=True) #Offset for start
                    #prev_logits = curr_logits

                with tf.variable_scope('lstm', reuse=(t != 0)):
                    _, (c, h) = lstm_cell(inputs=x, state=[c, h])

                curr_logits = self._decode_lstm(x, h, dropout=self.dropout, reuse=(t != 0))

                if t == 0:
                    logits = curr_logits
                else:
                    logits = tf.maximum(logits, curr_logits)
                    time_reg_mask_1 = tf.cast(tf.not_equal(tf.one_hot(tf.stop_gradient(sampled_word),self.C),1), tf.float32)
                    time_reg_mask = time_reg_mask * time_reg_mask_1
                    #prob_reg += mask * time_reg_mask * labels * tf.nn.relu(0.1 + prev_logits - curr_logits)

                sampled_word = tf.argmax(curr_logits, 1)
                time_reg += mask * (loss_ops.sigmoid(curr_logits, time_reg_mask * labels, reduction=False))

            lstm_loss = tf.reduce_mean(tf.reduce_sum(time_reg, 1))

        return lstm_loss, logits

    def build_sampler(self, init_prob, max_len, loss='sigmoid'):

        with tf.variable_scope(self.scope, [init_prob]):
            batch_size = tf.shape(init_prob)[0]

            # batch normalize feature vectors
            init_prob = self._batch_norm(init_prob, mode=False, name='init_prob')
            c, h = self._get_initial_lstm(init_prob=init_prob)

            sampled_word_list = []
            pred_time_list = []
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.H)

            time_reg_mask = tf.ones([batch_size, self.C], tf.float32)
            time_reg_not_mask = tf.zeros([batch_size, self.C], tf.float32)

            for t in range(max_len):
                if t == 0:
                    x = self._word_embedding(inputs=tf.fill([tf.shape(init_prob)[0]], self._start))
                else:
                    x = self._word_embedding(inputs=sampled_word+1, reuse=True) # Offset for start

                with tf.variable_scope('lstm', reuse=(t != 0)):
                    _, (c, h) = lstm_cell(inputs=x, state=[c, h])

                curr_logits = self._decode_lstm(x, h, reuse=(t != 0))
                pred_time_list.append(tf.nn.sigmoid(curr_logits))

                if t == 0:
                    logits = curr_logits
                else:
                    logits = tf.maximum(logits, curr_logits)
                    time_reg_mask_1 = tf.not_equal(tf.one_hot(sampled_word, self.C), 1)
                    time_reg_mask = time_reg_mask * tf.cast(time_reg_mask_1, tf.float32)
                    time_reg_not_mask = (1 - time_reg_mask) * -9

                curr_logits_1 = (curr_logits * time_reg_mask) + time_reg_not_mask
                sampled_word = tf.argmax(curr_logits_1, 1)
                sampled_word_list.append(sampled_word)

            sampled_captions = tf.transpose(tf.stack(sampled_word_list), (1, 0))  # (N, max_len)
            pred_time = tf.stack(pred_time_list, axis=1)

            if loss == 'sigmoid':
                prediction = tf.nn.sigmoid(logits)
            else:
                prediction = logits

        return prediction, sampled_captions, pred_time