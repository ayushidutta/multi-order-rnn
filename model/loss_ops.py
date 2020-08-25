from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np


def _batch_gather(input, indices):
  """
  output[i, ..., j] = input[i, indices[i, ..., j]]
  """
  shape_output = indices.get_shape().as_list()
  shape_input = input.get_shape().as_list()
  assert len(shape_input) == 2
  batch_base = shape_input[1] * np.arange(shape_input[0])
  batch_base_shape = [1] * len(shape_output)
  batch_base_shape[0] = shape_input[0]

  batch_base = batch_base.reshape(batch_base_shape)
  indices = batch_base + indices

  input = tf.reshape(input, [-1])
  return tf.gather(input, indices)


def _get_M(num_classes):
  alpha = [1./(i+1) for i in range(num_classes)]
  alpha = np.cumsum(alpha)
  return alpha.astype(np.float32)


def _get_ranking_weights(logits, num_classes):
    _, indices = tf.nn.top_k(tf.stop_gradient(logits), num_classes)
    _, ranks = tf.nn.top_k(-indices, num_classes)
    rank_weights = _get_M(num_classes)
    label_weights = tf.gather(rank_weights, ranks)
    return label_weights


def _pairwise(label_pairs, logits, num_classes):
  mapped = _batch_gather(logits, label_pairs)
  neg, pos = tf.split(mapped, 2, 2)  # Note-Split does not reduce rank.
  delta = neg - pos

  neg_idx, pos_idx = tf.split(label_pairs, 2, 2)
  _, indices = tf.nn.top_k(tf.stop_gradient(logits), num_classes)
  _, ranks = tf.nn.top_k(-indices, num_classes)
  pos_ranks = _batch_gather(ranks, pos_idx)

  weights = _get_M(num_classes)
  pos_weights = tf.gather(weights, pos_ranks)

  delta_nnz = tf.cast(tf.not_equal(neg_idx, pos_idx), tf.float32)
  return delta, delta_nnz, pos_weights


def sigmoid(logits, labels, reduction=True):
    tf.logging.info('Logistic Loss')
    loss = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=logits, labels=labels)
    if reduction:
        reduction = tf.reduce_sum(loss, 1)
        loss = tf.reduce_mean(reduction, name='sigmoid')
    return loss


def softmax(logits, labels):
    tf.logging.info('Softmax Loss')
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=labels)
    loss = tf.reduce_mean(cross_entropy, name='softmax')
    return loss


def log_loss(predictions, labels, epsilon=1e-7, reduction=True):
    tf.logging.info('Log Loss')
    loss = -tf.multiply(labels, tf.log(predictions + epsilon)) - tf.multiply(
        (1 - labels), tf.log(1 - predictions + epsilon))
    if reduction:
        reduction = tf.reduce_sum(loss, 1)
        loss = tf.reduce_mean(reduction, name='log_loss')
    return loss


def log_loss2(predictions, labels, epsilon=1e-7, reduction=True):
    tf.logging.info('Log Loss')
    loss = -tf.log(predictions * (labels-0.5) + 0.5)
    if reduction:
        reduction = tf.reduce_sum(loss, 1)
        loss = tf.reduce_mean(reduction, name='log_loss')
    return loss


margin = 1.0
def ranking(logits, labels, num_classes, weighted_pairs=False):
    tf.logging.info('Ranking Loss, weighted= %s' % weighted_pairs)
    delta, delta_nnz, pos_weights = _pairwise(labels, logits, num_classes)
    delta = tf.nn.relu(margin + delta)
    delta *= delta_nnz
    pairs_per_sample = tf.reduce_sum(delta_nnz,1)
    max_pairs = tf.reduce_max(pairs_per_sample)
    w_sample = tf.truediv(max_pairs,pairs_per_sample)
    if weighted_pairs:
        delta *= pos_weights
    reduction = tf.reduce_sum(delta, 1)
    reduction_2 = reduction * w_sample
    w_sum = tf.reduce_sum(w_sample)
    loss = tf.truediv(tf.reduce_sum(reduction_2),w_sum)
    #loss = tf.reduce_mean(reduction, name='ranking')
    return loss


def lstm_warp(logits, labels, max_logits, partition_mask, T):
    tf.logging.info('LSTM WARP Loss')
    num_classes = tf.shape(logits)[1]
    labels_not_mask = (1 - labels) * -99
    max_logits_1 = (tf.stop_gradient(max_logits) * -labels) + labels_not_mask
    _, indices = tf.nn.top_k(tf.stop_gradient(max_logits_1), T)
    pos_indices = indices * partition_mask
    caption = tf.cast(tf.reduce_sum(tf.one_hot(pos_indices, num_classes), 1), tf.float32)
    caption = caption * labels
    caption = tf.Print(caption,[caption[0], labels[0]], message='Caption', first_n=5, summarize=81)
    label_loss_mask = tf.cast(tf.not_equal(labels - caption, 1),tf.float32)
    label_loss_mask = tf.Print(label_loss_mask, [label_loss_mask[0], caption[0]], message='Caption mask', first_n=5, summarize=81)
    loss = tf.reduce_sum(
        label_loss_mask * tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=caption), 1)
    return loss


def lsep(logits, labels, num_classes, weighted_pairs):
    tf.logging.info('LSEP Loss')
    # compute label pairs
    # batch_size x num_pairs x 2
    delta, delta_nnz, pos_weights = _pairwise(labels, logits, num_classes)
    delta_max = tf.reduce_max(delta, 1, keep_dims=True)
    delta_max_nnz = tf.nn.relu(delta_max)
    inner_exp_diff = tf.exp(delta - delta_max_nnz)
    inner_exp_diff *= delta_nnz
    if weighted_pairs:
        inner_exp_diff *= pos_weights
    inner_sum = tf.reduce_sum(inner_exp_diff, 1, keep_dims=True)
    reduction = delta_max_nnz + tf.log(tf.exp(-delta_max_nnz) + inner_sum)
    loss = tf.reduce_mean(reduction, name='lsep')
    return loss


def sigmoid_relu(logits, labels, reduction=True):
    prob = tf.nn.sigmoid(logits)
    scores_diff = tf.abs(labels - prob)
    loss = tf.nn.relu(scores_diff - 0.5)
    if reduction:
        reduction = tf.reduce_sum(loss, 1)
        loss = tf.reduce_mean(reduction, name='sigmoid_relu')
    return loss


def mse(logits, labels, reduction=True):
    tf.logging.info('MSE Loss')
    loss = tf.squared_difference(labels, logits)
    if reduction:
        reduction = tf.reduce_sum(loss, 1)
        loss = tf.reduce_mean(reduction, name='mse')
    return loss