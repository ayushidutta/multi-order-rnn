from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np


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