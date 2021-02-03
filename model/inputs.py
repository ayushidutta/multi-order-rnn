from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
slim = tf.contrib.slim

def add_data_queue(dataset,
                   batch_size,
                   tfrecord_dict,
                   is_training=True,
                   dynamic_pad=False):
    if is_training:
        allow_smaller_final_batch = False
        shuffle = True
        num_epochs = None
    else:
        allow_smaller_final_batch = True
        shuffle = False
        num_epochs = 1
    with tf.variable_scope('inputs',[dataset]):
        provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            num_readers=1,
            common_queue_capacity=6 * batch_size,  # from 20
            common_queue_min=2 * batch_size,  # from 10
            shuffle=shuffle,
            num_epochs=num_epochs)
        tfrecord = provider.get(tfrecord_dict.keys())
        item = _process_data(tfrecord_dict, tfrecord)
        data_batch = tf.train.batch(
            item,
            batch_size=batch_size,
            num_threads=1,
            capacity=4*batch_size,  # from 5
            dynamic_pad=dynamic_pad,
            allow_smaller_final_batch=allow_smaller_final_batch)
        if is_training:
            batch_queue = slim.prefetch_queue.prefetch_queue(
                data_batch, capacity=2)
            return batch_queue.dequeue()
        else:
            return data_batch

def _process_data(tfrecord_dict, tfrecord):
    data_item = {}
    for idx, tfrecord_key in enumerate(tfrecord_dict.keys()):
        data_in = tfrecord[idx]
        preprocess_fn = tfrecord_dict[tfrecord_key]
        if preprocess_fn is not None:
            data_out = preprocess_fn(data_in)
            for data_key, data_val in data_out.items():
                data_item[data_key] = data_val
        else:
            data_item[tfrecord_key] = data_in
    return data_item

def label_fn(loss_str='sigmoid'):
    def preprocess_label(label):
        label = tf.cast(label, tf.float32)
        label_ret = {
            'label': label
        }
        loss_names = loss_str.split(',')
        if 'softmax' in loss_names:
            label_count = tf.reduce_sum(label)
            label_prob = tf.truediv(label, label_count)
            label_ret['label_prob'] = label_prob
        return label_ret
    return preprocess_label

def feed_fn(data,feed_keys):
    def feed_data(sess):
        feed_dict = {}
        data_vals = sess.run(data)
        for key in feed_keys:
            feed_dict[data[key]] = data_vals[key]
        return feed_dict, data_vals
    return feed_data