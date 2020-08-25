from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model.preprocessing import preprocessing_factory

import tensorflow as tf
import numpy as np
slim = tf.contrib.slim


def image_fn(preprocessing_name,
               image_size,
               is_training=False):
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name, is_training=is_training)
    def preprocess_image(image):
        image = image_preprocessing_fn(image, image_size, image_size)
        return {
            'image': image}
    return preprocess_image


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


def caption_fn(alllabel_freq, max_len=15, sort=1, _START=1, _END=2, _NULL=0):
    def preprocess_caption(label):
        label = tf.cast(label, tf.float32)
        label_count = tf.cast(tf.reduce_sum(label),tf.int32)
        label_ind = tf.equal(label,1)
        label_idx = tf.reshape(tf.where(label_ind), [-1])
        label_freq = tf.gather(alllabel_freq,label_idx)
        _, idx_sort = tf.nn.top_k(-sort * label_freq,label_count)
        label_sort = tf.gather(label_idx,idx_sort)
        caption = label_sort + 3
        caption = tf.concat([[_START],caption],0)
        caption = tf.concat([caption,[_END]],0)
        caption = tf.pad(caption,[[0,max_len - label_count]],constant_values=_NULL)
        caption.set_shape([max_len+2])
        return {
            'label': label,
            'caption': caption}
    return preprocess_caption


def caption_rand_fn(num_classes, max_len=5):
    def preprocess_caption(label):
        label = tf.cast(label, tf.float32)
        label_count = tf.cast(tf.reduce_sum(label),tf.int32)
        label_ind = tf.equal(label, 1)
        label_ind = tf.reshape(tf.where(label_ind), [-1])
        p = 0.5
        partition_size = tf.maximum(1.0, p * tf.cast(label_count, tf.float32))
        partition_size = tf.cast(partition_size, tf.int32)
        caption = []
        for i in range(max_len):
            label_idx = tf.random_shuffle(label_ind)
            label_idx = label_idx[:partition_size]
            label_idx = tf.reduce_sum(tf.one_hot(label_idx, num_classes), 0)
            caption.append(label_idx)
        caption = tf.stack(caption, axis = 0)
        caption.set_shape([max_len, num_classes])
        return {
            'label': label,
            'caption': caption}
    return preprocess_caption


def label_fn2(num_classes, max_len=5):
    def preprocess_label(label):
        label = tf.cast(label, tf.float32)
        label_count = tf.cast(tf.reduce_sum(label),tf.int32)
        label_ind = tf.equal(label, 1)
        label_ind = tf.reshape(tf.where(label_ind), [-1])
        pad_len = max_len - label_count
        label_idx_pad = tf.pad(label_ind, [[0, pad_len]], constant_values=num_classes)
        label_idx_pad = tf.slice(label_idx_pad, [0], [max_len])
        return {
            'label': label,
            'pos_label': label_idx_pad}
    return preprocess_label


def label_fn(loss_str='sigmoid'):
    def preprocess_label(label):
        label = tf.cast(label, tf.float32)
        label_ret = {
            'label': label
        }
        loss_names = loss_str.split(',')
        if 'ranking' in loss_names or 'warp' in loss_names or 'lsep' in loss_names:
            label_indic = tf.equal(label, 1)
            label_pairs = _lm2lp(label_indic)
            label_ret['label_pair']=label_pairs
        elif 'softmax' in loss_names:
            label_count = tf.reduce_sum(label)
            label_prob = tf.truediv(label, label_count)
            label_ret['label_prob'] = label_prob
        return label_ret
    return preprocess_label


def label_caption_fn(label_fn, caption_fn):
    def preprocess(label):
        label_data = label_fn(label)
        caption_data = caption_fn(label)
        ret_data = label_data
        for key,val in caption_data.items():
            if key not in ret_data:
                ret_data[key] = val
        return ret_data
    return preprocess


def feed_fn(data,feed_keys):
    def feed_data(sess):
        feed_dict = {}
        data_vals = sess.run(data)
        for key in feed_keys:
            feed_dict[data[key]] = data_vals[key]
        return feed_dict, data_vals
    return feed_data


MAX_PAIRS = 1000
def _lm2lp(label_map):
    pos = tf.reshape(tf.where(label_map), [-1])
    neg = tf.reshape(tf.where(tf.logical_not(label_map)), [-1])
    neg_pos = tf.meshgrid(neg, pos, indexing='ij')
    neg_pos_mat = tf.reshape(tf.transpose(tf.stack(neg_pos)), [-1, 2])
    neg_pos_rand = tf.random_shuffle(neg_pos_mat)
    neg_pos_pad = tf.pad(neg_pos_rand, [[0, MAX_PAIRS], [0, 0]]) # In case pairs < max_pairs
    neg_pos_res = tf.slice(neg_pos_pad, [0, 0], [MAX_PAIRS, -1])
    # MAX_PAIRS x 2
    return neg_pos_res