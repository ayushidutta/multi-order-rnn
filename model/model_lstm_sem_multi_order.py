from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import os

from model import inputs
from model.lstm.sem_multi_order import CaptionGenerator
from model.util import PredictionSave, LabelEval

FLAGS = tf.app.flags.FLAGS
_MAX_CAPTION_LEN = 15


def create_graph(train_dataset, test_dataset, is_training):

    # Datasets and properties
    train_annotations = train_dataset.all_reader.load_annotations()
    max_len = int(np.max(np.sum(train_annotations,1)))
    #max_len = max_len if is_training else _MAX_CAPTION_LEN
    max_len = FLAGS.time_step
    tf.logging.info('Max len of caption: %d' % max_len)

    # Ntw func()
    model = CaptionGenerator(train_dataset.num_classes, dim_embed=FLAGS.dim_embed, dim_hidden=FLAGS.dim_hidden,
                             n_time_step=FLAGS.time_step, prev2out=FLAGS.prev2out, ctx2out=FLAGS.ctx2out, dropout=False)
    label_fn = inputs.label_fn(FLAGS.loss + ',' + FLAGS.lstm_loss)
    tf_prob_key = 'sig_prob'

    def _connect_graph_features(slim_dataset, batch_size):
        data = inputs.add_data_queue(slim_dataset, batch_size,
                                     tfrecord_dict={
                                            tf_prob_key: None,
                                            'label': label_fn,
                                            'path': None,
                                            'feature': None
                                        },
                                     is_training=is_training)
        return data

    def _get_lstm_label_data(loss_name, data):
        if loss_name == 'ranking':
            return data['label_pair']
        else:
            return data['label']

    def _get_train_feed_keys(loss_names):
        feed_keys = [tf_prob_key, 'label', 'feature']
        if 'softmax' in loss_names:
            feed_keys.append('label_prob')
        elif 'ranking' in loss_names or 'warp' in loss_names or 'lsep' in loss_names:
            feed_keys.append('label_pair')
        return feed_keys
    train_data = None

    # Graph - Training
    if is_training:
        with tf.variable_scope(tf.get_variable_scope()):
            tf.logging.info('Graph-training,Current Scope: %s' % tf.get_variable_scope().name)
            train_data = _connect_graph_features(train_dataset, FLAGS.batch_size)
            caption_loss, _ = model.build_model(train_data[tf_prob_key],
                                                _get_lstm_label_data(FLAGS.lstm_loss, train_data))
            tf.add_to_collection(tf.GraphKeys.LOSSES, caption_loss)


    # Graph - Testing or Validation
    with tf.variable_scope(tf.get_variable_scope(), reuse=is_training):
        tf.logging.info('Graph-testing,Current Scope: %s' % tf.get_variable_scope().name)
        test_data = _connect_graph_features(test_dataset, FLAGS.eval_batch_size)
        prediction_scores, gen_captions, pred_time = model.build_sampler(test_data[tf_prob_key],
                                                              max_len,
                                                              loss='sigmoid')
    def _eval_fn(num_eval, batch_size, save_dir=None):
        _file_path = None
        if save_dir:
            _file_path = os.path.join(save_dir,test_dataset.all_reader.get_filename('pred'))
        return LabelEval(num_eval, train_dataset.num_classes,
                         max_len, train_dataset.topK, batch_size, _file_path)

    def _save_fn(save_dir):
        _file_path = None
        if save_dir:
            _file_path = os.path.join(save_dir,test_dataset.all_reader.get_filename('eval'))
        return PredictionSave(_file_path, ['prediction_scores', 'gen_captions', 'pred_time'])

    return {
        'train_feed_fn': inputs.feed_fn(train_data, _get_train_feed_keys([FLAGS.loss, FLAGS.lstm_loss])),
        'eval_ops': [prediction_scores, gen_captions, pred_time],
        'eval_feed_fn': inputs.feed_fn(test_data,[tf_prob_key]),
        'eval_obj_fn': _eval_fn,
        'eval_save_fn': _save_fn
    }