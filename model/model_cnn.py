from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from model import inputs
from model import loss_ops
from model.nets import nets_factory
from model.util import LabelEval, PredictionSave

FLAGS = tf.app.flags.FLAGS


def create_graph(train_dataset, test_dataset, is_training):

    # Ntw func()
    final_endpoint = FLAGS.end_point_cnn_final
    network_fn = nets_factory.get_network_fn(
        FLAGS.model_base,
        final_endpoint=final_endpoint,
        num_classes=train_dataset.num_classes,
        weight_decay=FLAGS.weight_decay)
    image_fn = inputs.image_fn(FLAGS.model_base,
                               network_fn.default_image_size,
                               is_training=False)
    label_fn = inputs.label_fn(FLAGS.loss)

    def _connect_graph_features(slim_dataset, batch_size, is_eval):
        data = inputs.add_data_queue(slim_dataset, batch_size,
                                     tfrecord_dict={
                                            'image': image_fn,
                                            'path': None,
                                            'label': label_fn
                                        },
                                     is_training=is_training)
        _, end_points = network_fn(data['image'],is_training=not is_eval)
        return data, end_points

    def _add_loss(loss_name, end_points, data, num_classes):
        with tf.variable_scope(FLAGS.cnn_logits_scope, values=data.values() + [end_points[final_endpoint]]):
            if loss_name == 'softmax':
                loss = loss_ops.softmax(end_points[final_endpoint], data['label_prob'])
            elif loss_name == 'sigmoid':
                loss = loss_ops.sigmoid(end_points[final_endpoint], data['label'])
            elif loss_name == 'ranking':
                loss = loss_ops.ranking(end_points[final_endpoint], data['label_pair'], num_classes, False)
            elif loss_name == 'warp':
                loss = loss_ops.ranking(end_points[final_endpoint], data['label_pair'], num_classes, True)
            elif loss_name == 'lsep':
                loss = loss_ops.lsep(end_points[final_endpoint], data['label_pair'], num_classes, False)
            return loss

    def _add_prediction(loss_name, end_points):
        with tf.name_scope("cnn_prediction", values=[end_points[final_endpoint]]):
            if loss_name == 'softmax':
                prediction = tf.nn.softmax(end_points[final_endpoint], name='predictions')
            elif loss_name == 'sigmoid':
                prediction = tf.nn.sigmoid(end_points[final_endpoint], name='predictions')
            else:
                prediction = end_points[final_endpoint]
            return prediction

    def _get_train_feed_keys(loss_name):
        feed_keys = ['image']
        if loss_name == 'softmax':
            feed_keys.append('label_prob')
        elif loss_name == 'sigmoid':
            feed_keys.append('label')
        elif loss_name == 'ranking' or loss_name == 'warp' or loss_name == 'lsep':
            feed_keys.append('label_pair')
        return feed_keys

    train_data = None
    # Graph - Training
    if is_training:
        with tf.variable_scope(tf.get_variable_scope()):
            tf.logging.info('Graph-training,Current Scope: %s' % tf.get_variable_scope().name)
            train_data, train_end_points = _connect_graph_features(train_dataset, FLAGS.batch_size, False)
            loss = _add_loss(FLAGS.loss, train_end_points, train_data, train_dataset.num_classes)
            tf.add_to_collection(tf.GraphKeys.LOSSES, loss)

    # Graph - Testing or Validation
    with tf.variable_scope(tf.get_variable_scope(), reuse=is_training):
        tf.logging.info('Graph-testing,Current Scope: %s' % tf.get_variable_scope().name)
        test_data, test_end_points = _connect_graph_features(test_dataset, FLAGS.eval_batch_size, True)
        label_prediction = _add_prediction(FLAGS.loss, test_end_points)

    def _eval_fn(num_eval, batch_size, save_dir=None):
        _file_path = None
        if save_dir:
            _file_path = os.path.join(save_dir,test_dataset.all_reader.get_filename('pred'))
        return LabelEval(num_eval, train_dataset.num_classes,
                           train_dataset.topK, batch_size, _file_path)

    def _save_fn(save_dir):
        _file_path = None
        if save_dir:
            _file_path = os.path.join(save_dir,test_dataset.all_reader.get_filename('eval'))
        return PredictionSave(_file_path, ['label_prediction'])

    return {
        'train_feed_fn': inputs.feed_fn(train_data, _get_train_feed_keys(FLAGS.loss)),
        'eval_ops': [label_prediction],
        'eval_feed_fn': inputs.feed_fn(test_data,['image']),
        'eval_obj_fn': _eval_fn,
        'eval_save_fn': _save_fn
    }