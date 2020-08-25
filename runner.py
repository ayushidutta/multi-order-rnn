from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from config import FLAGS
from data_utils import dataset_factory
from model import model_factory
from solver import solver


def _train():
    with tf.Graph().as_default():
        train_dataset = dataset_factory.get_dataset(
            FLAGS.dataset_name, 'train', FLAGS.dataset_dir)
        test_dataset = dataset_factory.get_dataset(
            FLAGS.dataset_name, 'valid', FLAGS.dataset_dir)
        tf.logging.info('Model: %s' % FLAGS.model)
        graph = model_factory.create_graph(FLAGS.model, train_dataset, test_dataset, True)
        #config = tf.ConfigProto(log_device_placement=True)
        solver.train(graph, train_dataset.num_samples)


def _test(save):
    with tf.Graph().as_default():
        train_dataset = dataset_factory.get_dataset(
            FLAGS.dataset_name, 'train', FLAGS.dataset_dir)
        test_dataset = dataset_factory.get_dataset(
            FLAGS.dataset_name, 'test', FLAGS.dataset_dir)
        graph = model_factory.create_graph(FLAGS.model, train_dataset, test_dataset, False)
        solver.test(graph, test_dataset.num_samples, save=save)


def main(_):
    if FLAGS.run_opt == 'train':
        _train()
    elif FLAGS.run_opt == 'test':
        _test(True)
    elif FLAGS.run_opt == 'validate':
        _test(False)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()