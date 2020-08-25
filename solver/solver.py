from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import math
import os

from solver import graph_ops

FLAGS = tf.app.flags.FLAGS
_NUM_EVAL = 500


def train(model, num_samples, config=None):
    train_op, variables_to_train, global_step = graph_ops.add_training(num_samples)
    summary_op = graph_ops.add_net_summaries(variables_to_train)
    tf.logging.info('Graph created. Training ops added!')
    n_steps = int(FLAGS.max_number_of_epochs * math.ceil(num_samples / FLAGS.batch_size))
    saver = tf.train.Saver()
    model_path = os.path.join(FLAGS.train_dir, 'model')
    num_eval = int(FLAGS.eval_batch_size * math.ceil(_NUM_EVAL/FLAGS.eval_batch_size))
    with tf.Session(config=config) as sess:
        train_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
        init_step = graph_ops.init_sess(sess, global_step)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        tf.logging.info('Session initialized! Begin training step: %d' % init_step)
        for i in range(init_step, n_steps):
            train_feed, _ = model['train_feed_fn'](sess)
            curr_loss = sess.run(train_op, feed_dict=train_feed)
            if i % FLAGS.log_every_n_steps == 0:
                tf.logging.info('Step: %d Loss:%.4f' % (i, curr_loss))
            if i % FLAGS.save_every_n_steps == 0:
                summary = sess.run(summary_op, feed_dict=train_feed)
                train_writer.add_summary(summary, i)
                saver.save(sess, model_path, global_step=i)
                # Evaluate
                eval_obj = model['eval_obj_fn'](num_eval, FLAGS.eval_batch_size)
                for j in range(0, num_eval, FLAGS.eval_batch_size):
                    eval_feed, eval_data = model['eval_feed_fn'](sess)
                    eval_pred_0, eval_pred_1 = sess.run([model['eval_ops'][0], model['eval_ops'][1]], feed_dict=eval_feed)
                    eval_obj.eval_update(j, eval_pred_0, eval_pred_1, eval_data)
                eval_obj.eval()
        # Stop the threads
        coord.request_stop()
        coord.join(threads)


def test(model, num_samples, config=None, save=True):
    with tf.Session(config=config) as sess:
        graph_ops.init_sess(sess)
        sess.run(tf.local_variables_initializer()) # Fix to use num_epochs=1
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        tf.logging.info('Session initialized. Begin testing:')
        save_dir = FLAGS.train_dir if save else None
        eval_obj = model['eval_obj_fn'](num_samples, FLAGS.eval_batch_size, save_dir=save_dir)
        if save:
            save_obj = model['eval_save_fn'](FLAGS.train_dir)
        for i in range(0,num_samples,FLAGS.eval_batch_size):
            if i % 1000 == 0:
                tf.logging.info('Eval image %d:' % i)
            eval_feed, eval_data = model['eval_feed_fn'](sess)
            eval_preds = sess.run(model['eval_ops'], feed_dict=eval_feed)
            eval_obj.eval_update(i, eval_preds[0], eval_preds[1], eval_data)
            if save:
                save_obj.save(i, eval_preds, eval_data)
        eval_obj.eval()
        # Stop the threads
        coord.request_stop()
        coord.join(threads)
