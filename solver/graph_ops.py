from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
slim = tf.contrib.slim


def _configure_learning_rate(num_samples_per_epoch, global_step):
    """Configures the learning rate.

  Args:
    num_samples_per_epoch: The number of samples in each epoch of training.
    global_step: The global_step tensor.

  Returns:
    A `Tensor` representing the learning rate.

  Raises:
    ValueError: if
  """
    decay_steps = int(num_samples_per_epoch / FLAGS.batch_size *
                      FLAGS.num_epochs_per_decay)

    if FLAGS.learning_rate_decay_type == 'exponential':
        return tf.train.exponential_decay(FLAGS.learning_rate,
                                          global_step,
                                          decay_steps,
                                          FLAGS.learning_rate_decay_factor,
                                          staircase=True,
                                          name='exponential_decay_learning_rate')
    elif FLAGS.learning_rate_decay_type == 'fixed':
        return tf.constant(FLAGS.learning_rate, name='fixed_learning_rate')
    elif FLAGS.learning_rate_decay_type == 'polynomial':
        return tf.train.polynomial_decay(FLAGS.learning_rate,
                                         global_step,
                                         decay_steps,
                                         FLAGS.end_learning_rate,
                                         power=1.0,
                                         cycle=False,
                                         name='polynomial_decay_learning_rate')
    else:
        raise ValueError('learning_rate_decay_type [%s] was not recognized',
                         FLAGS.learning_rate_decay_type)


def _configure_optimizer(learning_rate):
    """Configures the optimizer used for training.

  Args:
    learning_rate: A scalar or `Tensor` learning rate.

  Returns:
    An instance of an optimizer.

  Raises:
    ValueError: if FLAGS.optimizer is not recognized.
  """
    if FLAGS.optimizer == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(
            learning_rate,
            rho=FLAGS.adadelta_rho,
            epsilon=FLAGS.opt_epsilon)
    elif FLAGS.optimizer == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(
            learning_rate,
            initial_accumulator_value=FLAGS.adagrad_initial_accumulator_value)
    elif FLAGS.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(
            learning_rate,
            beta1=FLAGS.adam_beta1,
            beta2=FLAGS.adam_beta2,
            epsilon=FLAGS.opt_epsilon)
    elif FLAGS.optimizer == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(
            learning_rate,
            learning_rate_power=FLAGS.ftrl_learning_rate_power,
            initial_accumulator_value=FLAGS.ftrl_initial_accumulator_value,
            l1_regularization_strength=FLAGS.ftrl_l1,
            l2_regularization_strength=FLAGS.ftrl_l2)
    elif FLAGS.optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(
            learning_rate,
            momentum=FLAGS.momentum,
            name='Momentum')
    elif FLAGS.optimizer == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate,
            decay=FLAGS.rmsprop_decay,
            momentum=FLAGS.momentum,
            epsilon=FLAGS.opt_epsilon)
    elif FLAGS.optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    else:
        raise ValueError('Optimizer [%s] was not recognized', FLAGS.optimizer)
    return optimizer


def _get_loss_gradients(optimizer,
                        **kwargs):
    all_losses = []
    losses = _get_collection(tf.GraphKeys.LOSSES, FLAGS.trainable_scopes)
    regularization_losses = _get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, FLAGS.trainable_scopes)
    if losses:
        sum_loss = tf.add_n(losses)
        all_losses.append(sum_loss)
        tf.summary.scalar('sum_loss', sum_loss)
        for loss in losses:
            tf.summary.scalar('losses/%s' % loss.op.name, loss)
    if regularization_losses:
        regularization_loss = tf.add_n(regularization_losses,
                                       name='reg_loss')
        all_losses.append(regularization_loss)
        tf.summary.scalar('reg_loss', regularization_loss)
        for loss in regularization_losses:
            tf.summary.scalar('reg_losses/%s' % loss.op.name, loss)
    total_loss = tf.add_n(all_losses)
    grads_and_vars = optimizer.compute_gradients(total_loss, **kwargs)
    tf.summary.scalar('total_loss', total_loss)
    return total_loss, grads_and_vars


def _get_variables_to_train():
    """Returns a list of variables to train.

  Returns:
    A list of variables to train by the optimizer.
  """
    if FLAGS.trainable_scopes is None:
        return tf.trainable_variables()
    else:
        scopes = [scope.strip() for scope in FLAGS.trainable_scopes.split(',')]
    variables_to_train = []
    for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.extend(variables)
    return variables_to_train


def _get_collection(collection, scopes=None, exclusions=None):
    """Returns a list of variables/ops from the collection.

  Returns:
    Returns a list of variables/ops from the collection.
  """
    tf.logging.info('Collection:%s, All: %s' % (collection, tf.get_collection(collection)))
    # Inclusions
    if scopes is None:
        _var_list = tf.get_collection(collection)
    else:
        scopes = [scope.strip() for scope in scopes.split(',')]
        _var_list = []
        for scope in scopes:
            var = tf.get_collection(collection, scope)
            _var_list.extend(var)
    # Exclusions
    if exclusions:
        var_list = []
        exclusions = [scope.strip() for scope in exclusions.split(',')]
        for var in _var_list:
            excluded = False
            for exclusion in exclusions:
                if var.op.name.startswith(exclusion):
                    excluded = True
                    break
            if not excluded:
                var_list.append(var)
    else:
        var_list = _var_list
    tf.logging.info('Collection:%s, Selected %s' % (collection,var_list))
    return var_list


def add_net_summaries(variables_to_train):
    """
      Adds summaries to the graph
  """
    for var in variables_to_train:
        tf.summary.histogram(var.op.name, var)
    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
    summary_op = tf.summary.merge(list(summaries), name='summary_op')
    return summary_op


def add_training(num_samples):
    """
     Defines the classifier training with optimizer, summaries etc..
  """
    with tf.name_scope('optimizer'):
        global_step = tf.train.get_or_create_global_step()
        update_ops = _get_collection(tf.GraphKeys.UPDATE_OPS, FLAGS.trainable_scopes)

        # TODO: # Configure the moving averages
        # if FLAGS.moving_average_decay:
        #     moving_average_variables = slim.get_model_variables()
        #     variable_averages = tf.train.ExponentialMovingAverage(
        #         FLAGS.moving_average_decay, global_step)
        #     update_ops.append(variable_averages.apply(moving_average_variables))

        # Configure the optimization procedure.

        learning_rate = _configure_learning_rate(num_samples, global_step)
        optimizer = _configure_optimizer(learning_rate)
        tf.summary.scalar('learning_rate', learning_rate)

        # Variables to train, Gradient updates
        variables_to_train = _get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, FLAGS.trainable_scopes)
        total_loss, gradients = _get_loss_gradients(optimizer,
                                                    var_list=variables_to_train)
        grad_updates = optimizer.apply_gradients(gradients,
                                                 global_step=global_step)
        update_ops.append(grad_updates)
        update_op = tf.group(*update_ops)
        with tf.control_dependencies([update_op]):
            train_tensor = tf.identity(total_loss, name='train_op')
    return train_tensor, variables_to_train, global_step


def init_sess(sess, global_step=None):
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    var_to_restore = None
    tf.logging.info('Init Session!')
    # Latest Checkpoint: Restore all variables
    if tf.train.latest_checkpoint(FLAGS.train_dir):
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.train_dir)
        var_to_restore = tf.global_variables()
        tf.logging.info('Var restored from latest ckpt: %s' % var_to_restore)
    # Pretrained graph. Restore only Model Variables
    elif FLAGS.checkpoint_path != None:
        if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
            checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
        else:
            checkpoint_path = FLAGS.checkpoint_path
        tf.logging.info('Checkpoint: %s' % checkpoint_path)
        var_to_restore = _get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, FLAGS.checkpoint_scopes, FLAGS.checkpoint_exclude_scopes)
    if var_to_restore != None:
        tf.logging.info('Restoring Variables!')
        saver = tf.train.Saver(var_to_restore)
        saver.restore(sess, checkpoint_path)
    if global_step:
        return sess.run(global_step)
