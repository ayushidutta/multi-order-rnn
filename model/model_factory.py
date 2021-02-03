from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model import model_lstm_sem_multi_order

model_map = {
    'lstm_sem_multi_order': model_lstm_sem_multi_order,
}

def create_graph(name, train_dataset, test_dataset, is_training):
  print(name)
  if name not in model_map:
    raise ValueError('Name of model unknown %s' % name)
  return model_map[name].create_graph(train_dataset, test_dataset, is_training)