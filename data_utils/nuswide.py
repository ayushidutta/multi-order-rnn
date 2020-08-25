# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Provides data for the flowers dataset.

The dataset scripts used to create the dataset can be found at:
tensorflow/models/slim/datasets/download_and_convert_flowers.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf

from data_utils import dataset_utils

slim = tf.contrib.slim

_DS_NAME = 'nus1'
_FILE_PATTERN = 'nus1_%s_*.tfrecord'
_SPLITS_TO_SIZES = {'train': 125449, 'test': 83898, 'valid': 600}
_NUM_CLASSES = 81
_TOP_K = 3
_TFRECORD_DIR = '/home/ayushi/Lab/imageAnnotation/data/nuswide'
_TFRECORD_SUB_DIRECTORY='caffe-res1-101/tfrecord'
_FTR_DIM = 2048

class DatasetReaderFn(object):

    def __init__(self, dataset_dir, split_name):
        self.dataset_dir = dataset_dir
        if split_name not in _SPLITS_TO_SIZES:
            raise ValueError('split name %s was not recognized.' % split_name)
        self.split_name = split_name

    def load_annotations(self):
        file_annotations = _DS_NAME+'_'+self.split_name+'_annot.txt'
        file_annotations = os.path.join(self.dataset_dir,file_annotations)
        with open(file_annotations) as fid_image_annotations:
            img_annot_lines = fid_image_annotations.read().splitlines()
        n_imgs = len(img_annot_lines)
        n_labels = len(img_annot_lines[0].split(' '))
        tf.logging.info('No. of imgs:%d No. of labels:%d' % (n_imgs, n_labels))
        img_annots = np.zeros([n_imgs, n_labels])
        for idx, annot_line in enumerate(img_annot_lines):
            annot_val = [float(val) for val in annot_line.split(' ')]
            img_annots[idx, :] = annot_val
        return img_annots

    def get_filename(self, suffix):
        return _DS_NAME+'_'+self.split_name+'_' + suffix

def get_slim(dataset_dir, split_name, file_pattern=None, reader=None):
    """Gets a dataset tuple with instructions for reading flowers.

      Args:
        file_pattern: The file pattern to use when matching the dataset sources.
          It is assumed that the pattern contains a '%s' string so that the split
          name can be inserted.
        reader: The TensorFlow reader type.

      Returns:
        A `Dataset` namedtuple.

      Raises:
        ValueError: if `split_name` is not a valid train/validation split.
    """

    if not file_pattern:
        file_pattern = _FILE_PATTERN
        file_pattern = os.path.join(_TFRECORD_DIR, _TFRECORD_SUB_DIRECTORY, file_pattern % split_name)

    # Allowing None in the signature so that dataset_factory can use the default.
    if reader is None:
        reader = tf.TFRecordReader

    items_to_desc = {
        'image': 'A color image of varying size.',
        'label': 'Multiple integers between 0 and 81',
        'path': 'Image path',
        'sig_prob': 'Sigmoid Probability',
        'feature': 'CNN feature'
    }

    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
        'image/path': tf.FixedLenFeature([], dtype=tf.string),
        'image/class/label': tf.FixedLenFeature([_NUM_CLASSES], tf.int64,
                                                default_value=tf.zeros([_NUM_CLASSES], dtype=tf.int64)),
        'image/class/sig_prob': tf.FixedLenFeature([_NUM_CLASSES], tf.float32,
                                                default_value=tf.zeros([_NUM_CLASSES], dtype=tf.float32)),
        'image/feature': tf.FixedLenFeature([_FTR_DIM], tf.float32,
                                                   default_value=tf.zeros([_FTR_DIM], dtype=tf.float32))
    }

    items_to_handlers = {
        'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
        'label': slim.tfexample_decoder.Tensor('image/class/label'),
        'path': slim.tfexample_decoder.Tensor('image/path'),
        'sig_prob': slim.tfexample_decoder.Tensor('image/class/sig_prob'),
        'feature': slim.tfexample_decoder.Tensor('image/feature')
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)

    labels_to_names = None
    if dataset_utils.has_labels(dataset_dir):
        labels_to_names = dataset_utils.read_label_file(dataset_dir)

    return slim.dataset.Dataset(
          data_sources=file_pattern,
          reader=reader,
          decoder=decoder,
          num_samples=_SPLITS_TO_SIZES[split_name],
          items_to_descriptions=items_to_desc,
          num_classes=_NUM_CLASSES,
          topK=_TOP_K,
          labels_to_names=labels_to_names,
          all_reader=DatasetReaderFn(dataset_dir,split_name))