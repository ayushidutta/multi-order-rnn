r"""Converts NUSWIDE data to TFRecords of TF-Example protos.

This module reads the files that make up the NUSWIDE data and creates two
TFRecord datasets: one for train and one for test. Each TFRecord dataset
is comprised of a set of TF-Example protocol buffers, each of which contain
a single image and label.

The script should take about a minute to run.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import tensorflow as tf
import numpy as np
import h5py

from PIL import Image
from data_utils import dataset_utils

_DATASET_DIR = '/home/ayushi/Git/research/imageAnnotation/data/coco'
_SPLIT_NAMES = ['train','test','valid']
_DATA_FILENAMES = ['coco_train_list.txt','coco_test_list.txt','coco_test_list.txt']
_LABEL_FILENAMES = ['coco_train_annot.txt','coco_test_annot.txt','coco_test_annot.txt']
_MODEL_DIR = os.path.join(_DATASET_DIR,'net-res1-101')
_RESULTS_DIR = os.path.join(_MODEL_DIR,'sigmoid_logits')
_FTR_FILENAMES = ['coco_train_r101.mat','coco_test_r101.mat','coco_test_r101.mat']
_PROB_FILENAMES = ['coco_train_r101_pred.mat','coco_test_r101_pred.mat','coco_test_r101_pred.mat']
_NUM_PER_SHARD = 50000
_IMAGE_DIRECTORY = '/home/ayushi/Git/research/dataset/coco'
_TFRECORD_SUB_DIRECTORY=os.path.join(_MODEL_DIR,'tfrecord')
_CLASSNAMES_FILENAME = 'coco_dict80.txt'

def _get_image_list_by_diversity(image_list, img_annots, how_many_imgs):
    n_labels = np.shape(img_annots)[1]
    sel_min_label_freq = int(math.ceil(how_many_imgs / n_labels)) - 1
    sorted_labels = np.argsort(np.sum(img_annots, axis=0))
    sel_images_idx = []
    sel_images_annots = np.zeros([how_many_imgs, n_labels], dtype=np.int32)
    sel_label_freq = np.zeros([n_labels, 1], dtype=np.int32)
    cnt_sel_imgs = 0
    while cnt_sel_imgs < how_many_imgs:
        sel_min_label_freq = sel_min_label_freq + 1
        for i in range(n_labels):
            if how_many_imgs <= cnt_sel_imgs:
                break
            label_imgs_idx = np.argwhere(img_annots[:, sorted_labels[i]])
            label_imgs_idx = np.setdiff1d(label_imgs_idx, sel_images_idx, assume_unique=True)
            n_select = max(0, sel_min_label_freq - sel_label_freq[sorted_labels[i]])
            n_select = min(len(label_imgs_idx), n_select, how_many_imgs - cnt_sel_imgs)
            if n_select > 0:
                sel_label_imgs = np.random.choice(label_imgs_idx, n_select, replace=False)
                for sel_img in sel_label_imgs:
                    sel_images_idx.append(sel_img)
                    sel_images_annots[cnt_sel_imgs, :] = img_annots[sel_img, :]
                    sel_labels = np.argwhere(img_annots[sel_img, :])
                    sel_label_freq[sel_labels] = sel_label_freq[sel_labels] + 1
                    cnt_sel_imgs = cnt_sel_imgs + 1
    return image_list[sel_images_idx], sel_images_annots, sel_images_idx


def _extract_imagelist(filename):
  """Extract the imagelist.

  Args:
    filename: The path to an NUSWIDE images file.

  Returns:
    An image list.
  """
  print('Extracting images from: ', filename)
  imgList = []
  with open(filename) as fImgList:
      imgList1 = fImgList.read().splitlines()
  for x in imgList1:
      img = x.split(' ')[0]
      imgList.append(img)
  return np.array(imgList)


def _extract_labels(filename):
  """Extract the label list.

  Args:
    filename: The path to an NUSWIDE labels file.

  Returns:
    A label list.
  """
  print('Extracting labels from: ', filename)
  labels = []
  with open(filename) as fAnnot:
    annots=fAnnot.read().splitlines()
  for annotLine in annots:
    label_indicator = map(int,annotLine.strip().split(' '))
    labels.append(label_indicator)
  return np.array(labels)


def _convert_to_tfrecord(dataset_dir, split_name,
                         data_filename, label_filename,
                         prob_filename, feature_filename):
  """Loads data from the image list files and writes images to a TFRecord.

  Args:
    dataset_dir: The root directory.
    split_name: The name of the train/test split.
    data_filename: The filename of the MNIST images.
    label_filename: The filename of the MNIST labels.
  """
  image_list = _extract_imagelist(data_filename)
  labels = _extract_labels(label_filename)
  db_ftr = h5py.File(feature_filename, 'r')['ftr']
  db_scores = h5py.File(prob_filename, 'r')['testProb']
  if split_name == 'valid':
        image_list, labels, sel_images_idx = _get_image_list_by_diversity(image_list, labels, 600)
        num_images = 600
  else:
      num_images = len(image_list)
      sel_images_idx = range(num_images)
  labels = labels.tolist()
  assert(len(labels) == num_images)
  num_shards = int(math.floor(float(num_images)/_NUM_PER_SHARD))
  for i in range(num_shards+1):
    st_idx = i*_NUM_PER_SHARD
    ed_idx = min(st_idx+_NUM_PER_SHARD,num_images)
    output_filename = _get_tfrecord_filename(dataset_dir,split_name,i,num_shards)
    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
        for j in range(st_idx,ed_idx):
            sys.stdout.write('\r>> Converting image %d/%d' % (j + 1, num_images))
            sys.stdout.flush()
            image_path = os.path.join(_IMAGE_DIRECTORY, image_list[j])
            image_string = tf.gfile.FastGFile(image_path, 'rb').read()
            image = Image.open(image_path)
            image_ftr = db_ftr[:, sel_images_idx[j]].T
            image_scores = db_scores[sel_images_idx[j], :]
            example = dataset_utils.image_to_tfexample(
                image_string, image.format.encode(), image.size[0],
                image.size[1], labels[j], image_ftr.tolist(),
                image_scores.tolist(), image_path)
            tfrecord_writer.write(example.SerializeToString())


def _get_tfrecord_filename(dataset_dir, split_name, shard_id, num_shards):
  """Creates the output filename.

  Args:
    dataset_dir: The root directory.
    split_name: The name of the train/test split.

  Returns:
    An absolute file path.
  """
  output_filename = 'coco_%s_%05d-of-%05d.tfrecord' % (
      split_name, shard_id, num_shards)
  return os.path.join(_TFRECORD_SUB_DIRECTORY,output_filename)


def run(dataset_dir=_DATASET_DIR, create_label_dict=False):

  """Runs the conversion operation.

  Args:
    dataset_dir: The root directory.
  """

  # Process for each of the data splits:
  for idx, split in enumerate(_SPLIT_NAMES):
    data_filename = os.path.join(_DATASET_DIR, _DATA_FILENAMES[idx])
    label_filename = os.path.join(_DATASET_DIR, _LABEL_FILENAMES[idx])
    prob_filename = os.path.join(_RESULTS_DIR, _PROB_FILENAMES[idx])
    feature_filename = os.path.join(_MODEL_DIR, _FTR_FILENAMES[idx])
    _convert_to_tfrecord(_DATASET_DIR, split, data_filename, label_filename,
                         prob_filename,
                         feature_filename)

  if create_label_dict==True:
      class_filename = os.path.join(_DATASET_DIR, _CLASSNAMES_FILENAME)
      with open(class_filename) as fClassNames:
          class_names = fClassNames.read().splitlines()
      labels_to_class_names = dict(zip(range(len(class_names)), class_names))
      dataset_utils.write_label_file(labels_to_class_names, _DATASET_DIR)

  print('\nFinished converting the NUSWIDE dataset!')
