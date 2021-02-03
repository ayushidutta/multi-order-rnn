from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import h5py
import scipy.io

from eval.multilabel_metrics import evaluate_per_row_col_metrics, annotate_topK, annotate_by_probability

_NUM_REC_PREDICT_FILE = 125500

class LabelEval(object):

    def __init__(self, num_eval_samples, num_classes, caption_length, topK, batch_size, save_file=None):
        self.num_eval_samples = num_eval_samples
        self.predicted_scores = np.zeros([num_eval_samples, num_classes])
        self.annotations = np.zeros([num_eval_samples, num_classes], np.int32)
        self.predicted_caption = np.zeros([num_eval_samples, caption_length], np.int32)
        self.topK = topK
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.save_file = save_file

    def eval_update(self, st_idx, batch_pred_scores, batch_pred_label, batch_data):
        batch_annotations = batch_data['label']
        batch_remain = min(self.batch_size, self.num_eval_samples - st_idx)
        self.predicted_scores[st_idx:st_idx + batch_remain, :] = batch_pred_scores
        self.annotations[st_idx:st_idx + batch_remain, :] = batch_annotations
        self.predicted_caption[st_idx:st_idx + batch_remain, :] = batch_pred_label

    def eval(self):
        def _eval(annot_type, predicted_annot):
            resI, resL = evaluate_per_row_col_metrics(self.annotations, self.predicted_scores, predicted_annot, self.topK, True)
            perf = {
                'acc_top1': resI['acc_top1'],
                'acc_topK': resI['acc_topK'],
                'f1_label': resL['f1'],
                'prec_label': resL['prec'],
                'rec_label': resL['rec'],
                'map_label': resL['map'],
                'f1_image': resI['f1'],
                'prec_image': resI['prec'],
                'rec_image': resI['rec'],
                'map_image': resI['map'],
            }
            tf.logging.info('Annotation Type :%s, Performance :%s' % (annot_type, perf))
        predicted_topK = annotate_topK(self.predicted_scores, self.topK)
        predicted_prob = annotate_by_probability(self.predicted_scores)
        predicted_top1 = np.zeros([self.num_eval_samples, self.num_classes], np.int32)
        for i in range(self.num_eval_samples):
            _predictions = self.predicted_caption[i, :]
            predicted_top1[i, _predictions] = 1
        _eval('topK', predicted_topK)
        _eval('prob', predicted_prob)
        _eval('caption', predicted_top1)
        if self.save_file:
            scipy.io.savemat(self.save_file, mdict={
                'testScores': self.predicted_scores})


class PredictionSave(object):

    def __init__(self, file_path, pred_keys):
        self.file_path = file_path
        self.pred_keys = pred_keys
        self.file_idx = -1
        self._get_file(0)

    def _get_file(self, curr_rec_idx):
        if curr_rec_idx >= (self.file_idx+1) * _NUM_REC_PREDICT_FILE:
            self.file_idx = self.file_idx + 1
            _file_path = '%s_%05d.h5' % (self.file_path, self.file_idx)
            self.file = h5py.File(_file_path,'w')
        return self.file

    def save(self, st_idx, batch_predictions_list, batch_data):
        _file = self._get_file(st_idx)
        for idx, _ in enumerate(batch_predictions_list[0]):
            image_grp = _file.create_group(str(st_idx+idx))
            image_grp.create_dataset("image_path", data=batch_data['path'][idx])
            if 'caption' in batch_data:
                image_grp.create_dataset("image_caption", data=batch_data['caption'][idx])
            elif 'label' in batch_data:
                image_grp.create_dataset("image_label", data=batch_data['label'][idx])
            elif 'sig_prob' in batch_data:
                image_grp.create_dataset("cnn_sig_prob", data=batch_data['sig_prob'][idx])
            elif 'softmax_prob' in batch_data:
                image_grp.create_dataset("cnn_softmax_prob", data=batch_data['softmax_prob'][idx])
            for pred_idx, pred in enumerate(self.pred_keys):
                image_grp.create_dataset(pred, data=batch_predictions_list[pred_idx][idx])