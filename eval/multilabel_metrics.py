from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class MultilabelEvaluate:
    def __init__(self,
                 ground_annot,
                 predicted_scores,
                 predicted_annot):
        self.ground_annot = ground_annot
        self.predicted_annot = predicted_annot
        self.n_rows = np.size(ground_annot, 0)
        self.n_cols = np.size(ground_annot, 1)
        self.predicted_rank = np.zeros([self.n_rows, self.n_cols],dtype=np.int32)
        self.nCorrect_row = np.zeros([self.n_rows, 1],dtype=np.int32)
        self.nGround_row = np.zeros([self.n_rows, 1],dtype=np.int32)
        self.nPredict_row = np.zeros([self.n_rows, 1],dtype=np.int32)
        self.results = {}
        for i in range(self.n_rows):
            ground = np.argwhere(self.ground_annot[i, :] == 1)
            self.nGround_row[i] = len(ground)
            predicted = np.argwhere(self.predicted_annot[i, :] == 1)
            self.nPredict_row[i] = len(predicted)
            for j in ground:
                if self.predicted_annot[i, j] == 1:
                    self.nCorrect_row[i] = self.nCorrect_row[i] + 1
            self.predicted_rank[i, :] = np.argsort(predicted_scores[i, :])[::-1]


    def calc_accuracy(self,topK):
        n_top1 = 0
        n_topK = 0
        for i in range(self.n_rows):
                if self.ground_annot[i, self.predicted_rank[i, 0]] == 1:
                    n_top1 = n_top1 + 1
                if self.nCorrect_row[i] > 0:
                    n_topK = n_topK + 1
        acc_top1 = n_top1/self.n_rows
        acc_topK = n_topK/self.n_rows
        self.results['acc_top1'] = acc_top1
        self.results['acc_topK'] = acc_topK
        return acc_top1,acc_topK


    def calc_prec_rec_f1_map(self):
        self.calc_precision()
        self.calc_recall()
        self.calc_f1()
        self.calc_map()
        results = self.results
        return results

    def calc_precision(self):
        prec_row = np.zeros([self.n_rows, 1])
        for i in range(self.n_rows):
            if self.nPredict_row[i] > 0:
                prec_row[i] = self.nCorrect_row[i] / self.nPredict_row[i]
        prec = np.mean(prec_row)
        self.results['prec'] = prec
        self.results['prec_row'] = prec_row
        return prec, prec_row

    def calc_recall(self):
        rec_row = np.zeros([self.n_rows, 1])
        for i in range(self.n_rows):
            if self.nGround_row[i] > 0:
                rec_row[i] = self.nCorrect_row[i] / self.nGround_row[i]
        rec = np.mean(rec_row)
        pos_rec = np.argwhere(rec_row)
        nplus = len(pos_rec)
        self.results['rec'] = rec
        self.results['rec_row'] = rec_row
        self.results['nplus'] = nplus
        return rec, nplus, rec_row

    def calc_f1(self):
        if 'prec' not in self.results:
            self.calc_precision()
        if 'rec' not in self.results:
            self.calc_recall()
        f1 = 0
        f1_row = np.zeros([self.n_rows, 1])
        for i in range(self.n_rows):
            s = self.results['prec_row'][i] + self.results['rec_row'][i]
            if s > 0:
                f1_row[i] = (2 * self.results['prec_row'][i] * self.results['rec_row'][i]) / s
        s = self.results['prec'] + self.results['rec']
        if s > 0:
            f1 = (2 * self.results['prec'] * self.results['rec']) / s
        self.results['f1'] = f1
        self.results['f1_row'] = f1_row
        return f1, f1_row

    def calc_map(self):
        ap_row = np.zeros([self.n_rows, 1])
        for i in range(self.n_rows):
            nc = 0
            ap = 0
            for j in range(self.n_cols):
                if self.ground_annot[i, self.predicted_rank[i, j]] == 1:
                    nc = nc + 1
                    ap = ap + nc / (j+1)
            if self.nGround_row[i] > 0:
                ap_row[i] = ap / self.nGround_row[i]
        MAP = np.mean(ap_row)
        self.results['map'] = MAP
        self.results['ap_row'] = ap_row
        return MAP, ap_row


def evaluate_per_row_col_metrics(ground_annot, prediction_scores, predicted_annot, topK=0, row_wise=True):
    eval_per_row = MultilabelEvaluate(ground_annot, prediction_scores, predicted_annot)
    eval_per_row.calc_prec_rec_f1_map()
    eval_per_col = MultilabelEvaluate(np.transpose(ground_annot),
                                      np.transpose(prediction_scores),
                                      np.transpose(predicted_annot))
    eval_per_col.calc_prec_rec_f1_map()
    if topK > 0:
        if row_wise:
            eval_per_row.calc_accuracy(topK)
        else:
            eval_per_col.calc_accuracy(topK)
    return eval_per_row.results,eval_per_col.results


def annotate_topK(predicted_scores, topK):
    n_rows = np.size(predicted_scores, 0)
    n_labels = np.size(predicted_scores, 1)
    predicted_annot = np.zeros([n_rows,n_labels],dtype=np.int32)
    for i in range(n_rows):
        scores_list = list(predicted_scores[i,:])
        for j in range(topK):
            idx = np.argmax(scores_list)
            scores_list[idx] = -float('inf')
            predicted_annot[i, idx] = 1
    return predicted_annot


def annotate_by_probability(predicted_scores):
    n_rows = np.size(predicted_scores, 0)
    n_labels = np.size(predicted_scores, 1)
    predicted_annot = np.zeros([n_rows, n_labels],dtype=np.int32)
    for i in range(n_rows):
        gt_idx = np.where(predicted_scores[i, :] > 0.5)[0]
        predicted_annot[i, gt_idx] = 1
    return predicted_annot
