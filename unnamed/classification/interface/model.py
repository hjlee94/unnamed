from unnamed.log import Logger
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, average_precision_score
from sklearn.preprocessing import label_binarize
import numpy as np
import time
import pickle


class Metric:
    def __init__(self):
        self.eta = 1e-5

        self.metric_table = dict()
        self.metric_table['acc'] = self.accuracy
        self.metric_table['2-acc'] = self.top_2_accuracy
        self.metric_table['3-acc'] = self.top_3_accuracy
        self.metric_table['err'] = self.error
        self.metric_table['roc_auc'] = self.roc_auc
        self.metric_table['prc'] = self.prc
        self.metric_table['tpr'] = self.true_positive_rate
        self.metric_table['fpr'] = self.false_positive_rate
        self.metric_table['precision'] = self.precision
        self.metric_table['recall'] = self.recall
        self.metric_table['f1-score'] = self.f1_score

    def get_metric(self, metric):
        metric_func = None

        if metric in self.metric_table:
            metric_func = self.metric_table[metric]

        else:
            metric_func = None
            Logger.get_instance().log_e('Undefined metric %s'%(metric))

        return metric_func

    def accuracy(self, y_true, y_pred):
        acc = np.mean(y_true == y_pred)
        return acc

    def top_n_accuracy(self, y_true, y_score, n):
        y_candidate = np.argsort(y_score)[:, -n:]

        k_acc = list()
        for y, candidates in zip(y_true, y_candidate):
            if y in candidates:
                score = 1
            else:
                score = 0

            k_acc.append(score)

        return sum(k_acc) / len(k_acc)

    def top_2_accuracy(self, y_true, y_score):
        return self.top_n_accuracy(y_true, y_score, n=2)

    def top_3_accuracy(self, y_true, y_score):
        return self.top_n_accuracy(y_true, y_score, n=3)

    def error(self, y_true, y_pred):
        acc = self.accuracy(y_true, y_pred)
        err = 1 - acc

        return err

    def roc_auc(self, y_true, y_score):
        fpr, tpr, threshold = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        return roc_auc

    def prc(self, y_true, y_score):
        prc = average_precision_score(y_true, y_score)
        return prc

    def true_positive_rate(self, y_true, y_pred):
        # TPR = TP / (TP + FN)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        tpr = (tp + self.eta) / (float(tp + fn) + self.eta)

        return tpr

    def false_positive_rate(self, y_true, y_pred):
        # FPR = FP / (FP + TN)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        fpr = (fp + self.eta) / (float(fp + tn) + self.eta)

        return fpr

    def precision(self, y_true, y_pred):
        # precision = TP / (TP + FP)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        prec = (tp + self.eta) / (float(tp + fp) + self.eta)

        return prec

    def recall(self, y_true, y_pred):
        # recall = tp / (tp + fn)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        recall = (tp + self.eta) / (float(tp + fn) + self.eta)

        return recall

    def f1_score(self, y_true, y_pred):
        prec = self.precision(y_true, y_pred)
        recall = self.recall(y_true, y_pred)

        fscore = (2 * prec * recall) / float(prec + recall)

        return fscore

    def balanced_accuracy(self, y_true, y_pred):
        pass

    def balanced_error(self, y_true, y_pred):
        pass


class ModelInterface:
    def __init__(self, model):
        self.model = model

        self.X_tra = None
        self.y_tra = None

        self.classes = None

        self.marks = list()
        self.metric_list = list()
        self.score_result = dict()
        self.score_cls_result = dict()

        self.common_metric = ['acc','err']
        self.label_based_metric = ['tpr', 'fpr', 'precision', 'recall', 'f1-score']#, 'bacc', 'berr']
        self.proba_based_metric = ['roc_auc', 'prc']
        self.special_based_metric = ['2-acc','3-acc']

        self.metric = Metric()
        self.logger = Logger.get_instance()

    def fit(self, X_tra, y_tra, **kwargs):
        self.X_tra = X_tra
        self.y_tra = y_tra

        self.classes = set(self.y_tra)

        s0 = time.time()
        self.model.fit(self.X_tra, self.y_tra, **kwargs)
        e0 = time.time()

        training_time = e0 - s0

        if 'time' not in self.score_result:
            self.score_result['time'] = list()

        self.score_result['time'].append(training_time)

    def predict(self, X_tes):
        return self.model.predict(X_tes)

    def predict_proba(self, X_tes):
        return self.model.predict_proba(X_tes)

    def get_score(self, X_tes, y_tes, metric, mark):
        if mark not in self.score_result:
            self.score_result[mark] = dict()

        if mark not in self.score_cls_result:
            self.score_cls_result[mark] = dict()

        if metric in self.common_metric:
            score = self._get_common_score(X_tes, y_tes, metric)

        elif metric in self.label_based_metric:
            score = self._get_label_score(X_tes, y_tes, metric, mark)

        elif metric in self.special_based_metric:
            if not hasattr(self.model, 'predict_proba'):
                return

            score = self._get_special_score(X_tes, y_tes, metric)

        elif metric in self.proba_based_metric:
            if not hasattr(self.model, 'predict_proba'):
                return

            score = self._get_proba_score(X_tes, y_tes, metric, mark)

        else:
            self.logger.log_w('Wrong metric given')
            return

        if mark not in self.marks:
            self.marks.append(mark)

        if metric not in self.score_result[mark]:
            self.score_result[mark][metric] = list()

        self.score_result[mark][metric].append(score)

        return score

    def _get_common_score(self, X_tes, y_tes, metric):
        metric_func = self.metric.get_metric(metric)

        y_pred = self.predict(X_tes)
        y_true = y_tes

        score = metric_func(y_true, y_pred)

        return score

    def _get_proba_score(self, X_tes, y_tes, metric, mark):
        metric_func = self.metric.get_metric(metric)

        y_prob = self.predict_proba(X_tes)

        if len(self.classes) < 3:
            score = metric_func(y_tes, y_prob[:,1])

        else:
            if metric not in self.score_cls_result[mark]:
                self.score_cls_result[mark][metric] = dict()

            total_score = 0

            y_tes_encoded = label_binarize(y_tes, classes=np.unique(y_tes))

            for cls in range(len(self.classes)):
                score_cls = metric_func(y_tes_encoded[:, cls], y_prob[:, cls])

                if cls not in self.score_cls_result[mark][metric]:
                    self.score_cls_result[mark][metric][cls] = list()

                self.score_cls_result[mark][metric][cls].append(score_cls)

                total_score += score_cls

            score = total_score / float(len(self.classes))

        return score

    def _get_special_score(self, X_tes, y_tes, metric):
        metric_func = self.metric.get_metric(metric)

        y_prob = self.predict_proba(X_tes)
        y_true = y_tes

        score = metric_func(y_true, y_prob)

        return score

    def _get_label_score(self, X_tes, y_tes, metric, mark):
        metric_func = self.metric.get_metric(metric)

        y_pred = self.predict(X_tes)

        if len(self.classes) < 3:
            score = metric_func(y_tes, y_pred)

        else:
            if metric not in self.score_cls_result[mark]:
                self.score_cls_result[mark][metric] = dict()

            total_score = 0

            for cls in range(len(self.classes)):
                y_pred_temp = np.where(y_pred == cls, 1, 0)
                y_tes_temp = np.where(y_tes == cls, 1, 0)

                score_cls = metric_func(y_tes_temp, y_pred_temp)

                if cls not in self.score_cls_result[mark][metric]:
                    self.score_cls_result[mark][metric][cls] = list()

                self.score_cls_result[mark][metric][cls].append(score_cls)

                total_score += score_cls

            score = total_score / float(len(self.classes))

        return score

    def report(self):
        self.logger.log_i('======== Classification Algorithm ==========')
        self.logger.log_i(self.model)
        self.logger.log_i('')

        for mark in self.marks:
            self.logger.log_i('======== Classification Performance : %s =========='%mark)
            metrics = self.score_result[mark].keys()

            for metric in metrics:
                score_list = self.score_result[mark][metric]
                score_str = list(map(lambda x: float('%.3f'%x), score_list))
                self.logger.log_i(' {0} = {1}, mean = {2:.3f}, std = {3:.3f}'.format(metric, score_str, np.mean(score_list), np.std(score_list)))

            self.logger.log_i('')

        if len(self.classes) > 2:
            for mark in self.marks:
                metrics = self.score_cls_result[mark].keys() # mark-metrics-cls

                self.logger.log_i('====== Classification Average Performance Per Each Class : %s ======'%(mark))

                for metric in metrics:
                    class_score = self.score_cls_result[mark][metric]

                    for cls in class_score.keys():
                        class_score[cls] = float('%.3f'%(np.mean(class_score[cls])))

                    score_list = [class_score[cls] for cls in class_score]

                    report_line = str()
                    report_line += ' {0} = {1}'.format(metric, class_score)
                    report_line += ' mu = {0:.3f}'.format(np.mean(score_list))
                    report_line += ' std = {0:.3f}'.format(np.std(score_list))

                    self.logger.log_i(report_line)
                self.logger.log_i('')

        if 'time' in self.score_result:
            self.logger.log_i('======== Classification Performance : time ==========')
            score_list = self.score_result['time']
            score_str = list(map(lambda x: float('%.3f' % x), score_list))
            self.logger.log_i(' {0} = {1}, mean = {2:.3f}, std = {3:.3f}'.format('time', score_str, np.mean(score_list), np.std(score_list)))
            self.logger.log_i('')

    def save_model(self, filename):
        fd = open(filename, 'wb')
        pickle.dump(self.model, fd)
        fd.close()

    def load_model(self, filename):
        fd = open(filename, 'rb')
        self.model = pickle.load(fd)
        fd.close()

    def get_model(self):
        return self.model

    def get_score_on_classes(self):
        return self.score_cls_result
