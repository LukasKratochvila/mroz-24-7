import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend
from tensorflow.python.keras.metrics import Metric


class CM(Metric):
    def __init__(self, num_classes: int, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        #self._dtype = tf.int32

        # Variable to accumulate the predictions in the confusion matrix.
        self.total_cm = self.add_weight(
            'total_confusion_matrix',
            shape=(num_classes, num_classes),
            initializer=tf.compat.v1.zeros_initializer)

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates the confusion matrix statistics.

        Args:
          y_true: The ground truth values.
          y_pred: The predicted values.
          sample_weight: Optional weighting of each example. Defaults to 1. Can be a
            `Tensor` whose rank is either 0, or the same rank as `y_true`, and must
            be broadcastable to `y_true`.

        Returns:
          Update op.
        """
        if isinstance(y_pred, tuple):
            y_pred = y_pred[1]

        y_true = tf.cast(y_true, self._dtype)
        y_pred = tf.cast(y_pred, self._dtype)


        # Flatten the input if its rank > 1.
        if y_pred.shape.ndims > 1:
            y_pred = tf.reshape(y_pred, [-1])

        if y_true.shape.ndims > 1:
            y_true = tf.reshape(y_true, [-1])

        # Accumulate the prediction to current confusion matrix.
        current_cm = tf.math.confusion_matrix(
            y_true,
            y_pred,
            self.num_classes,
            dtype=self._dtype)
        return self.total_cm.assign_add(current_cm)

    def reset_state(self):
        backend.set_value(
            self.total_cm, np.zeros((self.num_classes, self.num_classes)))

    def result(self):
        return self.total_cm

    def get_config(self):
        config = {
            'num_classes': self.num_classes,
        }
        return config
    
    def cm_metrics(self):
        confusion_matrix = self.total_cm.numpy()
        FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
        FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
        TP = np.diag(confusion_matrix)
        TN = confusion_matrix.sum() - (FP + FN + TP)

        # Class accuracy
        acc = (TP + TN) / (TP + FP + FN + TN)

        # Sensitivity, hit rate, recall, or true positive rate
        recall = TP / (TP + FN)

        # Precision or positive predictive value
        precision = TP / (TP + FP)

        # F1 score
        f1 = 2 * precision * recall / (precision + recall)

        # JI Jaccard index
        ji = TP / (TP + FN + FP)

        return acc, recall, precision, f1, ji, [TP, FP, TN, FN]