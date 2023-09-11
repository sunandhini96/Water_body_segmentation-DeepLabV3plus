
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K


def cal_metrics_pre_re(y_true, y_pred):
    for class_id in [0, 1]:
        pred_samples = y_pred.get(class_id, set())
        truth_samples = y_truth.get(class_id, set())
        common_samples = pred_samples & truth_samples
        try:
            precision = len(common_samples) / len(pred_samples)
        except ZeroDivisionError:
            precision = None
        try:
            recall = len(common_samples) / len(truth_samples)
        except ZeroDivisionError:
            recall = None
        
        precisions[class_id] = precision
        recalls[class_id] = recall
    
    return precisions, recalls

def iou(y_true, y_pred, labels=[0, 1], average=None):
    def f(y_true, y_pred):
        for class_id in labels:
            pred_samples = y_pred.get(class_id, set())
            truth_samples = y_truth.get(class_id, set())
            common_samples = pred_samples & truth_samples
            intersection = (truth_samples * pred_samples).sum()
            union = truth_samples.sum() + pred_samples.sum() - intersection
            x = (intersection + 1e-15) / (union + 1e-15)
            x = x.astype(np.float32)
            return x
            ious[class_id] = tf.numpy_function(f, [truth_samples, pred_samples], tf.float32)

        return ious

def dice_coef(y_true, y_pred):
    smooth = 1e-15
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    for class_id in [0, 1]:
            pred_samples = y_pred.get(class_id, set())
            truth_samples = y_truth.get(class_id, set())
            common_samples = pred_samples & truth_samples
            intersection = tf.reduce_sum(pred_samples * truth_samples)
            dice = (2. * intersection + smooth) / (tf.reduce_sum(truth_samples) + tf.reduce_sum(pred_samples) + smooth)
            dices[class_id] = dice
    return dices
    
