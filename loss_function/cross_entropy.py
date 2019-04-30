import tensorflow as tf
import numpy as np

def cross(prediction, ground_truth, weight_map=None):

    if len(ground_truth.shape) == len(prediction.shape) :
        ground_truth = ground_truth[...,-1]
    ground_truth = tf.cast(ground_truth, tf.int32)
    entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=prediction, labels = ground_truth)
    if weight_map is None :
        return tf.reduce_mean(entropy)
    else:
        weight_map = tf.cast(weight_map, tf.float64)
    weight_sum = tf.maximum(tf.reduce_sum(weight_map), 1e-6)
    return tf.reduce_sum(entropy * weight_map / weight_sum)
