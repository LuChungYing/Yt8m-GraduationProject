# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Contains a collection of models which operate on variable-length sequences."""
import math

import model_utils as utils
import models
import tensorflow as tf
from tensorflow import flags
import tensorflow.contrib.slim as slim
import video_level_models

FLAGS = flags.FLAGS
flags.DEFINE_integer("iterations", 30, "Number of frames per batch for DBoF.")
flags.DEFINE_bool("dbof_add_batch_norm", True,
                  "Adds batch normalization to the DBoF model.")
flags.DEFINE_bool(
    "sample_random_frames", True,
    "If true samples random frames (for frame level models). If false, a random"
    "sequence of frames is sampled instead.")
flags.DEFINE_integer("dbof_cluster_size", 8192,
                     "Number of units in the DBoF cluster layer.")
flags.DEFINE_integer("dbof_hidden_size", 1024,
                     "Number of units in the DBoF hidden layer.")
flags.DEFINE_string(
    "dbof_pooling_method", "max",
    "The pooling method used in the DBoF cluster layer. "
    "Choices are 'average' and 'max'.")
flags.DEFINE_string(
    "dbof_activation", "sigmoid",
    "The nonlinear activation method for cluster and hidden dense layer, e.g., "
    "sigmoid, relu6, etc.")
flags.DEFINE_string(
    "video_level_classifier_model", "MoeModel",
    "Some Frame-Level models can be decomposed into a "
    "generalized pooling operation followed by a "
    "classifier layer")
flags.DEFINE_integer("lstm_cells", 1024, "Number of LSTM cells.")
flags.DEFINE_integer("lstm_layers", 2, "Number of LSTM layers.")
######################################################
flags.DEFINE_bool("rnn_swap_memory", False, "If true, swap_memory = True.")
flags.DEFINE_string("cnn_filter_sizes", "1,2,3", "Sizes of cnn filters.")
flags.DEFINE_string("cnn_filter_nums", "256,256,256", "Numbers of every cnn filters.")
flags.DEFINE_integer("cnn_pooling_k", 4, "The k value for max-k pooling.")
flags.DEFINE_string("video_level_classifier_support_model", "MoeModel",
                    "Some Frame-Level models can be decomposed into a "
                    "generalized pooling operation followed by a "
                    "classifier layer")
flags.DEFINE_integer("cnn_num_filters", 512, "number of filters in cnn conv layer.")
flags.DEFINE_integer("deep_cnn_base_size", 128,
                     "basic cnn size")
flags.DEFINE_integer("frame_seg_relu_cells", 256, "number of relu cells in frame-seg model.")
flags.DEFINE_integer("gru_cells", 1024, "Number of GRU cells.")
flags.DEFINE_integer("gru_layers", 2, "Number of GRU layers.")
flags.DEFINE_integer("lstm_attentions", 8, "Attention size in lstm_attention_max_pooling_model.")

flags.DEFINE_integer("lstm_look_back", 3,
                     "how many adjacent input for a cell to look at")
flags.DEFINE_string("lstm_normalization", "identical",
                    "which normalization method")

flags.DEFINE_integer("attention_size", 1, "Number of attention layers.")
flags.DEFINE_integer("num_attentions", 5, "Number of attention cells per layer.")

flags.DEFINE_integer("positional_embedding_size", 32, "Positional embedding dimension use in lstm_positional_attention_max_pooling_model.")


flags.DEFINE_integer("mm_label_embedding", 256,
                    "size of label embedding vector")

flags.DEFINE_integer("multiscale_cnn_lstm_layers", 1, "number of layers in multiscale cnn_lstm.")

flags.DEFINE_bool("is_training", False, "used in batch normalization.")


flags.DEFINE_string("wide_and_deep_models", "FrameLevelLogisticModel,LstmMemoryModel",
                    "size of label embedding vector")
######################################################
class FrameLevelLogisticModel(models.BaseModel):
  """Creates a logistic classifier over the aggregated frame-level features."""

  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
    """See base class.

    This class is intended to be an example for implementors of frame level
    models. If you want to train a model over averaged features it is more
    efficient to average them beforehand rather than on the fly.

    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
        input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
        frames for each video (before padding).

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """
    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    feature_size = model_input.get_shape().as_list()[2]

    denominators = tf.reshape(tf.tile(num_frames, [1, feature_size]),
                              [-1, feature_size])
    avg_pooled = tf.reduce_sum(model_input, axis=[1]) / denominators

    output = slim.fully_connected(avg_pooled,
                                  vocab_size,
                                  activation_fn=tf.nn.sigmoid,
                                  weights_regularizer=slim.l2_regularizer(1e-8))
    return {"predictions": output}


class DbofModel(models.BaseModel):
  """Creates a Deep Bag of Frames model.

  The model projects the features for each frame into a higher dimensional
  'clustering' space, pools across frames in that space, and then
  uses a configurable video-level model to classify the now aggregated features.

  The model will randomly sample either frames or sequences of frames during
  training to speed up convergence.
  """

  ACT_FN_MAP = {
      "sigmoid": tf.nn.sigmoid,
      "relu6": tf.nn.relu6,
  }

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_frames,
                   iterations=None,
                   add_batch_norm=None,
                   sample_random_frames=None,
                   cluster_size=None,
                   hidden_size=None,
                   is_training=True,
                   **unused_params):
    """See base class.

    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
        input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
        frames for each video (before padding).
      iterations: the number of frames to be sampled.
      add_batch_norm: whether to add batch norm during training.
      sample_random_frames: whether to sample random frames or random sequences.
      cluster_size: the output neuron number of the cluster layer.
      hidden_size: the output neuron number of the hidden layer.
      is_training: whether to build the graph in training mode.

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """
    iterations = iterations or FLAGS.iterations
    add_batch_norm = add_batch_norm or FLAGS.dbof_add_batch_norm
    random_frames = sample_random_frames or FLAGS.sample_random_frames
    cluster_size = cluster_size or FLAGS.dbof_cluster_size
    hidden1_size = hidden_size or FLAGS.dbof_hidden_size
    act_fn = self.ACT_FN_MAP.get(FLAGS.dbof_activation)
    assert act_fn is not None, ("dbof_activation is not valid: %s." %
                                FLAGS.dbof_activation)

    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    if random_frames:
      model_input = utils.SampleRandomFrames(model_input, num_frames,
                                             iterations)
    else:
      model_input = utils.SampleRandomSequence(model_input, num_frames,
                                               iterations)
    max_frames = model_input.get_shape().as_list()[1]
    feature_size = model_input.get_shape().as_list()[2]
    reshaped_input = tf.reshape(model_input, [-1, feature_size])
    tf.compat.v1.summary.histogram("input_hist", reshaped_input)

    if add_batch_norm:
      reshaped_input = slim.batch_norm(reshaped_input,
                                       center=True,
                                       scale=True,
                                       is_training=is_training,
                                       scope="input_bn")

    cluster_weights = tf.compat.v1.get_variable(
        "cluster_weights", [feature_size, cluster_size],
        initializer=tf.random_normal_initializer(stddev=1 /
                                                 math.sqrt(feature_size)))
    tf.compat.v1.summary.histogram("cluster_weights", cluster_weights)
    activation = tf.matmul(reshaped_input, cluster_weights)
    if add_batch_norm:
      activation = slim.batch_norm(activation,
                                   center=True,
                                   scale=True,
                                   is_training=is_training,
                                   scope="cluster_bn")
    else:
      cluster_biases = tf.compat.v1.get_variable(
          "cluster_biases", [cluster_size],
          initializer=tf.random_normal_initializer(stddev=1 /
                                                   math.sqrt(feature_size)))
      tf.compat.v1.summary.histogram("cluster_biases", cluster_biases)
      activation += cluster_biases
    activation = act_fn(activation)
    tf.compat.v1.summary.histogram("cluster_output", activation)

    activation = tf.reshape(activation, [-1, max_frames, cluster_size])
    activation = utils.FramePooling(activation, FLAGS.dbof_pooling_method)

    hidden1_weights = tf.compat.v1.get_variable(
        "hidden1_weights", [cluster_size, hidden1_size],
        initializer=tf.random_normal_initializer(stddev=1 /
                                                 math.sqrt(cluster_size)))
    tf.compat.v1.summary.histogram("hidden1_weights", hidden1_weights)
    activation = tf.matmul(activation, hidden1_weights)
    if add_batch_norm:
      activation = slim.batch_norm(activation,
                                   center=True,
                                   scale=True,
                                   is_training=is_training,
                                   scope="hidden1_bn")
    else:
      hidden1_biases = tf.compat.v1.get_variable(
          "hidden1_biases", [hidden1_size],
          initializer=tf.random_normal_initializer(stddev=0.01))
      tf.compat.v1.summary.histogram("hidden1_biases", hidden1_biases)
      activation += hidden1_biases
    activation = act_fn(activation)
    tf.compat.v1.summary.histogram("hidden1_output", activation)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(model_input=activation,
                                           vocab_size=vocab_size,
                                           **unused_params)


class LstmModel(models.BaseModel):
  """Creates a model which uses a stack of LSTMs to represent the video."""

  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
    """See base class.

    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
        input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
        frames for each video (before padding).

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """
    lstm_size = FLAGS.lstm_cells
    number_of_layers = FLAGS.lstm_layers

    stacked_lstm = tf.contrib.rnn.MultiRNNCell([
        tf.contrib.rnn.BasicLSTMCell(lstm_size, forget_bias=1.0)
        for _ in range(number_of_layers)
    ])

    _, state = tf.nn.dynamic_rnn(stacked_lstm,
                                 model_input,
                                 sequence_length=num_frames,
                                 dtype=tf.float32)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)

    return aggregated_model().create_model(model_input=state[-1].h,
                                           vocab_size=vocab_size,
                                           **unused_params)
################################################################################
import sys
from os.path import dirname
if dirname(__file__) not in sys.path:
  sys.path.append(dirname(__file__))
from all_frame_models import *

