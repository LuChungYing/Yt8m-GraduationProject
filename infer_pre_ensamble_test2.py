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
"""Binary for evaluating Tensorflow models on the YouTube-8M dataset."""

import json
import os
import time

from absl import logging
import eval_util
import frame_level_models
import losses
import readers
import tensorflow as tf
from tensorflow import flags
from tensorflow.python.lib.io import file_io
import utils
import video_level_models

FLAGS = flags.FLAGS

if __name__ == "__main__":
  # Dataset flags.
  flags.DEFINE_string(
      "train_dir", "/tmp/yt8m_model/",
      "The directory to load the model files from. "
      "The tensorboard metrics files are also saved to this "
      "directory.")
  flags.DEFINE_string(
      "input_data_pattern", "",
      "File glob defining the evaluation dataset in tensorflow.SequenceExample "
      "format. The SequenceExamples are expected to have an 'rgb' byte array "
      "sequence feature as well as a 'labels' int64 context feature.")
  flags.DEFINE_bool(
      "segment_labels", False,
      "If set, then --input_data_pattern must be frame-level features (but with"
      " segment_labels). Otherwise, --input_data_pattern must be aggregated "
      "video-level features. The model must also be set appropriately (i.e. to "
      "read 3D batches VS 4D batches.")

  # Other flags.
  flags.DEFINE_integer("batch_size", 32,
                       "How many examples to process per batch.")
  flags.DEFINE_integer("num_readers", 8,
                       "How many threads to use for reading input files.")
  flags.DEFINE_boolean("run_once", False, "Whether to run eval only once.")
  flags.DEFINE_integer("top_k", 20, "How many predictions to output per video.")

def find_class_by_name(name, modules):
  """Searches the provided modules for the named class and returns it."""
  modules = [getattr(module, name, None) for module in modules]
  return next(a for a in modules if a)


def get_input_evaluation_tensors(reader,
                                 data_pattern,
                                 batch_size=1024,
                                 num_readers=1):
  """Creates the section of the graph which reads the evaluation data.

  Args:
    reader: A class which parses the training data.
    data_pattern: A 'glob' style path to the data files.
    batch_size: How many examples to process at a time.
    num_readers: How many I/O threads to use.

  Returns:
    A tuple containing the features tensor, labels tensor, and optionally a
    tensor containing the number of frames per video. The exact dimensions
    depend on the reader being used.

  Raises:
    IOError: If no files matching the given pattern were found.
  """
  logging.info("Using batch size of %d for evaluation.", batch_size)
  with tf.name_scope("eval_input"):
    files = tf.io.gfile.glob(data_pattern)
    if not files:
      raise IOError("Unable to find the evaluation files.")
    logging.info("number of evaluation files: %d", len(files))
    filename_queue = tf.train.string_input_producer(files,
                                                    shuffle=False,
                                                    num_epochs=1)
    eval_data = [
        reader.prepare_reader(filename_queue) for _ in range(num_readers)
    ]
    return tf.train.batch_join(eval_data,
                               batch_size=batch_size,
                               capacity=3 * batch_size,
                               allow_smaller_final_batch=True,
                               enqueue_many=True)
def get_input_data_tensors(reader, data_pattern, batch_size, num_readers=1):
  """Creates the section of the graph which reads the input data.

  Args:
    reader: A class which parses the input data.
    data_pattern: A 'glob' style path to the data files.
    batch_size: How many examples to process at a time.
    num_readers: How many I/O threads to use.

  Returns:
    A tuple containing the features tensor, labels tensor, and optionally a
    tensor containing the number of frames per video. The exact dimensions
    depend on the reader being used.

  Raises:
    IOError: If no files matching the given pattern were found.
  """
  with tf.name_scope("input"):
    files = gfile.Glob(data_pattern)
    if not files:
      raise IOError("Unable to find input files. data_pattern='" +
                    data_pattern + "'")
    logging.info("number of input files: " + str(len(files)))
    filename_queue = tf.train.string_input_producer(files,
                                                    num_epochs=1,
                                                    shuffle=False)
    examples_and_labels = [
        reader.prepare_reader(filename_queue) for _ in range(num_readers)
    ]

    input_data_dict = (tf.train.batch_join(examples_and_labels,
                                           batch_size=batch_size,
                                           allow_smaller_final_batch=True,
                                           enqueue_many=True))
    video_id_batch = input_data_dict["video_ids"]
    video_batch = input_data_dict["video_matrix"]
    num_frames_batch = input_data_dict["num_frames"]
    labels_batch = input_data_dict["labels"]
    return video_id_batch, video_batch, num_frames_batch, labels_batch

def build_graph(reader,
                model,
                input_data_pattern,
                label_loss_fn,
                batch_size=1024,
                num_readers=1):
  """Creates the Tensorflow graph for evaluation.

  Args:
    reader: The data file reader. It should inherit from BaseReader.
    model: The core model (e.g. logistic or neural net). It should inherit from
      BaseModel.
    eval_data_pattern: glob path to the evaluation data files.
    label_loss_fn: What kind of loss to apply to the model. It should inherit
      from BaseLoss.
    batch_size: How many examples to process at a time.
    num_readers: How many threads to use for I/O operations.
  """

  global_step = tf.Variable(0, trainable=False, name="global_step")
  input_data_dict =(get_input_data_tensors(
          reader,
          input_data_pattern,
          batch_size=batch_size))
  video_id_batch = input_data_dict["video_ids"]
  model_input_raw = input_data_dict["video_matrix"]
  labels_batch = input_data_dict["labels"]
  num_frames = input_data_dict["num_frames"]
  tf.compat.v1.summary.histogram("model_input_raw", model_input_raw)

  feature_dim = len(model_input_raw.get_shape()) - 1

  # Normalize input features.
  model_input = tf.nn.l2_normalize(model_input_raw, feature_dim)

  with tf.compat.v1.variable_scope("tower"):
    result = model.create_model(model_input,
                                num_frames=num_frames,
                                vocab_size=reader.num_classes,
                                labels=labels_batch,
                                is_training=False)

    predictions = result["predictions"]
    tf.compat.v1.summary.histogram("model_activations", predictions)
    if "loss" in result.keys():
      label_loss = result["loss"]
    else:
      label_loss = label_loss_fn.calculate_loss(predictions, labels_batch)

  tf.compat.v1.add_to_collection("global_step", global_step)
  tf.compat.v1.add_to_collection("loss", label_loss)
  tf.compat.v1.add_to_collection("predictions", predictions)
  tf.compat.v1.add_to_collection("input_batch", model_input)
  tf.compat.v1.add_to_collection("input_batch_raw", model_input_raw)
  tf.compat.v1.add_to_collection("video_id_batch", video_id_batch)
  tf.compat.v1.add_to_collection("num_frames", num_frames)
  tf.compat.v1.add_to_collection("labels", tf.cast(labels_batch, tf.float32))
  if FLAGS.segment_labels:
    tf.compat.v1.add_to_collection("label_weights",
                                   input_data_dict["label_weights"])
  tf.compat.v1.add_to_collection("summary_op", tf.compat.v1.summary.merge_all())



def evaluate():
  """Starts main evaluation loop."""
  tf.compat.v1.set_random_seed(0)  # for reproducibility

  # Write json of flags
  model_flags_path = os.path.join(FLAGS.train_dir, "model_flags.json")
  if not file_io.file_exists(model_flags_path):
    raise IOError(("Cannot find file %s. Did you run train.py on the same "
                   "--train_dir?") % model_flags_path)
  flags_dict = json.loads(file_io.FileIO(model_flags_path, mode="r").read())

  with tf.Graph().as_default():
    # convert feature_names and feature_sizes to lists of values
    feature_names, feature_sizes = utils.GetListOfFeatureNamesAndSizes(
        flags_dict["feature_names"], flags_dict["feature_sizes"])

    if flags_dict["frame_features"]:
      reader = readers.YT8MFrameFeatureReader(
          feature_names=feature_names,
          feature_sizes=feature_sizes,
          segment_labels=FLAGS.segment_labels)
    else:
      reader = readers.YT8MAggregatedFeatureReader(feature_names=feature_names,
                                                   feature_sizes=feature_sizes)

    model = find_class_by_name(flags_dict["model"],
                               [frame_level_models, video_level_models])()
    label_loss_fn = find_class_by_name(flags_dict["label_loss"], [losses])()

    if not FLAGS.input_data_pattern:
      raise IOError("'input_data_pattern' was not specified. Nothing to "
                    "evaluate.")

    build_graph(reader=reader,
                model=model,
                input_data_pattern=FLAGS.input_data_pattern,
                label_loss_fn=label_loss_fn,
                num_readers=FLAGS.num_readers,
                batch_size=FLAGS.batch_size)
    logging.info("built inference graph")

    # A dict of tensors to be run in Session.
    
    saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
    
    inference(saver, FLAGS.train_dir, 
      FLAGS.output_dir, FLAGS.batch_size, FLAGS.top_k)

    
###############################3
def write_to_record(id_batch, label_batch, predictions, filenum, num_examples_processed):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_dir + '/' + 'predictions-%04d.tfrecord' % filenum)
    for i in range(num_examples_processed):
        video_id = id_batch[i]
        label = np.nonzero(label_batch[i,:])[0]
        example = get_output_feature(video_id, label, [predictions[i,:]], ['predictions'])
        serialized = example.SerializeToString()
        writer.write(serialized)
    writer.close()
def get_output_feature(video_id, labels, features, feature_names):
    feature_maps = {'video_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[video_id])),
                    'labels': tf.train.Feature(int64_list=tf.train.Int64List(value=labels))}
    for feature_index in range(len(feature_names)):
        feature_maps[feature_names[feature_index]] = tf.train.Feature(
            float_list=tf.train.FloatList(value=features[feature_index]))
    example = tf.train.Example(features=tf.train.Features(feature=feature_maps))
    return example
#################################3
def inference(saver, model_checkpoint_path, out_file_location, batch_size, top_k):
  with tf.Session() as sess:

    print model_checkpoint_path, FLAGS.train_dir
    if model_checkpoint_path is None:
       model_checkpoint_path = tf.train.latest_checkpoint(FLAGS.train_dir)

    print model_checkpoint_path, FLAGS.train_dir
    if model_checkpoint_path is None:
      raise Exception("unable to find a checkpoint at location: %s" % model_checkpoint_path)

    logging.info("restoring variables from " + model_checkpoint_path)
    saver.restore(sess, model_checkpoint_path)

    input_tensor = tf.get_collection("input_batch_raw")[0]
    num_frames_tensor = tf.get_collection("num_frames")[0]
    predictions_tensor = tf.get_collection("predictions")[0]
    video_id_tensor = tf.get_collection("video_id_batch")[0]
    labels_tensor = tf.get_collection("labels")[0]
    init_op = tf.global_variables_initializer()

    # Workaround for num_epochs issue.
    def set_up_init_ops(variables):
      init_op_list = []
      for variable in list(variables):
        if "train_input" in variable.name:
          init_op_list.append(tf.assign(variable, 1))
          variables.remove(variable)
      init_op_list.append(tf.variables_initializer(variables))
      return init_op_list

    sess.run(set_up_init_ops(tf.get_collection_ref(
        tf.GraphKeys.LOCAL_VARIABLES)))

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    start_time = time.time()

    filenum = 0
    video_id = []
    video_label = []
    video_features = []
    num_examples_processed = 0

    directory = FLAGS.output_dir
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        raise IOError("Output path exists! path='" + directory + "'")

    try:
      while not coord.should_stop():
          predictions_batch_val, video_id_batch_val, labels_batch_val = sess.run([predictions_tensor, video_id_tensor, labels_tensor])

          video_id.append(video_id_batch_val)
          video_label.append(labels_batch_val)
          video_features.append(predictions_batch_val)

          num_examples_processed += len(video_id_batch_val)
          now = time.time()
          logging.info("num examples processed: " + str(num_examples_processed) + " elapsed seconds: " + "{0:.2f}".format(now-start_time))

          if num_examples_processed >= FLAGS.file_size:
            assert num_examples_processed==FLAGS.file_size, "num_examples_processed should be equal to file_size"
            video_id = np.concatenate(video_id, axis=0)
            video_label = np.concatenate(video_label, axis=0)
            video_features = np.concatenate(video_features, axis=0)
            write_to_record(video_id, video_label, video_features, filenum, num_examples_processed)

            filenum += 1
            video_id = []
            video_label = []
            video_features = []
            num_examples_processed = 0

    except tf.errors.OutOfRangeError:
        logging.info('Done with inference. The output file was written to ' + out_file_location)
    finally:
        coord.request_stop()
        if 0 < num_examples_processed <= FLAGS.file_size:
            video_id = np.concatenate(video_id,axis=0)
            video_label = np.concatenate(video_label,axis=0)
            video_features = np.concatenate(video_features,axis=0)
            write_to_record(video_id, video_label, video_features, filenum,num_examples_processed)

    coord.join(threads)
    sess.close()


##########################################3
def main(unused_argv):
  logging.set_verbosity(logging.INFO)
  logging.info("tensorflow version: %s", tf.__version__)
  evaluate()


if __name__ == "__main__":
  tf.compat.v1.app.run()
