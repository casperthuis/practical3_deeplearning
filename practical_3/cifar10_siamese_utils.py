"""
This module implements utility functions for downloading and reading CIFAR10 data.
You don't need to change anything here.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import cPickle as pickle

from six.moves import xrange

import random
from tensorflow.contrib.learn.python.learn.datasets import base

# Default paths for downloading CIFAR10 data
CIFAR10_FOLDER = 'cifar10/cifar-10-batches-py'

def load_cifar10_batch(batch_filename):
  """
  Loads single batch of CIFAR10 data.
  Args:
    batch_filename: Filename of batch to get data from.
  Returns:
    X: CIFAR10 batch data in numpy array with shape (10000, 32, 32, 3).
    Y: CIFAR10 batch labels in numpy array with shape (10000, ).
  """
  with open(batch_filename, 'rb') as f:
    batch = pickle.load(f)
    X = batch['data']
    Y = batch['labels']
    X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype(np.float32)
    Y = np.array(Y)
    return X, Y

def load_cifar10(cifar10_folder):
  """
  Loads CIFAR10 train and test splits.
  Args:
    cifar10_folder: Folder which contains downloaded CIFAR10 data.
  Returns:
    X_train: CIFAR10 train data in numpy array with shape (50000, 32, 32, 3).
    Y_train: CIFAR10 train labels in numpy array with shape (50000, ).
    X_test: CIFAR10 test data in numpy array with shape (10000, 32, 32, 3).
    Y_test: CIFAR10 test labels in numpy array with shape (10000, ).

  """
  Xs = []
  Ys = []
  for b in range(1, 6):
    batch_filename = os.path.join(cifar10_folder, 'data_batch_' + str(b))
    X, Y = load_cifar10_batch(batch_filename)
    Xs.append(X)
    Ys.append(Y)
  X_train = np.concatenate(Xs)
  Y_train = np.concatenate(Ys)
  X_test, Y_test = load_cifar10_batch(os.path.join(cifar10_folder, 'test_batch'))
  return X_train, Y_train, X_test, Y_test

def get_cifar10_raw_data(data_dir):
  """
  Gets raw CIFAR10 data from http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz.

  Args:
    data_dir: Data directory.
  Returns:
    X_train: CIFAR10 train data in numpy array with shape (50000, 32, 32, 3).
    Y_train: CIFAR10 train labels in numpy array with shape (50000, ).
    X_test: CIFAR10 test data in numpy array with shape (10000, 32, 32, 3).
    Y_test: CIFAR10 test labels in numpy array with shape (10000, ).
  """

  X_train, Y_train, X_test, Y_test = load_cifar10(data_dir)

  return X_train, Y_train, X_test, Y_test

def preprocess_cifar10_data(X_train_raw, Y_train_raw, X_test_raw, Y_test_raw):
  """
  Preprocesses CIFAR10 data by substracting mean from all images.
  Args:
    X_train_raw: CIFAR10 raw train data in numpy array.
    Y_train_raw: CIFAR10 raw train labels in numpy array.
    X_test_raw: CIFAR10 raw test data in numpy array.
    Y_test_raw: CIFAR10 raw test labels in numpy array.
    num_val: Number of validation samples.
  Returns:
    X_train: CIFAR10 train data in numpy array.
    Y_train: CIFAR10 train labels in numpy array.
    X_test: CIFAR10 test data in numpy array.
    Y_test: CIFAR10 test labels in numpy array.
  """
  X_train = X_train_raw.copy()
  Y_train = Y_train_raw.copy()
  X_test = X_test_raw.copy()
  Y_test = Y_test_raw.copy()

  # Substract the mean
  mean_image = np.mean(X_train, axis=0)
  X_train -= mean_image
  X_test -= mean_image

  return X_train, Y_train, X_test, Y_test

def dense_to_one_hot(labels_dense, num_classes):
  """
  Convert class labels from scalars to one-hot vectors.
  Args:
    labels_dense: Dense labels.
    num_classes: Number of classes.

  Outputs:
    labels_one_hot: One-hot encoding for labels.
  """
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot

def create_dataset(source='Train', num_tuples = 500, batch_size = 128, fraction_same = 0.2):
    """
    Creates a list of validation tuples. A tuple consist of image pairs and a label.
    A tuple is basically a minibatch to be used in validation.

    One way to sample data for a minibatch is as follows:
              X_1            X_2               Y
        | image_cl1_1, image_cl1_10  | -->   | 1 |
        | image_cl1_1, image_cl1_4   | -->   | 1 |
        | image_cl1_1, image_cl1_163 | -->   | 1 |
        | image_cl1_1, image_cl1_145 | -->   | 1 |
        | image_cl1_1, image_cl3_8   | -->   | 0 |
        |      .            .        | -->   | 0 |
        |      .            .        | -->   | 0 |
        |      .            .        | -->   | 0 |
        | image_cl1_1, image_cl5_8   | -->   | 0 |
        | image_cl1_1, image_cl2_    | -->   | 0 |
        | image_cl1_1, image_cl10_8  | -->   | 0 |

    In this example, image_cl1_1 is an anchor image. All pairs in this batch contains this
    one as reference paired against random samples from the same class and opposite classes.
    The ratio between the number of + and - cases is controlled by fraction_same.

    Args:
      source: Where to sample from train or test set.
      num_tuples: Number of tuples to be used in the validation
      batch_size: Batch size.
      fraction_same: float in range [0,1], defines the fraction
                        of genuine pairs in the batch

    Returns:
      dset: A list of tuples of length num_tuples.
            Each tuple (minibatch) is of shape [batch_size, 32, 32, 3]
    """
    ########################
    # PUT YOUR CODE HERE  #
    ########################
    dataobj = DataSet()
    dset = []
    for i in range(num_tuples):
      dset.append(dataobj.next_batch(batch_size, fraction_same))

    ########################
    # END OF YOUR CODE    #
    ########################
    return dset

class DataSet(object):
  """
  Utility class to handle dataset structure.
  """

  def __init__(self, images, labels):
    """
    Builds dataset with images and labels.
    Args:
      images: Images data.
      labels: Labels data
    """
    assert images.shape[0] == labels.shape[0], (
          "images.shape: {0}, labels.shape: {1}".format(str(images.shape), str(labels.shape)))

    self._num_examples = images.shape[0]
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0
    self._id_list = []

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fraction_same = 0.2):
    """
    Returns the next `batch_size` examples from this data set. A batch consist of image pairs and a label.

    One way to sample data for a minibatch is as follows:
              X_1            X_2             Labels
        | image_cl1_1, image_cl1_10  | -->   | 1 |
        | image_cl1_1, image_cl1_4   | -->   | 1 |
        | image_cl1_1, image_cl1_163 | -->   | 1 |
        | image_cl1_1, image_cl1_145 | -->   | 1 |
        | image_cl1_1, image_cl3_8   | -->   | 0 |
        |      .            .        | -->   | 0 |
        |      .            .        | -->   | 0 |
        |      .            .        | -->   | 0 |
        | image_cl1_1, image_cl5_8   | -->   | 0 |
        | image_cl1_1, image_cl2_    | -->   | 0 |
        | image_cl1_1, image_cl10_8  | -->   | 0 |

    In this example, image_cl1_1 is an anchor image. All pairs in this batch contains this
    one as reference paired against random samples from the same class and opposite classes.
    The ratio between the number of + and - cases is controlled by fraction_same.

    Args:
      batch_size: Batch size.
      fraction_same: float in range [0,1], defines the fraction
                        of genuine pairs in the batch

    Returns:if
      x1: 4D numpy array of shape [batch_size, 32, 32, 3]
      x2: 4D numpy array of shape [batch_size, 32, 32, 3]
      labels: numpy array of shape [batch_size]
    """

    ########################
    # PUT YOUR CODE HERE  #
    ########################
    fraction_same_number = int(fraction_same * batch_size)

    index_anchor_image = np.random.choice(len(self.images), 1)[0]
    label_anchor_image = self.labels[index_anchor_image]
    anchor_image = self.images[index_anchor_image]

    class_images_index = np.nonzero(label_anchor_image)[0][0] == np.nonzero(self.labels)[1]
    class_images = self.images[class_images_index]
    # class_labels = self.labels[class_images_index]
    random_class_indexes = np.random.choice(len(class_images), fraction_same_number)
    random_class_images = class_images[random_class_indexes]
    random_class_labels = np.ones(len(random_class_images))

    # non_class_images_index = np.nonzero(label_anchor_image)[0][0] != np.nonzero(self.labels)[1]
    non_class_images = self.images[~class_images_index]
    random_non_class_indexes = np.random.choice(len(non_class_images), batch_size - fraction_same_number)
    random_non_class_images = non_class_images[random_non_class_indexes]
    random_non_class_labels = np.zeros(len(random_non_class_images))

    x1 = np.tile(anchor_image, (batch_size, 1, 1, 1))
    labels = np.append(random_class_labels, random_non_class_labels)
    x2 = np.concatenate((random_class_images, random_non_class_images))

    ########################
    # END OF YOUR CODE    #
    ########################

    return x1, x2, labels

def read_data_sets(data_dir, one_hot = True, validation_size = 0):
  """
  Returns the dataset readed from data_dir.
  Uses or not uses one-hot encoding for the labels.
  Subsamples validation set with specified size if necessary.
  Args:
    data_dir: Data directory.
    one_hot: Flag for one hot encoding.
    validation_size: Size of validation set
  Returns:
    Train, Validation, Test Datasets
  """
  # Extract CIFAR10 data
  train_images_raw, train_labels_raw, test_images_raw, test_labels_raw = \
      get_cifar10_raw_data(data_dir)
  train_images, train_labels, test_images, test_labels = \
      preprocess_cifar10_data(train_images_raw, train_labels_raw, test_images_raw, test_labels_raw)

  # Apply one-hot encoding if specified
  if one_hot:
    num_classes = len(np.unique(train_labels))
    train_labels = dense_to_one_hot(train_labels, num_classes)
    test_labels = dense_to_one_hot(test_labels, num_classes)

  # Subsample the validation set from the train set
  if not 0 <= validation_size <= len(train_images):
    raise ValueError("Validation size should be between 0 and {0}. Received: {1}.".format(
        len(train_images), validation_size))

  validation_images = train_images[:validation_size]
  validation_labels = train_labels[:validation_size]
  train_images = train_images[validation_size:]
  train_labels = train_labels[validation_size:]

  # Create datasets
  train = DataSet(train_images, train_labels)
  validation = DataSet(validation_images, validation_labels)
  test = DataSet(test_images, test_labels)

  return base.Datasets(train=train, validation=validation, test=test)


def get_cifar10(data_dir = CIFAR10_FOLDER, one_hot = True, validation_size = 0):
  """
  Prepares CIFAR10 dataset.
  Args:
    data_dir: Data directory.
    one_hot: Flag for one hot encoding.
    validation_size: Size of validation set
  Returns:
    Train, Validation, Test Datasets
  """
  return read_data_sets(data_dir, one_hot, validation_size)
