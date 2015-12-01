"""Auxiliary functions to be used for preparing the training data."""

import numpy
from ._library import BICMachine

def bic_intra_extra_pair_count(training_data):
  """bic_intra_extra_pair_count(training_data) -> intra_count, extra_count

  Returns the total number of intra-class and extra-class pairs generatable from the given list of training data.

  **Keyword parameters**

  training_data : [[object]]
    The training data, where the data for each class are enclosed in one list.

  **Return values**

  intra_count : int
    The total maximum number of intra-class pairs that can be generated from the given ``training_data``.

  intra_count : int
    The total maximum number of between-class pairs that can be generated from the given ``training_data``.
  """

  intra_count = sum(len(c)*(len(c)-1)/2 for c in training_data)
  extra_count = sum(len(training_data[c1])*len(training_data[c2]) for c1 in range(len(training_data)-1) for c2 in range(c1+1, len(training_data)))

  return intra_count, extra_count


def bic_intra_pairs(training_data):
  """bic_intra_pairs(training_data) -> iterator

  Computes intra-class pairs from given training data.

  The ``training_data`` should be aligned in a list of sub-lists, where each sub-list contains the data of one class.
  This function will return an iterator to tuples of data from the same class.
  These tuples can be used to compute difference vectors, which then can be fed into the :py:meth:`BICTrainer.train` method.

  **Keyword parameters**

  training_data : [[object]]
    The training data, where the data for each class are enclosed in one list.

  **Return values**

  iterator : iterator over [(object, object)]
    An iterator over tuples of data, where both data belong to the same class, where each data element is a reference to one element of the given ``training_data``.
  """
  for clazz in training_data:
    for c1 in range(len(clazz)-1):
      for c2 in range (c1+1, len(clazz)):
        yield clazz[c1], clazz[c2]


def bic_extra_pairs(training_data):
  """bic_extra_pairs(training_data) -> iterator

  Computes extra-class pairs from given training data.

  The ``training_data`` should be aligned in a list of sub-lists, where each sub-list contains the data of one class.
  This function will return an iterator to tuples of data of different classes.
  These tuples can be used to compute difference vectors, which then can be fed into the :py:meth:`BICTrainer.train` method.

  **Keyword parameters**

  training_data : [[object]]
    The training data, where the data for each class are enclosed in one list.

  **Return values**

  iterator : iterator over [(object, object)]
    A iterator over tuples of data, where both data belong to different classes, where each data element is a reference to one element of the given ``training_data``.
  """
  for clazz1 in range(len(training_data)-1):
    for c1 in training_data[clazz1]:
      for clazz2 in range(clazz1+1, len(training_data)):
        for c2 in training_data[clazz2]:
          yield (c1, c2)


def bic_intra_extra_pairs(training_data):
  """bic_intra_extra_pairs(training_data) -> intra_pairs, extra_pairs

  Computes intra-class and extra-class pairs from given training data.

  The ``training_data`` should be aligned in a list of sub-lists, where each sub-list contains the data of one class.
  This function will return two lists of tuples of data, where the first list contains tuples of the same class, while the second list contains tuples of different classes.
  These tuples can be used to compute difference vectors, which then can be fed into the :py:meth:`BICTrainer.train` method.

  .. note::
     In general, many more ``extra_pairs`` than ``intra_pairs`` are returned.

  .. warning::
     This function actually returns a two lists of pairs of references to the given data.
     Even for relatively low numbers of classes and elements per class, the returned lists may contain billions of pairs, which require huge amounts of memory.

  **Keyword parameters**

  training_data : [[object]]
    The training data, where the data for each class are enclosed in one list.

  **Return values**

  intra_pairs : [(object, object)]
    A list of tuples of data, where both data belong to the same class, where each data element is a reference to one element of the given ``training_data``.

  extra_pairs : [(object, object)]
    A list of tuples of data, where both data belong to different classes, where each data element is a reference to one element of the given ``training_data``.
  """
  # generate intra-class pairs
  intra_pairs = list(bic_intra_pairs(training_data))

  # generate extra-class pairs
  extra_pairs = list(bic_extra_pairs(training_data))

  # return a tuple of pairs
  return (intra_pairs, extra_pairs)


def bic_intra_extra_pairs_between_factors(first_factor, second_factor):
  """bic_intra_extra_pairs_between_factors(first_factor, second_factor) -> intra_pairs, extra_pairs

  Computes intra-class and extra-class pairs from given training data, where only pairs between the first and second factors are considered.

  Both ``first_factor`` and ``second_factor`` should be aligned in a list of sub-lists, where corresponding sub-list contains the data of one class.
  Both lists need to contain the same classes in the same order; empty classes (empty lists) are allowed.
  This function will return two lists of tuples of data, where the first list contains tuples of the same class, while the second list contains tuples of different classes.
  These tuples can be used to compute difference vectors, which then can be fed into the :py:meth:`BICTrainer.train` method.

  .. note::
     In general, many more ``extra_pairs`` than ``intra_pairs`` are returned.

  .. warning::
     This function actually returns a two lists of pairs of references to the given data.
     Even for relatively low numbers of classes and elements per class, the returned lists may contain billions of pairs, which require huge amounts of memory.

  **Keyword parameters**

  first_factor : [[object]]
    The training data for the first factor, where the data for each class are enclosed in one list.

  second_factor : [[object]]
    The training data for the second factor, where the data for each class are enclosed in one list.
    Must have the same size as ``first_factor``.

  **Return values**

  intra_pairs : [(array_like, array_like)]
    A list of tuples of data, where both data belong to the same class, but different factors.

  extra_pairs : [(array_like, array_like)]
    A list of tuples of data, where both data belong to different classes and different factors.
  """

  assert len(first_factor) == len(second_factor), "The data for both factors must contain the same number of classes"

  # generate intra-class pairs
  intra_pairs = [(c1,c2) \
                  for clazz in range(len(first_factor)) \
                  for c1 in first_factor[clazz] \
                  for c2 in second_factor[clazz]
                ]

  # generate extra-class pairs
  extra_pairs = [(c1, c2) \
                  for clazz1 in range(len(first_factor)) \
                  for c1 in first_factor[clazz1] \
                  for clazz2 in range(len(second_factor)) \
                  for c2 in second_factor[clazz2] \
                  if clazz1 != clazz2
                ]

  # return a tuple of pairs
  return (intra_pairs, extra_pairs)


def train_iec(intra_iterator, extra_iterator, machine=None):
  """train_iec(intra_iterator, extra_iterator, bic_machine=None) -> machine

  Trains a BIC machine using IEC (i.e., without estimating subspaces) from the given training data **iterators**.

  This function exists for convenience as it uses iterators to compute the means and variances of intra-class and extra-class difference vectors.
  Hence, it is identical to the :py:meth:`BICTrainer.train` function, but much more memory-efficient (though it might be slower).

  .. note::
     This function will not compute projection matrices, but only the means and variances of the difference vectors according to [Guenther09]_.

  **Keyword parameters**

  intra_iterator : iterator to array_like(1D, float)
    An iterator to the intra-class difference vectors.

  extra_iterator : iterator to array_like(1D, float)
    An iterator to the extra-class difference vectors.

  machine : :py:class:`bob.learn.linear.BICMachine` or ``None``
    The machine to be trained.
    If not given, a new machine will be created.

  **Returns**

  machine : :py:class:`bob.learn.linear.BICMachine`
    The trained machine.
    If the ``machine`` parameter was given, the returned machine is identical to the given one.
  """

  def _mean_and_variance(iterator):
    # computes mean and variance using the given iterator
    mean = None
    variances = None

    # iterate through the data, and calculate mean and variance
    # also count the number of elements (starting the counter at 1, so that we get the real number of elements)
    for counter, diff in enumerate(iterator,1):
      if mean is None:
        mean = diff
        variances = numpy.square(diff)
      else:
        mean += diff
        variances += numpy.square(diff)

    # normalize mean and variance
    variances = (variances - numpy.square(mean) / counter) / (counter - 1.);
    mean /= counter;

    return mean, variances

  # get means and variances for intra and extra-class comparisons
  intra_mean, intra_variances = _mean_and_variance(intra_iterator)
  extra_mean, extra_variances = _mean_and_variance(extra_iterator)

  # create a new machine, if necessary
  if machine is None:
    machine = BICMachine()

  # set the machine's means and variances
  machine.set_iec(intra_mean, intra_variances, extra_mean, extra_variances)

  return machine
