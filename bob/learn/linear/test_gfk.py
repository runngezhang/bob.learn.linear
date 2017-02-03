#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>
# Thu Jun 14 14:45:06 CEST 2012
#
# Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland

"""Test BIC trainer and machine
"""

import numpy
import nose.tools
import bob.learn.linear
import bob.io.matlab
from bob.io.base.test_utils import datafile
from bob.learn.linear import GFKTrainer, GFKMachine


def compute_accuracy(K, Xs, Ys, Xt, Yt):
    source = numpy.diag(numpy.dot(numpy.dot(Xs, K), Xs.T))
    source = numpy.reshape(source, (Xs.shape[0], 1))
    source = numpy.matlib.repmat(source, 1, Yt.shape[0])

    target = numpy.diag(numpy.dot(numpy.dot(Xt, K), Xt.T))
    target = numpy.reshape(target, (Xt.shape[0], 1))
    target = numpy.matlib.repmat(target, 1, Ys.shape[0]).T

    dist = source + target - 2 * numpy.dot(numpy.dot(Xs, K), Xt.T)

    indices = numpy.argmin(dist, axis=0)
    prediction = Ys[indices]
    accuracy = sum(prediction == Yt)[0] / float(Yt.shape[0])

    return accuracy


def test_matlab_baseline():
    """

    Tests based on this matlab baseline

    http://www-scf.usc.edu/~boqinggo/domainadaptation.html#intro

    """
    source_webcam = bob.io.matlab.read_matrix(datafile("webcam.mat", __name__))
    webcam_labels = bob.io.matlab.read_matrix(datafile("webcam_labels.mat", __name__))

    target_dslr = bob.io.matlab.read_matrix(datafile("dslr.mat", __name__))
    dslr_labels = bob.io.matlab.read_matrix(datafile("dslr_labels.mat", __name__))

    D = 100
    d = 10
    # for d in range(1,D):

    # Creating the GFK
    gfk = GFKTrainer(10)

    gfk_trainer = GFKTrainer(10)
    gfk_machine = gfk_trainer.train(source_webcam, target_dslr)

    accuracy = compute_accuracy(gfk_machine.K, source_webcam, webcam_labels, target_dslr, dslr_labels)*100

    assert accuracy > 70

