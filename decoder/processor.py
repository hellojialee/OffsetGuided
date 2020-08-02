"""Wrap the NN inference and post-processing"""
import io
import logging
import multiprocessing
import pstats
import time

import numpy as np
import torch


LOG = logging.getLogger(__name__)


class DummyPool():  # todo 只需要把GreedyGroupg给变成多进程
    @staticmethod
    def starmap(f, iterable):
        return [f(*i) for i in iterable]


