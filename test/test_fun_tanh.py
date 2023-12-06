import unittest
import pytest

import numpy as np

from mygrad.tensor import Tensor
from mygrad.act_fun import tanh

class TestTensorMatMul(unittest.TestCase):
    def test_simple_tanh(self):
        # t1 is (3, 2)
        t1 = Tensor([[1, 2], [3, 4], [5, 6]], requires_grad=True)

        # t2 is a (2, 1)
        t2 = tanh(t1)

        np.testing.assert_array_equal(t2.data, np.tanh(t1.data))