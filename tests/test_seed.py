import os
import random

import numpy as np

from ecommerce_buy_predictor.config import settings
from ecommerce_buy_predictor.seed import set_global_seeds


def test_returns_the_applied_seed():
    assert set_global_seeds(123) == 123


def test_defaults_to_settings_seed():
    assert set_global_seeds() == settings.random_seed


def test_seeds_python_and_numpy_globals():
    set_global_seeds(7)
    py_first = [random.random() for _ in range(5)]
    np_first = np.random.rand(5)

    set_global_seeds(7)
    py_second = [random.random() for _ in range(5)]
    np_second = np.random.rand(5)

    assert py_first == py_second
    assert np.array_equal(np_first, np_second)


def test_different_seeds_produce_different_streams():
    set_global_seeds(1)
    first = np.random.rand(5)
    set_global_seeds(2)
    second = np.random.rand(5)

    assert not np.array_equal(first, second)


def test_sets_pythonhashseed_env_var():
    set_global_seeds(99)
    assert os.environ["PYTHONHASHSEED"] == "99"
