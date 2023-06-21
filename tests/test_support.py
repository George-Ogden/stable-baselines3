import torch as th
import numpy as np

import pytest

from stable_baselines3.common.support import (
    support_to_value,
    value_to_support,
    scale_values,
    inverse_scale_values
)

def test_support_to_value():
    # test with a single support
    support = th.tensor([1, 2, 3], dtype=th.float32)
    coefficients = th.tensor([[0.1, 0.3, 0.6], [0.4, 0.6, 0.]], dtype=th.float32)
    expected_output = th.tensor([2.5, 1.6], dtype=th.float32)

    output = support_to_value(coefficients, support)
    assert np.allclose(output.numpy(), expected_output.numpy())

def test_batch_support_to_value():
    # test with multiple supports
    supports = th.tensor([[1, 2, 3], [2, 3, 4]], dtype=th.float32)
    coefficients = th.tensor([[0.1, 0.3, 0.6], [0.4, 0.6, 0.]], dtype=th.float32)
    expected_output = th.tensor([2.5, 2.6], dtype=th.float32)

    output = support_to_value(coefficients, supports)
    assert np.allclose(output.numpy(), expected_output.numpy())

def test_value_to_support():
    # test with a single support
    support = th.tensor([1, 2, 3], dtype=th.float32)
    values = th.tensor([2.5, 1.6], dtype=th.float32)
    expected_output = th.tensor([[0, .5, .5], [.4, .6, 0.]], dtype=th.float32)

    output = value_to_support(values, support)
    assert np.allclose(output.numpy(), expected_output.numpy())

def test_batch_value_to_support():
    # test with multiple supports
    support = th.tensor([[1, 2, 3], [2, 3, 4]], dtype=th.float32)
    values = th.tensor([2.5, 2.6], dtype=th.float32)
    expected_output = th.tensor([[0, .5, .5], [.4, .6, 0.]], dtype=th.float32)

    output = value_to_support(values, support)
    assert np.allclose(output.numpy(), expected_output.numpy())

scaling_test_data = [
    (0, 0),
    (2, 0.734),
    (-2, -0.734),
    (100, 9.150),
    (-100, -9.150),
]

@pytest.mark.parametrize("value, expected", scaling_test_data)
def test_value_scaling(value: float, expected: float):
    assert np.allclose(
        scale_values(th.Tensor([value])).numpy(),
        expected,
        atol=1e-3
    )

@pytest.mark.parametrize("value, expected", scaling_test_data)
def test_value_inverse_scaling(value: float, expected: float):
    assert np.allclose(
        inverse_scale_values(
            scale_values(th.Tensor([value]))
        ).numpy(),
        value,
        atol=1e-3
    )