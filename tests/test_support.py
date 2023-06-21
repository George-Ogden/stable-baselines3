import torch
import numpy as np

from stable_baselines3.common.support import support_to_value, value_to_support

def test_support_to_value():
    # test with a single support
    support = torch.tensor([1, 2, 3], dtype=torch.float32)
    coefficients = torch.tensor([[0.1, 0.3, 0.6], [0.4, 0.6, 0.]], dtype=torch.float32)
    expected_output = torch.tensor([2.5, 1.6], dtype=torch.float32)

    output = support_to_value(coefficients, support)
    assert np.allclose(output.numpy(), expected_output.numpy())

def test_batch_support_to_value():
    # test with multiple supports
    supports = torch.tensor([[1, 2, 3], [2, 3, 4]], dtype=torch.float32)
    coefficients = torch.tensor([[0.1, 0.3, 0.6], [0.4, 0.6, 0.]], dtype=torch.float32)
    expected_output = torch.tensor([2.5, 2.6], dtype=torch.float32)

    output = support_to_value(coefficients, supports)
    assert np.allclose(output.numpy(), expected_output.numpy())

def test_value_to_support():
    # test with a single support
    support = torch.tensor([1, 2, 3], dtype=torch.float32)
    values = torch.tensor([2.5, 1.6], dtype=torch.float32)
    expected_output = torch.tensor([[0, .5, .5], [.4, .6, 0.]], dtype=torch.float32)

    output = value_to_support(values, support)
    assert np.allclose(output.numpy(), expected_output.numpy())

def test_batch_value_to_support():
    # test with multiple supports
    support = torch.tensor([[1, 2, 3], [2, 3, 4]], dtype=torch.float32)
    values = torch.tensor([2.5, 2.6], dtype=torch.float32)
    expected_output = torch.tensor([[0, .5, .5], [.4, .6, 0.]], dtype=torch.float32)

    output = value_to_support(values, support)
    assert np.allclose(output.numpy(), expected_output.numpy())
