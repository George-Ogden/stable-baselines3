import torch
import numpy as np

def support_to_value(coefficients: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    """convert support (or supports) to value (or values)

    Args:
        coefficients (torch.Tensor): logits along the supports
        support (torch.Tensor): values to weight the support

    Returns:
        torch.Tensor: values
    """
    if support.ndim == 1:
        support = support.unsqueeze(0).expand(coefficients.size(0), -1)
    assert support.ndim == 2, "support must be 1 or 2 dimensional"
    assert support.size(-1) == coefficients.size(-1), "support and coefficients must match"
    assert torch.allclose(coefficients.sum(dim=-1), torch.tensor(1.)), "coefficients must sum to 1"
    return torch.sum(support * coefficients, dim=-1)

def value_to_support(values: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    """convert values to supports

    Args:
        values (torch.Tensor): values to convert to support
        support (torch.Tensor): values to weight the support

    Returns:
        torch.Tensor: support
    """
    assert support.dtype == values.dtype, "values and support must have the same dtype"
    if support.ndim == 1:
        support = support.unsqueeze(0).expand(len(values), -1)
    assert support.ndim == 2, "support must be 1 or 2 dimensional"
    assert values.ndim == 1, "values must be 1 dimensional"
    assert support.size(-2) == values.size(-1), "support and values must match"
    values = values.view(-1, 1)
    upper_bounds = torch.clamp(torch.searchsorted(support, values, side='left'), 1, support.size(-1) - 1)
    lower_bounds = upper_bounds - 1
    additional_indices = torch.arange(support.size(0), dtype=torch.int32)
    lower_bounds = torch.stack((additional_indices, lower_bounds.squeeze(dim=-1)), dim=-1)
    upper_bounds = torch.stack((additional_indices, upper_bounds.squeeze(dim=-1)), dim=-1)
    values = values.squeeze(dim=-1)
    interpolation = (values - support[tuple(lower_bounds.t())]) / (
        support[tuple(upper_bounds.t())] - support[tuple(lower_bounds.t())]
    )
    lower_bounds = lower_bounds[:, 1]
    upper_bounds = upper_bounds[:, 1]
    interpolation = interpolation.unsqueeze(dim=-1)
    support = torch.nn.functional.one_hot(
        lower_bounds,
        num_classes=support.size(-1)
    ) * (1 - interpolation) + torch.nn.functional.one_hot(
        upper_bounds,
        num_classes=support.size(-1)
    ) * interpolation
    return support
