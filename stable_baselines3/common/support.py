import torch.nn.functional as F
import torch as th

def support_to_value(coefficients: th.Tensor, support: th.Tensor) -> th.Tensor:
    """convert support (or supports) to value (or values)

    Args:
        coefficients (th.Tensor): logits along the supports
        support (th.Tensor): values to weight the support

    Returns:
        th.Tensor: values
    """
    if support.ndim == 1:
        support = th.tile(
            support.unsqueeze(0),
            [coefficients.shape[-2], 1]
        )
    assert support.ndim == 2, "support must be 1 or 2 dimensional"
    assert support.size(-1) == coefficients.size(-1), "support and coefficients must match"
    assert th.allclose(coefficients.sum(dim=-1), th.tensor(1.)), "coefficients must sum to 1"
    return th.sum(support * coefficients, dim=-1)

def value_to_support(values: th.Tensor, support: th.Tensor) -> th.Tensor:
    """convert values to supports

    Args:
        values (th.Tensor): values to convert to support
        support (th.Tensor): values to weight the support

    Returns:
        th.Tensor: support
    """
    assert support.dtype == values.dtype, "values and support must have the same dtype"
    if support.ndim == 1:
        support = support.unsqueeze(0).expand(len(values), -1)
    assert support.ndim == 2, "support must be 1 or 2 dimensional"
    assert values.ndim == 1, "values must be 1 dimensional"
    assert support.size(-2) == values.size(-1), "support and values must match"
    values = values.view(-1, 1)
    upper_bounds = th.clamp(th.searchsorted(support, values, side='left'), 1, support.size(-1) - 1)
    lower_bounds = upper_bounds - 1
    additional_indices = th.arange(support.size(0), dtype=th.int32)
    lower_bounds = th.stack((additional_indices, lower_bounds.squeeze(dim=-1)), dim=-1)
    upper_bounds = th.stack((additional_indices, upper_bounds.squeeze(dim=-1)), dim=-1)
    values = values.squeeze(dim=-1)
    interpolation = (values - support[tuple(lower_bounds.t())]) / (
        support[tuple(upper_bounds.t())] - support[tuple(lower_bounds.t())]
    )
    lower_bounds = lower_bounds[:, 1]
    upper_bounds = upper_bounds[:, 1]
    interpolation = interpolation.unsqueeze(dim=-1)
    support = F.one_hot(
        lower_bounds,
        num_classes=support.size(-1)
    ) * (1 - interpolation) + F.one_hot(
        upper_bounds,
        num_classes=support.size(-1)
    ) * interpolation
    return support

EPSILON = 0.001

def scale_values(values: th.Tensor) -> th.Tensor:
    """scale targets using an invertible transform 
    $h(x) = sign(x)(\sqrt{|x| + 1} - 1 + \epsilon x)$, where $\epsilon = 0.001$

    Args:
        values (th.Tensor): values to convert

    Returns:
        th.Tensor: converted values
    """
    assert isinstance(values, th.Tensor), "values must be a tensor"
    return th.sign(values) * (
        th.sqrt(
            th.abs(values) + 1
        ) - 1
    ) + EPSILON * values

def inverse_scale_values(values: th.Tensor) -> th.Tensor:
    """invert the scaling transform

    Args:
        values (th.Tensor): values to invert

    Returns:
        th.Tensor: inverted values
    """
    assert isinstance(values, th.Tensor), "values must be a tensor"
    return th.sign(values) * (
        (
            (
                th.sqrt(
                    1 + 4 * EPSILON * (th.abs(values) + 1 + EPSILON)
                ) - 1
            ) / (2 * EPSILON)
        ) ** 2 - 1
    )