"""Tensor-level F143 quantization and calibration, independent of model frontends."""

import math

import torch


def scale_for_maximum(maximum):
    return max(-32, min(31, math.ceil(math.log2(maximum / 240.0)))) if maximum else 0


def block_hessians(width, block_size, device):
    return [
        torch.zeros((min(block_size, width - start),) * 2, device=device)
        for start in range(0, width, block_size)
    ]


def accumulate_input_moments(values, hessians, total):
    rows = values.detach().float().reshape(-1, values.shape[-1])
    total.add_(rows.sum(0))
    start = 0
    for hessian in hessians:
        end = start + hessian.shape[0]
        block = rows[:, start:end]
        hessian.addmm_(block.T, block)
        start = end
    return rows


def f143_tensor_scale(values: torch.Tensor) -> int:
    maximum = values.abs().max().item()
    return scale_for_maximum(maximum)


def f143_scales_by_row_block(values: torch.Tensor, block_size: int) -> torch.Tensor:
    rows = values.shape[0]
    padded_rows = math.ceil(rows / block_size) * block_size
    if padded_rows != rows:
        values = torch.nn.functional.pad(values, (0, 0, 0, padded_rows - rows))
    maximum = values.abs().reshape(-1, block_size, values.shape[1]).amax(dim=(1, 2))
    scales = torch.where(
        maximum == 0.0,
        torch.zeros_like(maximum),
        torch.ceil(torch.log2(maximum / 240.0)),
    ).clamp(-32.0, 31.0)
    return scales.repeat_interleave(block_size)[:rows]


def project_f143(values: torch.Tensor, scale: torch.Tensor | float) -> torch.Tensor:
    output_dtype = values.dtype
    values = values.float()
    factor = torch.exp2(
        torch.as_tensor(-scale, device=values.device, dtype=torch.float32)
    )
    magnitude = values.abs() * factor
    subnormal = torch.round(magnitude * 1024.0).clamp(max=8.0) * 2.0**-10
    exponent = torch.floor(
        torch.log2(magnitude.clamp_min(torch.finfo(torch.float32).tiny))
    )
    unit = torch.exp2(exponent)
    mantissa = torch.round((magnitude / unit - 1.0) * 8.0)
    carry = mantissa == 8.0
    exponent += carry
    mantissa = torch.where(carry, torch.zeros_like(mantissa), mantissa)
    normal = (1.0 + mantissa / 8.0) * torch.exp2(exponent)
    projected = torch.where(magnitude < 2.0**-7, subnormal, normal).clamp(max=240.0)
    return torch.copysign(projected / factor, values).to(output_dtype)


def inverse_hessian_factor(hessian: torch.Tensor, damp: float) -> torch.Tensor:
    hessian = hessian.double()
    diagonal_mean = torch.diagonal(hessian).mean()
    hessian.diagonal().add_(
        max(torch.finfo(torch.float64).eps, damp * diagonal_mean.item())
    )
    try:
        return torch.linalg.cholesky(torch.linalg.inv(hessian), upper=True).float()
    except torch.linalg.LinAlgError:
        return torch.diag(torch.diagonal(hessian).rsqrt()).float()


def gptq_block(
    weight: torch.Tensor, inverse_factor: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor:
    width = weight.shape[1]

    working = weight.float().clone()
    output = torch.empty_like(working)
    for column in range(width):
        quantized = project_f143(working[:, column], scales)
        output[:, column] = quantized
        divisor = inverse_factor[column, column].clamp_min(
            torch.finfo(torch.float32).eps
        )
        error = (working[:, column] - quantized) / divisor
        working[:, column:].sub_(
            error[:, None] * inverse_factor[column, column:][None, :]
        )
    return output


def nearest_f143_weight(
    weight: torch.Tensor, block_size: int, tensor_scale: float | None = None
) -> torch.Tensor:
    if tensor_scale is not None:
        return project_f143(weight, tensor_scale)
    output = torch.empty_like(weight)
    for start in range(0, weight.shape[1], block_size):
        block = weight[:, start : start + block_size]
        scales = f143_scales_by_row_block(block, block_size)
        output[:, start : start + block_size] = project_f143(block, scales[:, None])
    return output


def equalize_layernorm_group(
    norm, consumers, moment, block_size, scale_limit, tensor_scales=False
):
    """Historical bounded channel equalization, optionally priced with tensor scales."""
    activation_rms = moment.sqrt().clamp_min(1e-8)
    weight_rms = torch.cat([consumer.weight for consumer in consumers])
    weight_rms = weight_rms.float().square().mean(dim=0).sqrt().clamp_min(1e-8)
    baseline = 0.0
    candidates = []
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        scale = activation_rms.pow(alpha) / weight_rms.pow(1.0 - alpha)
        scale /= (scale.min() * scale.max()).sqrt()
        scale.clamp_(1.0 / scale_limit, scale_limit)
        objective = 0.0
        for consumer in consumers:
            transformed = consumer.weight.float() * scale
            quantized = nearest_f143_weight(
                transformed,
                block_size,
                f143_tensor_scale(transformed) if tensor_scales else None,
            )
            error = transformed - quantized
            objective += torch.sum(error.square() * (moment / scale.square())).item()
            if alpha == 0.0:
                unscaled = nearest_f143_weight(
                    consumer.weight.float(),
                    block_size,
                    f143_tensor_scale(consumer.weight) if tensor_scales else None,
                )
                baseline += torch.sum(
                    (consumer.weight.float() - unscaled).square() * moment
                ).item()
        candidates.append((objective, alpha, scale))
    objective, alpha, scale = min(candidates, key=lambda candidate: candidate[0])
    norm.weight.div_(scale.to(norm.weight.dtype))
    norm.bias.div_(scale.to(norm.bias.dtype))
    for consumer in consumers:
        consumer.weight.mul_(scale.to(consumer.weight.dtype))
    return {
        "alpha": alpha,
        "objective_ratio": objective / baseline if baseline else 0.0,
        "scale_minimum": scale.min().item(),
        "scale_maximum": scale.max().item(),
    }


def reconstruct_linear(
    module: torch.nn.Linear,
    hessians: list[torch.Tensor],
    block_size: int,
    damp: float,
    algorithm: str,
    input_mean: torch.Tensor | None,
    tensor_scale: float | None = None,
) -> tuple[float, float, float]:
    original = module.weight.detach()
    reconstructed = torch.empty_like(original)
    nearest_objective = 0.0
    reconstructed_objective = 0.0
    for input_index, input_start in enumerate(range(0, module.in_features, block_size)):
        input_end = min(input_start + block_size, module.in_features)
        hessian = hessians[input_index]
        inverse_factor = (
            inverse_hessian_factor(hessian, damp) if algorithm == "gptq" else None
        )
        block = original[:, input_start:input_end]
        scales = (
            f143_scales_by_row_block(block, block_size)
            if tensor_scale is None
            else torch.full((module.out_features,), tensor_scale, device=block.device)
        )
        nearest = project_f143(block, scales[:, None])
        rebuilt = (
            gptq_block(block, inverse_factor, scales)
            if inverse_factor is not None
            else nearest
        )
        reconstructed[:, input_start:input_end] = rebuilt
        nearest_error = block.float() - nearest
        rebuilt_error = block.float() - rebuilt
        nearest_objective += torch.sum((nearest_error @ hessian) * nearest_error).item()
        reconstructed_objective += torch.sum(
            (rebuilt_error @ hessian) * rebuilt_error
        ).item()
    correction = torch.zeros(module.out_features, device=original.device)
    if input_mean is not None and module.bias is not None:
        correction = (original.float() - reconstructed.float()) @ input_mean
        module.bias.add_(correction.to(module.bias.dtype))
    module.weight.copy_(reconstructed)
    return (
        nearest_objective,
        reconstructed_objective,
        correction.square().mean().sqrt().item(),
    )
