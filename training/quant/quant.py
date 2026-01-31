from accelerate.logging import get_logger

try:
    from ..constants import FINETRAINERS_LOG_LEVEL

    logger = get_logger("qvgen")  # pylint: disable=invalid-name
    logger.setLevel(FINETRAINERS_LOG_LEVEL)
except ImportError:
    from loguru import logger
import torch
from torch.utils.checkpoint import checkpoint
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import math
import gc

import torch.nn as nn
import torch
from typing import Tuple


class UniformQuantizer(nn.Module):
    def __init__(
        self,
        bit: int = 4,
        sym: bool = False,
        granularity: str = "per_channel",
        cali: str = "mse",
        group_size: int = -1,
        round_zero: bool = True,
        clip_ratio: float = 1,
        **kwargs,
    ) -> None:
        super(UniformQuantizer, self).__init__()

        self.bit = bit
        self.sym = sym
        self.granularity = granularity

        if self.sym:
            self.P = 2 ** (self.bit - 1) - 1
            self.N = -(2 ** (self.bit - 1))
        else:
            self.P = 2**self.bit - 1
            self.N = 0.0

        self.group_size = group_size
        self.cali = cali
        self.round_func = torch.round
        self.round_zero = round_zero
        self.clip_ratio = clip_ratio
        self.max_x = None
        self.min_x = None
        self.scale = None
        self.zero_point = None
        self.init = False

    def reshape(self, x: torch.Tensor) -> torch.Tensor:
        # act: [B, F * H * W, C]
        shape = {
            "per_tensor": (1, -1),  # w, act
            "per_group": (-1, self.group_size),  # w
            "per_channel": (x.shape[0], -1),  # w
            "per_token": (-1, x.shape[-1]),  # act
        }
        return x.reshape(shape[self.granularity])

    def get_minmax_range(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        max_x = x.amax(dim=-1, keepdim=True) * self.clip_ratio
        min_x = x.amin(dim=-1, keepdim=True) * self.clip_ratio
        return (min_x, max_x)

    def get_mse_range(
        self, x: torch.Tensor, grid: int = 100, norm: int = 2.4, max_shrink: int = 0.8
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.float()
        dev = x.device
        dtype = x.dtype
        org_min_x, org_max_x = self.get_minmax_range(x)
        shape = org_max_x.shape
        (best_min_x, best_max_x) = torch.full(
            shape, float("inf"), device=dev, dtype=dtype
        ), torch.full(shape, float("inf"), device=dev, dtype=dtype)
        best = torch.full(shape, float("inf"), device=dev, dtype=dtype)

        for i in range(int(max_shrink * grid)):
            p = 1 - i / grid

            min_x = p * org_min_x
            max_x = p * org_max_x

            scales, zeros = self.get_qparams(min_x, max_x)
            q_x = self.quant_dequant(x, scales, zeros)

            q_x -= x
            q_x.abs_()
            q_x.pow_(norm)
            err = torch.sum(q_x, 1, keepdim=True)

            if torch.isnan(err).any():
                raise ValueError("NaN detected in MSE calibration")

            tmp = err < best

            if torch.any(tmp):
                best[tmp] = err[tmp]
                best_min_x[tmp] = min_x[tmp]
                best_max_x[tmp] = max_x[tmp]

        return (best_min_x, best_max_x)

    def get_range(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.cali == "mse":
            return self.get_mse_range(x)
        elif self.cali == "minmax":
            return self.get_minmax_range(x)

    def get_qparams(
        self, min_x: torch.Tensor, max_x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.sym:
            abs_max_x = torch.max(min_x.abs(), max_x.abs()).clamp(min=1e-5)
            scale = abs_max_x / self.P
            zero_point = None
        else:
            scale = ((max_x - min_x) / self.P).clamp_min(1e-9)
            zero_point = (-torch.round(min_x / scale)).clamp(self.N, self.P)
            if not self.round_zero:
                zero_point = -min_x / scale
        return (scale, zero_point)

    def quant(
        self, x: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor
    ) -> torch.Tensor:
        if zero_point is None:
            return torch.clamp(self.round_func(x / scale), self.N, self.P)
        elif self.round_zero:
            return torch.clamp(self.round_func(x / scale) + zero_point, self.N, self.P)
        else:
            return torch.clamp(self.round_func(x / scale + zero_point), self.N, self.P)

    def dequant(
        self, x: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor
    ) -> torch.Tensor:
        if zero_point is None:
            return x * scale
        else:
            return scale * (x - zero_point)

    def quant_dequant(
        self, x: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor
    ) -> torch.Tensor:
        return self.dequant(self.quant(x, scale, zero_point), scale, zero_point)

    def init_qparams(self, x: torch.Tensor) -> None:
        _x = self.reshape(x)
        self.min_x, self.max_x = self.get_range(_x)
        scale, zero_point = self.get_qparams(self.min_x, self.max_x)
        del self.scale
        self.register_buffer("scale", scale)
        del self.zero_point
        self.register_buffer(
            "zero_point",
            torch.tensor(
                zero_point, requires_grad=False, device=x.device, dtype=x.dtype
            ),
        )
        self.init = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.init:
            self.init_qparams(x)
        org_shape = x.shape
        x = self.reshape(x)
        qdq_x = self.quant_dequant(x, self.scale, self.zero_point)
        return qdq_x.reshape(org_shape)

    def build(self) -> None:
        pass

    def extra_repr(self) -> str:
        return f"bit={self.bit}, sym={self.sym}, granularity={self.granularity}, cali={self.cali}, group_size={self.group_size}, round_zero={self.round_zero}"


class LSQQuantizer(UniformQuantizer):
    def __init__(self, use_grad_scaling: bool = False, **kwargs) -> None:
        super(LSQQuantizer, self).__init__(**kwargs)
        self.use_grad_scaling = use_grad_scaling
        self.round_func_build_str = kwargs.get("round_func", "STE")
        self.round_func_str = "STE"
        self.timesteps_ste = kwargs.get("timesteps_ste", 50)
        self.idx_ste = 0  # for inference

    def build(self) -> None:
        assert self.init
        self.round_func_str = self.round_func_build_str
        if self.round_func_str == "STE":
            self.round_func = lambda x: (x.round() - x).detach() + x
        elif self.round_func_str == "AdaSTE":
            self.round_func = AdaSTE.apply
        elif self.round_func_str == "LearnableAdaSTE":
            self.round_func = LearnableAdaSTE.apply
            self.ps = nn.Parameter(torch.tensor([5.0] * self.timesteps_ste))
        else:
            raise ValueError("Invalid round function")
        scale = self.scale
        del self.scale
        self.scale = nn.Parameter(scale)

    def grad_scale(self, t: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return (t - (t * scale)).detach() + (t * scale)

    def quant(
        self, x: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor
    ) -> torch.Tensor:
        if self.round_func_str == "STE":
            return super().quant(x, scale, zero_point)
        elif self.round_func_str == "AdaSTE":
            raise NotImplementedError("AdaSTE is not implemented")
        elif self.round_func_str == "LearnableAdaSTE":
            if zero_point is None:
                return torch.clamp(
                    self.round_func(x / scale, F.sigmoid(self.ps[self.idx_ste])),
                    self.N,
                    self.P,
                )
            elif self.round_zero:
                return torch.clamp(
                    self.round_func(x / scale, F.sigmoid(self.ps[self.idx_ste]))
                    + zero_point,
                    self.N,
                    self.P,
                )
            else:
                return torch.clamp(
                    self.round_func(
                        x / scale + zero_point, F.sigmoid(self.ps[self.idx_ste])
                    ),
                    self.N,
                    self.P,
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.init:
            self.init_qparams(x)
        org_shape = x.shape
        x = self.reshape(x)
        if self.use_grad_scaling:
            grad_factor = 1.0 / (x.numel() / x.shape[0] * self.P) ** 0.5
        else:
            grad_factor = 1.0
        scale = self.grad_scale(self.scale, grad_factor)
        zero_point = self.grad_scale(self.zero_point, grad_factor)
        qdq_x = self.quant_dequant(x, scale, zero_point)
        return qdq_x.reshape(org_shape)

    def set_idx_ste(self, t) -> None:
        if isinstance(t, torch.Tensor):
            t = t[0].item()
        self.idx_ste = int(int(t) % self.timesteps_ste)

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, use_grad_scaling={self.use_grad_scaling}, learn_zero={self.learn_zero if hasattr(self, 'learn_zero') else False}"


class LSQPlusQuantizer(LSQQuantizer):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.round_zero = False
        self.learn_zero = True

    def build(self) -> None:
        super().build()
        zero_point = self.zero_point
        del self.zero_point
        self.zero_point = nn.Parameter(zero_point)


def low_rank_decomposition(weight, reduced_rank=0.15):
    """Compute a low-rank SVD-based decomposition of a 2D weight matrix.

    Args:
        weight: 2D weight matrix tensor.
        reduced_rank: Target rank (int) or ratio if <= 1.0; -1 uses full rank.

    Returns:
        Tuple of (L, R) factors such that L @ R approximates weight.
    """
    matrix_dimension = len(weight.size())
    assert matrix_dimension == 2, "Only Support 2D matrix"
    if reduced_rank == -1:
        reduced_rank = weight.shape[1]
    U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
    L = U @ (torch.sqrt(torch.diag(S)[:, 0:reduced_rank]))
    R = torch.sqrt(torch.diag(S)[0:reduced_rank, :]) @ Vh

    return L, R


class ProgressiveLoRAQuantizer(LSQPlusQuantizer):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.rank = kwargs.get("rank", 32)
        self.iters = kwargs.get("iters", 1)
        self.cur_alpha_strategy = kwargs.get("cur_alpha_strategy", "cosine")
        self.alpha_T = kwargs.get("alpha_T", 200)
        self.warm_up_T = kwargs.get("warm_up", 0)
        self.warm_up = self.warm_up_T > 0
        self.resume = kwargs.get("resume", False)
        self.valid = False
        assert not self.resume or (
            self.cali == "minmax" and self.iters == 1
        ), "resume only support minmax"

    def set_alpha_L_R(self, train_iter) -> None:
        train_iter = train_iter - self.warm_up_T
        if train_iter < 0:
            return
        elif train_iter == 0:
            # L, R = self.L.data.clone(), self.R.data.clone()
            # del self.L, self.R
            # self.register_buffer("L", L)
            # self.register_buffer("R", R)
            self.warm_up = False
        elif not hasattr(self, "L") or not hasattr(self, "R"):
            return
        _train_iter = train_iter % self.alpha_T
        if _train_iter == 0 and train_iter > 0:
            if self.rank >= 2:
                # L, R = self.L, self.R
                # del self.L, self.R
                # self.register_buffer("L", L[:, : self.rank // 2].contiguous())
                # self.register_buffer("R", R[: self.rank // 2, :].contiguous())
                self.rank = self.rank // 2
                with torch.no_grad():
                    # logger.info(f"Decomposing L and R (rank {self.rank})")
                    tmp_L, tmp_R = self.L[:, : self.rank], self.R[: self.rank, :]
                    org_w = tmp_L @ tmp_R
                    del tmp_L, tmp_R
                    new_L, new_R = low_rank_decomposition(
                        org_w.to(torch.float32), reduced_rank=self.rank
                    )
                    # assign new_L and new_R to self.L and self.R
                    self.L[:, : self.rank] = new_L.to(self.L.dtype).contiguous()
                    self.R[: self.rank, :] = new_R.to(self.R.dtype).contiguous()
                    del new_L, new_R, org_w
                    torch.cuda.empty_cache()
                    gc.collect()
            else:
                del self.L, self.R
                self.rank = 0
        if self.cur_alpha_strategy == "cosine":
            self.alpha = (1 + math.cos(math.pi * _train_iter / self.alpha_T)) / 2
        elif self.cur_alpha_strategy == "linear":
            self.alpha = 1 - _train_iter / self.alpha_T
        else:
            raise ValueError("Invalid alpha strategy")

    def weight_quant_fn(self, x: torch.Tensor) -> torch.Tensor:
        org_shape = x.shape
        x = self.reshape(x)
        min_x, max_x = self.get_range(x)
        scale, zero_point = self.get_qparams(min_x, max_x)
        qdq_x = self.quant_dequant(x, scale, zero_point)
        qdq_x = qdq_x.reshape(org_shape)
        return qdq_x, scale, zero_point

    def quant_first_iter(self, weight, L, R, reduced_rank):
        low_rank_product = L @ R if torch.is_tensor(L) else 0
        residual = weight - low_rank_product
        quant_w, scale, zero_point = self.weight_quant_fn(residual)
        if not self.resume:
            output = low_rank_decomposition(weight - quant_w, reduced_rank=reduced_rank)
            L, R = output[0], output[1]
            final_residual = weight - quant_w - L @ R
        else:
            L = torch.randn(
                quant_w.shape[0], self.rank, dtype=quant_w.dtype, device=quant_w.device
            )
            R = torch.randn(
                self.rank, quant_w.shape[1], dtype=quant_w.dtype, device=quant_w.device
            )
            final_residual = None
        return weight, scale, zero_point, L, R, final_residual, quant_w

    def init_qparams(self, x: torch.Tensor) -> None:
        # _x = self.reshape(x)
        # self.min_x, self.max_x = self.get_range(_x)
        # scale, zero_point = self.get_qparams(self.min_x, self.max_x)
        L, R = 0, 0
        weights = x  # for weights quantization
        # init_q_w = None
        for i in range(self.iters):
            (
                weights,
                scale,
                zero_point,
                L,
                R,
                final_residual,
                quant_w,
            ) = self.quant_first_iter(weights, L, R, self.rank)
            # if init_q_w == None:
            #         init_q_w = quant_w
            if ((i + 1) % 10 == 0 or i == 0 or i == self.iters - 1) and not self.resume:
                logger.info(
                    f"Iter {i}: {torch.mean(final_residual ** 2)}|"
                    f" {torch.mean((weights - quant_w) ** 2)}"
                )
        del final_residual
        del self.scale
        self.register_buffer("scale", scale)
        del self.zero_point
        self.register_buffer(
            "zero_point",
            torch.tensor(
                zero_point, requires_grad=False, device=x.device, dtype=x.dtype
            ),
        )
        self.register_buffer("L", L)
        self.register_buffer("R", R)
        self.init = True

    def build(
        self,
    ) -> None:
        super().build()
        if hasattr(self, "L") and hasattr(self, "R"):
            self.L = nn.Parameter(self.L)
            self.R = nn.Parameter(self.R)  # for init. training

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        qdq_x = super().forward(x)
        if hasattr(self, "L") and hasattr(self, "R"):
            if self.valid:
                return qdq_x
            if not self.warm_up and hasattr(self, "alpha"):
                rank = self.L.shape[1]
                alpha = torch.tensor(
                    [self.alpha] * self.rank + [0] * (rank - self.rank)
                ).to(x.device, x.dtype)
                if self.rank >= 2:
                    alpha[: self.rank // 2] = 1
                qdq_x = qdq_x + (self.L * alpha.unsqueeze(0)) @ self.R
            else:
                qdq_x = qdq_x + self.L @ self.R
        return qdq_x

    def validation(self, valid) -> None:
        self.valid = valid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return checkpoint(self._forward, x, use_reentrant=False)


class DynamicQuantizer(UniformQuantizer):
    def __init__(
        self,
        bit: int = 4,
        sym: bool = False,
        granularity: str = "per_token",
        cali: str = "minmax",
        group_size: int = -1,
        round_zero: bool = True,
        clip_ratio: float = 0.95,
        **kwargs,
    ) -> None:
        assert cali == "minmax"
        super().__init__(
            bit, sym, granularity, cali, group_size, round_zero, clip_ratio
        )
        del self.max_x
        del self.min_x
        del self.scale
        del self.zero_point
        self.round_func_build_str = kwargs.get("round_func", "STE")
        self.round_func_str = "STE"
        self.timesteps_ste = kwargs.get("timesteps_ste", 50)
        self.idx_ste = 0  # for inference

    def quant(
        self, x: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor
    ) -> torch.Tensor:
        self.round_func_str = self.round_func_build_str
        if self.round_func_str == "STE":
            return super().quant(x, scale, zero_point)
        elif self.round_func_str == "AdaSTE":
            raise NotImplementedError("AdaSTE is not implemented")
        elif self.round_func_str == "LearnableAdaSTE":
            if zero_point is None:
                return torch.clamp(
                    self.round_func(x / scale, F.sigmoid(self.ps[self.idx_ste])),
                    self.N,
                    self.P,
                )
            elif self.round_zero:
                return torch.clamp(
                    self.round_func(x / scale, F.sigmoid(self.ps[self.idx_ste]))
                    + zero_point,
                    self.N,
                    self.P,
                )
            else:
                return torch.clamp(
                    self.round_func(
                        x / scale + zero_point, F.sigmoid(self.ps[self.idx_ste])
                    ),
                    self.N,
                    self.P,
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        org_shape = x.shape
        x = self.reshape(x)
        min_x, max_x = self.get_range(x)
        scale, zero_point = self.get_qparams(min_x, max_x)
        qdq_x = self.quant_dequant(x, scale, zero_point)
        return qdq_x.reshape(org_shape)

    def build(self) -> None:
        self.round_func_str = self.round_func_build_str
        if self.round_func_str == "STE":
            self.round_func = lambda x: (x.round() - x).detach() + x
        elif self.round_func_str == "AdaSTE":
            self.round_func = AdaSTE.apply
        elif self.round_func_str == "LearnableAdaSTE":
            self.round_func = LearnableAdaSTE.apply
            self.ps = nn.Parameter(torch.tensor([5.0] * self.timesteps_ste))
        else:
            raise ValueError("Invalid round function")

    def set_idx_ste(self, t) -> None:
        if isinstance(t, torch.Tensor):
            t = t[0].item()
        self.idx_ste = int(int(t) % self.timesteps_ste)


class LearnableClippedDynamicQuantizer(UniformQuantizer):
    def __init__(
        self,
        bit: int = 4,
        sym: bool = False,
        granularity: str = "per_token",
        cali: str = "minmax",
        group_size: int = -1,
        round_zero: bool = True,
        clip_ratio: float = 0.95,
        timesteps: int = 50,
        **kwargs,
    ) -> None:
        assert cali == "minmax"
        super().__init__(
            bit, sym, granularity, cali, group_size, round_zero, clip_ratio
        )
        del self.max_x
        del self.min_x
        del self.scale
        del self.zero_point
        self.logit = lambda x: torch.log(x / (1 - x))
        t_clips = torch.torch.full((timesteps,), clip_ratio)
        self.timesteps = timesteps
        self.register_buffer("upper", self.logit(t_clips))
        self.register_buffer("lower", self.logit(t_clips))
        self.sigmoid = nn.Sigmoid()
        self.idx = -1
        self.auto = False
        self.round_func_build_str = kwargs.get("round_func", "STE")
        self.round_func_str = "STE"
        self.timesteps_ste = kwargs.get("timesteps_ste", 50)
        self.idx_ste = 0  # for inference

    def quant(
        self, x: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor
    ) -> torch.Tensor:
        self.round_func_str = self.round_func_build_str
        if self.round_func_str == "STE":
            return super().quant(x, scale, zero_point)
        elif self.round_func_str == "AdaSTE":
            raise NotImplementedError("AdaSTE is not implemented")
        elif self.round_func_str == "LearnableAdaSTE":
            if zero_point is None:
                return torch.clamp(
                    self.round_func(x / scale, F.sigmoid(self.ps[self.idx_ste])),
                    self.N,
                    self.P,
                )
            elif self.round_zero:
                return torch.clamp(
                    self.round_func(x / scale, F.sigmoid(self.ps[self.idx_ste]))
                    + zero_point,
                    self.N,
                    self.P,
                )
            else:
                return torch.clamp(
                    self.round_func(
                        x / scale + zero_point, F.sigmoid(self.ps[self.idx_ste])
                    ),
                    self.N,
                    self.P,
                )

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        org_shape = x.shape
        x = self.reshape(x)
        min_x, max_x = self.get_range(x)
        if self.auto:
            # print(f"Auto: {self.idx}")
            self.idx -= 1
            if self.idx == 0:
                self.auto = False
            assert self.idx >= 0, "Index should be greater than 0"
        min_x = (
            self.sigmoid(self.lower[self.idx].to(device=x.device, dtype=x.dtype))
            * min_x
        )
        max_x = (
            self.sigmoid(self.upper[self.idx].to(device=x.device, dtype=x.dtype))
            * max_x
        )
        scale, zero_point = self.get_qparams(min_x, max_x)
        qdq_x = self.quant_dequant(x, scale, zero_point)
        return qdq_x.reshape(org_shape)

    def set_idx(self, t) -> None:
        if isinstance(t, torch.Tensor):
            t = t[0].item()
        self.idx = int(int(t) % self.timesteps)

    def set_idx_ste(self, t) -> None:
        if isinstance(t, torch.Tensor):
            t = t[0].item()
        self.idx_ste = int(int(t) % self.timesteps_ste)

    def set_auto(self, auto: bool) -> None:
        self.auto = auto
        self.idx = self.timesteps

    def build(self) -> None:
        self.round_func_str = self.round_func_build_str
        if self.round_func_str == "STE":
            self.round_func = lambda x: (x.round() - x).detach() + x
        elif self.round_func_str == "AdaSTE":
            self.round_func = AdaSTE.apply
        elif self.round_func_str == "LearnableAdaSTE":
            self.round_func = LearnableAdaSTE.apply
            self.ps = nn.Parameter(torch.tensor([5.0] * self.timesteps_ste))
        else:
            raise ValueError("Invalid round function")
        upper = self.upper
        lower = self.lower
        del self.upper
        del self.lower
        self.upper = nn.Parameter(upper)
        self.lower = nn.Parameter(lower)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return checkpoint(self._forward, x, use_reentrant=False)


class AdaSTE(Function):
    @staticmethod
    def forward(ctx, input, o):
        ctx.save_for_backward(input, o)
        out = torch.round(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # the grad_output has already been passed through the clip function
        input, o = ctx.saved_tensors
        interval = 0.1
        frac = input - torch.floor(input) - 0.5
        # f = 1 / 2 * torch.sign(frac) * torch.pow(torch.abs(2 * frac), o)
        abs_frac = torch.abs(frac)
        temp = o * torch.pow(2 * abs_frac, o - 1)
        temp[(frac >= -interval) & (interval >= frac)] = (
            1 / 2 * torch.pow(2 * interval, o) / interval
        )
        grad_input = temp * grad_output.clone()
        return grad_input, None


class LearnableAdaSTE(Function):
    @staticmethod
    def forward(ctx, input, o):
        ctx.save_for_backward(input, o)
        out = torch.round(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # the grad_output has already been passed through the clip function
        input, o = ctx.saved_tensors
        interval = 0.1
        frac = input - torch.floor(input) - 0.5
        # f = 1 / 2 * torch.sign(frac) * torch.pow(torch.abs(2 * frac), o)
        abs_frac = torch.abs(frac)
        temp1 = o * torch.pow(2 * abs_frac, o - 1)
        temp1[(frac >= -interval) & (interval >= frac)] = (
            1 / 2 * torch.pow(2 * interval, o) / interval
        )
        grad_input = temp1 * grad_output.clone()
        # fac \in [-0.5, 0.5], o \in [1, +\inf)
        temp2 = (
            1
            / 2
            * torch.sign(frac)
            * torch.pow(2 * abs_frac, o)
            * torch.log(2 * abs_frac.clamp_min(1e-5))
        )
        grad_o = torch.sum(temp2 * grad_output.clone())
        return grad_input, grad_o


QuantizerMap = {
    "uniform": UniformQuantizer,
    "lsq": LSQQuantizer,
    "lsq+": LSQPlusQuantizer,
    "dynamic": DynamicQuantizer,
    "learnable_clipped_dynamic": LearnableClippedDynamicQuantizer,
    "progressive_lora": ProgressiveLoRAQuantizer,
}


class QuantLinear(nn.Module):
    def __init__(
        self,
        layer: nn.Linear,
        quantizer_type: dict = {"w": "uniform", "act": "uniform"},
        q_params: dict = {"w": {}, "act": {}},
    ) -> None:
        super().__init__()
        self.kwd_func = F.linear
        self.register_parameter("w", layer.weight)
        if layer.bias is not None:
            self.register_parameter("b", layer.bias)
        else:
            self.b = None
        self.use_wq = False
        self.use_aq = False
        self.wquantizer: UniformQuantizer = QuantizerMap[quantizer_type["w"]](
            **q_params["w"]
        )
        self.aquantizer: UniformQuantizer = QuantizerMap[quantizer_type["act"]](
            **q_params["act"]
        )
        self.extra_repr_prefix = layer.extra_repr()

    def _forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if self.use_aq:
            x = self.aquantizer(x)
        if self.use_wq:
            w = self.wquantizer(self.w)
        else:
            w = self.w
        b = self.b
        w = w.to(x.device)
        if type(b) == torch.Tensor:
            b = b.to(x.device)
        x = self.kwd_func(x, w, b)
        return x

    def set_quant_state(self, use_wq: bool = False, use_aq: bool = False) -> None:
        self.use_wq = use_wq
        self.use_aq = use_aq

    def extra_repr(self) -> str:
        return f"{self.extra_repr_prefix}, use_aq={self.use_aq}, use_wq={self.use_wq}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return checkpoint(self._forward, x, use_reentrant=False)


class QuantLoRALinear(nn.Module):
    def __init__(
        self,
        layer: nn.Linear,
        quantizer_type: dict = {"w": "uniform", "act": "uniform"},
        q_params: dict = {"w": {}, "act": {}},
        lora_params: dict = {"rank": 64, "dropout": 0.0, "alpha": 64},
    ) -> None:
        super().__init__()
        self.kwd_func = F.linear
        self.register_parameter("w", layer.weight)
        if layer.bias is not None:
            self.register_parameter("b", layer.bias)
        else:
            self.b = None
        self.use_wq = False
        self.use_aq = False
        self.wquantizer: UniformQuantizer = QuantizerMap[quantizer_type["w"]](
            **q_params["w"]
        )
        self.aquantizer: UniformQuantizer = QuantizerMap[quantizer_type["act"]](
            **q_params["act"]
        )
        dropout = lora_params["dropout"]
        r = lora_params["rank"]
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()
        if isinstance(layer, nn.Linear):
            self.loraA = nn.Linear(
                layer.in_features, r, False, layer.weight.device, layer.weight.dtype
            )
            self.loraB = nn.Linear(
                r, layer.out_features, False, layer.weight.device, layer.weight.dtype
            )

        nn.init.kaiming_uniform_(self.loraA.weight, a=math.sqrt(5))
        nn.init.zeros_(self.loraB.weight)
        self.alpha = lora_params["alpha"]
        self.r = r
        self.extra_repr_prefix = layer.extra_repr()

    def _forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if hasattr(self, "loraA"):
            eye = torch.eye(self.w.shape[1], device=x.device, dtype=x.dtype)
            lora_w = self.loraB(self.loraA(self.dropout(eye)))
            lora_w = lora_w.T
            w = self.w + lora_w * self.alpha / self.r
        else:
            w = self.w
        if self.use_aq:
            x = self.aquantizer(x)
        if self.use_wq:
            w = self.wquantizer(w)
        else:
            pass
        b = self.b
        w = w.to(x.device)
        if type(b) == torch.Tensor:
            b = b.to(x.device)
        x = self.kwd_func(x, w, b)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return checkpoint(self._forward, x, use_reentrant=False)

    def set_quant_state(self, use_wq: bool = False, use_aq: bool = False) -> None:
        self.use_wq = use_wq
        self.use_aq = use_aq

    def extra_repr(self) -> str:
        return f"{self.extra_repr_prefix}, use_aq={self.use_aq}, use_wq={self.use_wq}"

    def merge_lora(self) -> None:
        assert hasattr(
            self, "loraA"
        ), "LoRA weights have been merged into the latent weights."
        eye = torch.eye(self.w.shape[1], device=self.w.device)
        lora_w = self.loraB(self.loraA(eye))
        lora_w = lora_w.T
        self.w.data = self.w + lora_w * self.alpha / self.r
        del self.loraA
        del self.loraB


def replace_linear(
    model: nn.Module,
    quantizer_type: dict = {"w": "uniform", "act": "uniform"},
    q_params: dict = {"w": {}, "act": {}},
    lora_params: dict = {"rank": 32, "dropout": 0.0},
    use_lora: bool = False,
    wo_quant_adaln: bool = False,
    wo_quant_only_time_embedding: bool = False,
    wo_quant_only_norm: bool = False,
    random_list: list = [],
    count: int = -1,
    parent_name: str = "transformer",
) -> nn.Module:
    """Recursively replace Linear layers with quantized variants.

    Args:
        model: Root module to modify in-place.
        quantizer_type: Quantizer type mapping for weights/activations.
        q_params: Quantizer parameter dicts.
        lora_params: LoRA parameters for QuantLoRALinear.
        use_lora: Whether to use LoRA-aware quantized layers.
        wo_quant_adaln: Skip quantization for AdaLN-related layers.
        wo_quant_only_time_embedding: Skip all but time embedding layers.
        wo_quant_only_norm: Skip all but normalization layers.
        random_list: Indices of layers to skip.
        count: Running layer index for recursion.
        parent_name: Parent module name for logging.

    Returns:
        Tuple of (modified_model, updated_count).
    """
    for name, module in model.named_children():
        if wo_quant_adaln and (
            ("time_embedding" in name and not wo_quant_only_norm)
            or (
                ("norm1" in name or "norm2" in name)
                and not wo_quant_only_time_embedding
            )
        ):
            logger.info(f"Skip {parent_name}.{name}")
            continue
        if isinstance(module, nn.Linear):
            count += 1
            if count in random_list:
                logger.info(f"Skip {parent_name}.{name}")
                continue
            if use_lora:
                setattr(
                    model,
                    name,
                    QuantLoRALinear(module, quantizer_type, q_params, lora_params),
                )
            else:
                setattr(model, name, QuantLinear(module, quantizer_type, q_params))
        else:
            _, count = replace_linear(
                module,
                quantizer_type,
                q_params,
                lora_params,
                use_lora,
                wo_quant_adaln,
                wo_quant_only_time_embedding,
                wo_quant_only_norm,
                random_list,
                count,
                f"{parent_name}.{name}",
            )
    return model, count


def merge_lora(
    model: nn.Module,
) -> nn.Module:
    """Merge LoRA weights into base weights for all quantized layers.

    Args:
        model: Root module to update in-place.

    Returns:
        None. Modules are updated in-place.
    """
    for name, module in model.named_children():
        if isinstance(module, QuantLoRALinear):
            module.merge_lora()


if __name__ == "__main__":
    weights = torch.randn(1024, 1024, dtype=torch.float16, device="cuda")
    quantizer = ProgressiveLoRAQuantizer(
        bit=4,
        sym=False,
        granularity="per_tensor",
        cali="mse",
        goup_size=-1,
        round_zero=False,
        use_grad_scaling=True,
        round_func="STE",
        rank=32,
        iters=3,
        cur_alpha_strategy="cosine",
        alpha_T=3,
        warm_up=3,
    )
    quantizer(weights.detach())
    quantizer.build()
    lr = 1e3
    parameters = [
        quantizer.L,
        quantizer.R,
        quantizer.scale,
        quantizer.zero_point,
        weights,
    ]
    quantizer.scale.requires_grad = False
    quantizer.zero_point.requires_grad = False
    weights.requires_grad = False
    optimizer = torch.optim.SGD(parameters, lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    for _ in range(3):
        logger.info(f"Iter: {_}")
        quantizer.set_alpha_L_R(_)
        q_w = quantizer(weights)
        loss = torch.mean((q_w - weights.detach()) ** 2)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        logger.info(
            f"Loss: {loss.item()}, scale: {quantizer.scale.item()}, zero_point: {quantizer.zero_point.item()}, weights: {torch.norm(weights)}"
        )
        if hasattr(quantizer, "L") and hasattr(quantizer, "R"):
            logger.info(
                f"rank: {quantizer.L.shape[1]}, norm: {torch.norm(quantizer.L)}, norm: {torch.norm(quantizer.R)}"
            )
    quantizer.scale.requires_grad = True
    quantizer.zero_point.requires_grad = True
    weights.requires_grad = True
    x = weights.detach().clone()
    for _ in range(3, 30):
        logger.info(f"Iter: {_}")
        quantizer.set_alpha_L_R(_)
        q_w = quantizer(weights)
        loss = torch.mean((q_w - x) ** 2)
        loss.backward()
        # logger.info(f"{quantizer.scale.grad} {quantizer.zero_point.grad} {weights.grad}")
        optimizer.step()
        optimizer.zero_grad()
        logger.info(
            f"Loss: {loss.item()}, scale: {quantizer.scale.item()}, zero_point: {quantizer.zero_point.item()}, weights: {torch.norm(weights)}"
        )
        if hasattr(quantizer, "L") and hasattr(quantizer, "R"):
            logger.info(
                f"rank: {quantizer.L.shape[1]}, norm: {torch.norm(quantizer.L)}, norm: {torch.norm(quantizer.R)}"
            )
