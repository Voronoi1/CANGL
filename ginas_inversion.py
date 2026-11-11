import copy
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from torchvision import transforms

import global_var
from model_structure import (
    CalTech256_model_generator,
    Cifar100_model_generator,
    TinyImageNet_model_generator,
)
from utils import cal_LPIPS, cal_PSNR, cal_SSIM, get_logger


device = global_var.get_device()


def _make_norm(num_channels: int) -> nn.Module:
    groups = min(8, num_channels)
    while num_channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


def _activation(name: str) -> nn.Module:
    if name == "ReLU":
        return nn.ReLU(inplace=True)
    if name == "PReLU":
        return nn.PReLU()
    return nn.LeakyReLU(0.1, inplace=True)


def total_variation(x: torch.Tensor) -> torch.Tensor:
    diff1 = torch.norm(x[:, :, :, :-1] - x[:, :, :, 1:])
    diff2 = torch.norm(x[:, :, :-1, :] - x[:, :, 1:, :])
    return diff1 + diff2


def total_variation_mean(x: torch.Tensor) -> torch.Tensor:
    dx = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    dy = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return dx + dy


def l2_norm(x: torch.Tensor) -> torch.Tensor:
    return torch.norm(x, p=2)


def clip_loss(x: torch.Tensor) -> torch.Tensor:
    clipped = x.detach().clone()
    clipped.data[clipped.data < 0] = 0.0
    clipped.data[clipped.data > 1] = 1.0
    return torch.norm(x - clipped, p=2)


def cosine_lr(optimizer: torch.optim.Optimizer, base_lr: float, step: int, total_steps: int) -> None:
    if total_steps <= 1:
        lr = base_lr
    else:
        lr = 0.5 * (1.0 + math.cos(math.pi * step / float(total_steps - 1))) * base_lr
    for group in optimizer.param_groups:
        group["lr"] = lr


def ginas_prefix(save_name: str) -> str:
    if save_name.endswith("_GINAS"):
        return save_name
    return "{}_GINAS".format(save_name)


def gradient_cosine_loss(
    current_gradient: Sequence[torch.Tensor],
    target_gradient: Sequence[torch.Tensor],
    eps: float = 1e-12,
) -> torch.Tensor:
    cost = 0.0
    current_norm = 0.0
    target_norm = 0.0
    for current_grad, target_grad in zip(current_gradient, target_gradient):
        if current_grad is None or target_grad is None:
            continue
        target_grad = target_grad.to(current_grad.device, dtype=current_grad.dtype)
        cost = cost - (current_grad * target_grad).sum()
        current_norm = current_norm + current_grad.pow(2).sum()
        target_norm = target_norm + target_grad.pow(2).sum()
    return 1.0 + cost / (current_norm.sqrt() * target_norm.sqrt() + eps)


class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        activation: str = "LeakyReLU",
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=True),
            _make_norm(out_channels),
            _activation(activation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation: str) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct(in_channels, out_channels, kernel_size=3, stride=2, activation=activation),
            ConvNormAct(out_channels, out_channels, kernel_size=3, stride=1, activation=activation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, channels: int, activation: str) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct(channels, channels, kernel_size=3, stride=1, activation=activation),
            ConvNormAct(channels, channels, kernel_size=3, stride=1, activation=activation),
            ConvNormAct(channels, channels, kernel_size=1, stride=1, activation=activation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResizeProject(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation: str) -> None:
        super().__init__()
        self.project = ConvNormAct(in_channels, out_channels, kernel_size=1, stride=1, activation=activation)

    def forward(
        self,
        x: torch.Tensor,
        target_size: Tuple[int, int],
        mode: str,
    ) -> torch.Tensor:
        if x.shape[-2:] != target_size:
            if mode in ("bilinear", "bicubic"):
                x = F.interpolate(x, size=target_size, mode=mode, align_corners=False)
            else:
                x = F.interpolate(x, size=target_size, mode=mode)
        return self.project(x)


class GINASSkipGenerator(nn.Module):
    """
    Over-parameterized convolutional generator used by the new GI-NAS third stage.

    The architecture mirrors the GI-NAS idea: a five-level encoder-decoder network
    receives fixed image-shaped noise, and a 5x5 binary matrix decides which
    cross-scale skip projections are enabled for each decoder level.
    """

    def __init__(
        self,
        skip_matrix: np.ndarray,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 128,
        upsample_mode: str = "bilinear",
        activation: str = "LeakyReLU",
    ) -> None:
        super().__init__()
        skip_matrix = np.asarray(skip_matrix, dtype=np.int64)
        if skip_matrix.shape != (5, 5):
            raise ValueError("skip_matrix must have shape [5, 5].")

        self.register_buffer("skip_matrix", torch.as_tensor(skip_matrix, dtype=torch.long))
        self.upsample_mode = upsample_mode

        widths = [base_channels] * 5
        self.encoders = nn.ModuleList(
            [
                EncoderBlock(in_channels, widths[0], activation),
                EncoderBlock(widths[0], widths[1], activation),
                EncoderBlock(widths[1], widths[2], activation),
                EncoderBlock(widths[2], widths[3], activation),
                EncoderBlock(widths[3], widths[4], activation),
            ]
        )

        source_channels = [in_channels, widths[0], widths[1], widths[2], widths[3]]
        self.skip_projects = nn.ModuleList(
            [
                nn.ModuleList(
                    [ResizeProject(src_channels, widths[row], activation) for src_channels in source_channels]
                )
                for row in range(5)
            ]
        )

        self.main_projects = nn.ModuleList(
            [
                ResizeProject(widths[4], widths[4], activation),
                ResizeProject(widths[4], widths[3], activation),
                ResizeProject(widths[3], widths[2], activation),
                ResizeProject(widths[2], widths[1], activation),
                ResizeProject(widths[1], widths[0], activation),
            ]
        )
        self.decoders = nn.ModuleList([DecoderBlock(widths[4 - idx], activation) for idx in range(5)])
        self.output = nn.Sequential(nn.Conv2d(widths[0], out_channels, kernel_size=1, bias=True), nn.Sigmoid())

    def _resize_main(self, x: torch.Tensor, target_size: Tuple[int, int], project: ResizeProject) -> torch.Tensor:
        return project(x, target_size, self.upsample_mode)

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        sources = [noise]
        x = noise
        for encoder in self.encoders:
            x = encoder(x)
            sources.append(x)

        # sources: noise, enc1, enc2, enc3, enc4, enc5.
        decoder_targets = [
            noise.shape[-2:],
            sources[1].shape[-2:],
            sources[2].shape[-2:],
            sources[3].shape[-2:],
            sources[4].shape[-2:],
        ]

        x = sources[5]
        for decoder_idx, row in enumerate(range(4, -1, -1)):
            target_size = decoder_targets[row]
            x = self._resize_main(x, target_size, self.main_projects[decoder_idx])
            for source_idx in range(5):
                if bool(self.skip_matrix[row, source_idx].item()):
                    x = x + self.skip_projects[row][source_idx](
                        sources[source_idx],
                        target_size,
                        self.upsample_mode,
                    )
            x = self.decoders[decoder_idx](x)

        return self.output(x)


@dataclass
class GINASSearchResult:
    model_index: int
    upsample_mode: str
    activation: str
    skip_matrix: np.ndarray
    search_metric: float

    def as_dict(self) -> Dict:
        return {
            "model_index": self.model_index,
            "upsample_mode": self.upsample_mode,
            "activation": self.activation,
            "skip_matrix": self.skip_matrix.astype(int).tolist(),
            "search_metric": self.search_metric,
        }


def _load_skip_matrices_from_file(path: str) -> List[np.ndarray]:
    matrices = []
    if not path or not os.path.exists(path):
        return matrices
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip().replace(" ", "")
            if not stripped:
                continue
            values = [int(ch) for ch in stripped if ch in ("0", "1")]
            if len(values) != 25:
                continue
            matrices.append(np.asarray(values, dtype=np.int64).reshape(5, 5))
    return matrices


def _default_skip_matrices(seed: int, random_count: int) -> List[np.ndarray]:
    identity = np.eye(5, dtype=np.int64)
    full = np.ones((5, 5), dtype=np.int64)
    lower = np.tril(np.ones((5, 5), dtype=np.int64))
    upper = np.triu(np.ones((5, 5), dtype=np.int64))
    band = np.zeros((5, 5), dtype=np.int64)
    for row in range(5):
        for col in range(5):
            if abs(row - col) <= 1:
                band[row, col] = 1

    sparse_cross = np.asarray(
        [
            [1, 1, 0, 0, 1],
            [1, 1, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 1, 1],
            [1, 0, 0, 1, 1],
        ],
        dtype=np.int64,
    )

    rng = np.random.RandomState(seed)
    matrices = [identity, full, lower, upper, band, sparse_cross]
    for _ in range(random_count):
        matrix = rng.binomial(1, 0.45, size=(5, 5)).astype(np.int64)
        if matrix.sum() == 0:
            matrix[rng.randint(0, 5), rng.randint(0, 5)] = 1
        matrices.append(matrix)

    unique = []
    seen = set()
    for matrix in matrices:
        key = tuple(matrix.reshape(-1).tolist())
        if key not in seen:
            seen.add(key)
            unique.append(matrix)
    return unique


def _prepare_target_gradient(gradient: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    return [grad.detach().clone().to(device) for grad in gradient]


class GINASInverter:
    def __init__(
        self,
        model: nn.Module,
        image_shape: Tuple[int, int, int],
        config: Optional[Dict] = None,
    ) -> None:
        self.model = copy.deepcopy(model).to(device)
        self.model.eval()
        self.image_shape = tuple(image_shape)
        self.config = {
            "lr": 1e-3,
            "num_iterations": 20000,
            "search_base_channels": 64,
            "train_base_channels": 128,
            "search_random_candidates": 10,
            "max_search_candidates": 18,
            "search_space_path": None,
            "upsample_modes": ["bilinear", "nearest"],
            "activations": ["LeakyReLU"],
            "regularization_mode": "ginas",
            "tv_l2_tv_reduction": "mean",
            "grad_coeff": 1.0,
            "tv_coeff": 1e-4,
            "l2_coeff": 0.0,
            "clip_coeff": 0.0,
            "scale_coeff": 0.0,
            "save_every": 100,
            "lr_decay": True,
            "signed_grad": False,
            "single_sample_update": False,
            "seed": 0,
        }
        if config is not None:
            self.config.update(config)

    def attack(
        self,
        target_gradient: Sequence[torch.Tensor],
        target_class: int,
        original_image: Optional[torch.Tensor] = None,
        save_name: str = "ginas",
        output_dir: str = "./output",
        result_dir: str = "./result",
        logger=None,
        show_progress: bool = True,
        output_size: int = 1,
    ) -> Tuple[List[torch.Tensor], torch.Tensor, Dict]:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(result_dir, exist_ok=True)

        self._seed_everything(int(self.config["seed"]))
        target_gradient = _prepare_target_gradient(target_gradient)
        labels = torch.LongTensor([target_class] * output_size).to(device)
        noise = torch.randn((output_size, *self.image_shape), device=device)

        search_result, net_g = self._search_architecture(target_gradient, labels, noise, logger, show_progress)
        message = "[GI-NAS] selected model_index={} metric={:.6f} upsample={} activation={}".format(
            search_result.model_index,
            search_result.search_metric,
            search_result.upsample_mode,
            search_result.activation,
        )
        self._log(message, logger, show_progress)
        self._log(
            "[GI-NAS] regularization_mode={}".format(self.config["regularization_mode"]),
            logger,
            show_progress,
        )

        history, final_image, losses = self._optimize_generator(
            net_g=net_g,
            noise=noise,
            labels=labels,
            target_gradient=target_gradient,
            save_name=save_name,
            output_dir=output_dir,
            logger=logger,
            show_progress=show_progress,
        )

        final_prefix = ginas_prefix(save_name)
        vutils.save_image(
            final_image.detach().cpu(),
            os.path.join(result_dir, "{}.png".format(final_prefix)),
            normalize=False,
            scale_each=True,
            nrow=max(1, min(4, output_size)),
        )

        info = {
            "search": search_result.as_dict(),
            "losses": losses,
        }
        if original_image is not None:
            original_image = original_image.detach().clone().to(final_image.device)
            info["metrics"] = self._safe_metrics(final_image, original_image, logger, show_progress)
        return history, final_image, info

    def _search_architecture(
        self,
        target_gradient: Sequence[torch.Tensor],
        labels: torch.Tensor,
        noise: torch.Tensor,
        logger,
        show_progress: bool,
    ) -> Tuple[GINASSearchResult, nn.Module]:
        matrices = _load_skip_matrices_from_file(self.config.get("search_space_path"))
        if not matrices:
            matrices = _default_skip_matrices(
                seed=int(self.config["seed"]),
                random_count=int(self.config["search_random_candidates"]),
            )
        max_candidates = self.config.get("max_search_candidates")
        if max_candidates is not None:
            matrices = matrices[: int(max_candidates)]

        best_metric = float("inf")
        best_model = None
        best_result = None
        model_index = 0

        for matrix in matrices:
            for upsample_mode in self.config["upsample_modes"]:
                for activation in self.config["activations"]:
                    self._seed_everything(int(self.config["seed"]) + model_index)
                    try:
                        trial_net = GINASSkipGenerator(
                            skip_matrix=matrix,
                            base_channels=int(self.config["search_base_channels"]),
                            upsample_mode=upsample_mode,
                            activation=activation,
                        ).to(device)
                        metric = self._search_metric(trial_net, noise, labels, target_gradient)
                    except RuntimeError as exc:
                        self._log(
                            "[GI-NAS] skip candidate {} failed: {}".format(model_index, str(exc)),
                            logger,
                            show_progress,
                        )
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        model_index += 1
                        continue

                    if metric < best_metric:
                        best_metric = metric
                        best_result = GINASSearchResult(
                            model_index=model_index,
                            upsample_mode=upsample_mode,
                            activation=activation,
                            skip_matrix=matrix.copy(),
                            search_metric=metric,
                        )
                        best_model = copy.deepcopy(trial_net).to(device)
                        self._log(
                            "[GI-NAS] new best model_index={} metric={:.6f} skip={}".format(
                                model_index,
                                metric,
                                matrix.reshape(-1).astype(int).tolist(),
                            ),
                            logger,
                            show_progress,
                        )

                    del trial_net
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    model_index += 1

        if best_model is None or best_result is None:
            raise RuntimeError("GI-NAS architecture search did not produce a valid generator.")

        if int(self.config["train_base_channels"]) != int(self.config["search_base_channels"]):
            self._seed_everything(int(self.config["seed"]) + best_result.model_index)
            best_model = GINASSkipGenerator(
                skip_matrix=best_result.skip_matrix,
                base_channels=int(self.config["train_base_channels"]),
                upsample_mode=best_result.upsample_mode,
                activation=best_result.activation,
            ).to(device)

        return best_result, best_model

    def _search_metric(
        self,
        net_g: nn.Module,
        noise: torch.Tensor,
        labels: torch.Tensor,
        target_gradient: Sequence[torch.Tensor],
    ) -> float:
        net_g.eval()
        self.model.zero_grad()
        fake = net_g(noise)
        logits = self.model(fake)
        loss = nn.CrossEntropyLoss()(logits, labels)
        fake_gradient = torch.autograd.grad(loss, self.model.parameters(), create_graph=False)
        metric = gradient_cosine_loss(fake_gradient, target_gradient)
        return float(metric.detach().cpu().item())

    def _optimize_generator(
        self,
        net_g: nn.Module,
        noise: torch.Tensor,
        labels: torch.Tensor,
        target_gradient: Sequence[torch.Tensor],
        save_name: str,
        output_dir: str,
        logger,
        show_progress: bool,
    ) -> Tuple[List[torch.Tensor], torch.Tensor, Dict[str, List[float]]]:
        net_g.train()
        optimizer = torch.optim.Adam(net_g.parameters(), lr=float(self.config["lr"]))
        history = []
        losses = {"total": [], "grad": [], "tv": [], "l2": [], "clip": [], "scale": [], "aux": []}
        best_loss = float("inf")
        best_image = None
        output_prefix = ginas_prefix(save_name)
        regularization_mode = str(self.config.get("regularization_mode", "ginas"))
        if regularization_mode not in ("ginas", "tv_l2"):
            raise ValueError("Unsupported GI-NAS regularization_mode: {}".format(regularization_mode))
        tv_l2_tv_reduction = str(self.config.get("tv_l2_tv_reduction", "mean")).lower()
        if tv_l2_tv_reduction not in ("mean", "variation"):
            raise ValueError("Unsupported GI-NAS tv_l2_tv_reduction: {}".format(tv_l2_tv_reduction))

        total_steps = int(self.config["num_iterations"])
        for step in range(total_steps):
            if self.config["lr_decay"]:
                cosine_lr(optimizer, float(self.config["lr"]), step, total_steps)

            optimizer.zero_grad()
            self.model.zero_grad()
            fake = net_g(noise)
            selected_index = None
            if bool(self.config.get("single_sample_update", False)) and fake.shape[0] > 1:
                selected_index = step % fake.shape[0]
                mask = torch.zeros_like(fake)
                mask[selected_index : selected_index + 1] = 1.0
                fake.register_hook(lambda grad, mask=mask: grad * mask)
            logits = self.model(fake)
            loss = nn.CrossEntropyLoss()(logits, labels)
            fake_gradient = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
            grad_loss = gradient_cosine_loss(fake_gradient, target_gradient)
            if regularization_mode == "tv_l2":
                if tv_l2_tv_reduction == "variation":
                    tv_reg = total_variation(fake)
                else:
                    tv_reg = total_variation_mean(fake)
                l2_reg = l2_norm(fake)
                clip_reg = fake.new_tensor(0.0)
                scale_reg = fake.new_tensor(0.0)
                aux_loss = (
                    float(self.config["l2_coeff"]) * l2_reg
                    + float(self.config["tv_coeff"]) * tv_reg
                )
                total_loss = float(self.config["grad_coeff"]) * grad_loss + aux_loss
            else:
                tv_reg = total_variation_mean(fake)
                l2_reg = torch.norm(fake, p=2)
                clip_reg = clip_loss(fake)
                scale_reg = fake.new_tensor(0.0)
                aux_loss = (
                    float(self.config["tv_coeff"]) * tv_reg
                    + float(self.config["l2_coeff"]) * l2_reg
                    + float(self.config["clip_coeff"]) * clip_reg
                )
                total_loss = grad_loss + aux_loss
            total_loss.backward()

            if self.config["signed_grad"]:
                for parameter in net_g.parameters():
                    if parameter.grad is not None:
                        parameter.grad.sign_()

            optimizer.step()
            for parameter in self.model.parameters():
                parameter.grad = None

            losses["total"].append(float(total_loss.detach().cpu().item()))
            losses["grad"].append(float(grad_loss.detach().cpu().item()))
            losses["tv"].append(float(tv_reg.detach().cpu().item()))
            losses["l2"].append(float(l2_reg.detach().cpu().item()))
            losses["clip"].append(float(clip_reg.detach().cpu().item()))
            losses["scale"].append(float(scale_reg.detach().cpu().item()))
            losses["aux"].append(float(aux_loss.detach().cpu().item()))

            if total_loss.item() < best_loss:
                best_loss = float(total_loss.item())
                best_image = fake.detach().clone()

            if (step + 1) % int(self.config["save_every"]) == 0 or step == total_steps - 1:
                snapshot = fake.detach().cpu()
                history.append(snapshot)
                vutils.save_image(
                    snapshot,
                    os.path.join(output_dir, "{}_{:05d}.png".format(output_prefix, step + 1)),
                    normalize=False,
                    scale_each=True,
                    nrow=max(1, min(4, snapshot.shape[0])),
                )

            if step % 100 == 0 or step == total_steps - 1:
                if selected_index is None:
                    message = "[GI-NAS] iter={} total={:.6f} grad={:.6f} tv={:.6f} l2={:.6f} clip={:.6f} scale={:.6f} aux={:.6f}".format(
                        step,
                        total_loss.item(),
                        grad_loss.item(),
                        tv_reg.item(),
                        l2_reg.item(),
                        clip_reg.item(),
                        scale_reg.item(),
                        aux_loss.item(),
                    )
                else:
                    message = "[GI-NAS] iter={} sample={} total={:.6f} grad={:.6f} tv={:.6f} l2={:.6f} clip={:.6f} scale={:.6f} aux={:.6f}".format(
                        step,
                        selected_index,
                        total_loss.item(),
                        grad_loss.item(),
                        tv_reg.item(),
                        l2_reg.item(),
                        clip_reg.item(),
                        scale_reg.item(),
                        aux_loss.item(),
                    )
                self._log(message, logger, show_progress)

        if best_image is None:
            best_image = net_g(noise).detach()
        return history, best_image.detach(), losses

    def _safe_metrics(
        self,
        final_image: torch.Tensor,
        original_image: torch.Tensor,
        logger,
        show_progress: bool,
    ) -> Dict[str, float]:
        metrics = {}
        try:
            metrics["psnr"] = float(cal_PSNR(final_image, original_image))
            metrics["ssim"] = float(cal_SSIM(final_image, original_image))
            metrics["lpips"] = float(cal_LPIPS(final_image, original_image))
            self._log(
                "[GI-NAS] PSNR={:.6f} SSIM={:.6f} LPIPS={:.6f}".format(
                    metrics["psnr"],
                    metrics["ssim"],
                    metrics["lpips"],
                ),
                logger,
                show_progress,
            )
        except Exception as exc:
            self._log("[GI-NAS] metric evaluation skipped: {}".format(str(exc)), logger, show_progress)
        return metrics

    @staticmethod
    def _seed_everything(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _log(message: str, logger, show_progress: bool) -> None:
        if show_progress:
            print(message)
        if logger is not None:
            logger.info(message)


def ginas_gradient_inversion_attack(
    model: nn.Module,
    gradient: Sequence[torch.Tensor],
    target_class: int,
    original_input: torch.Tensor,
    save_name: str,
    logger,
    image_shape: Sequence[int],
    output_size: int = 1,
    config: Optional[Dict] = None,
    save_path: str = "./result/",
    output_path: str = "./output/",
    ifprint: bool = False,
) -> Tuple[List[torch.Tensor], torch.Tensor, Dict]:
    inverter = GINASInverter(model=model, image_shape=tuple(image_shape), config=config)
    return inverter.attack(
        target_gradient=gradient,
        target_class=target_class,
        original_image=original_input,
        save_name=save_name,
        output_dir=output_path,
        result_dir=save_path,
        logger=logger,
        show_progress=ifprint,
        output_size=output_size,
    )


def inversion_ginas(
    dataset_name: str,
    target_class: int,
    knowledge_level: str,
    agr_name: str,
    gradient_tuple,
    ginas_config: Optional[Dict] = None,
    output_size: int = 1,
    ifprint: bool = True,
):
    model_generator_dict = {
        "TinyImageNet": TinyImageNet_model_generator,
        "Cifar100": Cifar100_model_generator,
        "CalTech256": CalTech256_model_generator,
    }
    image_shape_dict = {
        "TinyImageNet": [1, 3, 64, 64],
        "Cifar100": [1, 3, 32, 32],
        "CalTech256": [1, 3, 112, 112],
    }

    target_img_batch, gradient_list, inference_gradient, previous_global_model_state_dict = gradient_tuple
    del gradient_list

    model = model_generator_dict[dataset_name]()
    model.load_state_dict(previous_global_model_state_dict)

    x, _ = target_img_batch
    original_image = x[0:output_size].detach().clone()
    save_name = "{}_{}_{}_{}".format(dataset_name, knowledge_level, target_class, agr_name)
    final_prefix = ginas_prefix(save_name)
    os.makedirs("./result", exist_ok=True)
    os.makedirs("./output", exist_ok=True)
    os.makedirs("./log/S3", exist_ok=True)
    vutils.save_image(
        original_image.reshape(output_size, *image_shape_dict[dataset_name][1:]),
        "./result/{}_Original.png".format(final_prefix),
        normalize=False,
        scale_each=True,
        nrow=max(1, min(4, output_size)),
    )

    logger = get_logger(
        "./log/S3/{}_Inversion.log".format(final_prefix),
        verbosity=1,
        name="{}_Inversion".format(final_prefix),
    )

    history, final_image, info = ginas_gradient_inversion_attack(
        model=model,
        gradient=inference_gradient,
        target_class=target_class,
        original_input=original_image,
        save_name=save_name,
        logger=logger,
        image_shape=image_shape_dict[dataset_name][1:],
        output_size=output_size,
        config=ginas_config,
        save_path="./result/",
        output_path="./output/",
        ifprint=ifprint,
    )

    final_image = final_image.to(device)
    vutils.save_image(
        final_image.reshape(output_size, *image_shape_dict[dataset_name][1:]),
        "./result/{}_Inversion.png".format(final_prefix),
        normalize=False,
        scale_each=True,
        nrow=max(1, min(4, output_size)),
    )

    tp = transforms.ToPILImage()
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 5))
        plt.suptitle("{}_{}_{}_GINAS".format(dataset_name, agr_name, knowledge_level))
        plt.subplot(1, 2, 1)
        plt.imshow(tp(original_image[0].cpu()))
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(tp(final_image[0].detach().cpu()))
        plt.axis("off")
        plt.show()
    except Exception:
        pass

    return history, final_image, info
