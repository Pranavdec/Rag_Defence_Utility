"""
Privacy-Aware Decoding (PAD) defense.

Adapted from the reference implementation (Wang et al., arXiv:2508.03098), PAD/
llm.py — CC BY-NC 4.0. Adds adaptive or static Gaussian noise at decode time via
Hugging Face LogitsProcessor hooks.

Paper: https://arxiv.org/abs/2508.03098
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from transformers import LogitsProcessor

from .base import BaseDefense


class RDPAccountant:
    """
    Tracks cumulative privacy loss using Rényi Differential Privacy (RDP) composition for Gaussian mechanisms.
    Converts to (epsilon, delta)-DP guarantee.
    """

    def __init__(self, alpha=10.0, delta=1e-5):
        self.alpha = alpha
        self.delta = delta
        self.rdp = 0.0
        self.steps_with_noise = 0
        self.total_steps = 0

    def add_gaussian_step(self, sensitivity, sigma, noise_injected=True):
        self.total_steps += 1
        if noise_injected:
            self.steps_with_noise += 1
            self.rdp += (self.alpha * (sensitivity ** 2)) / (2 * (sigma ** 2))

    def get_epsilon(self):
        return self.rdp + np.log(1 / self.delta) / (self.alpha - 1)

    def get_gamma(self):
        if self.total_steps == 0:
            return 1.0
        return self.steps_with_noise / self.total_steps


class StaticNoiseProcessor(LogitsProcessor):
    """Static noise baseline: uniform Gaussian noise on logits each step."""

    def __init__(self, epsilon_base=1.0, alpha=10.0, delta=1e-5, noise_scale=0.1):
        self.noise_scale = noise_scale
        self.epsilon_base = epsilon_base
        self.accountant = RDPAccountant(alpha=alpha, delta=delta)
        self.step_count = 0
        self.sensitivity = 1.0

    def __call__(self, input_ids, scores):
        self.step_count += 1
        noise = torch.randn_like(scores) * self.noise_scale
        self.accountant.add_gaussian_step(sensitivity=self.sensitivity, sigma=self.noise_scale, noise_injected=True)
        return scores + noise

    def get_total_privacy_loss(self):
        return self.accountant.get_epsilon()

    def get_gamma(self):
        return self.accountant.get_gamma()


class DataDependentCalibrator:
    """Calibrates noise scale based on token entropy, position, and confidence."""

    def __init__(self, entropy_weight=0.3, position_weight=0.2):
        self.entropy_weight = entropy_weight
        self.position_weight = position_weight

    def calibrate_noise_scale(self, scores, position, base_scale):
        with torch.no_grad():
            probs = F.softmax(scores, dim=-1)
            log_probs = F.log_softmax(scores, dim=-1)
            token_entropy = -(probs * log_probs).sum().item()
            max_entropy = np.log(probs.numel())
            normalized_entropy = token_entropy / max_entropy
            position_factor = 1.0 / (1.0 + position * 0.1)
            top1_prob = probs.max().item()
            confidence_factor = 1.0 - top1_prob
            calibration_factor = (
                (1 - self.entropy_weight) * 1.0
                + self.entropy_weight * normalized_entropy
                + self.position_weight * position_factor
                + confidence_factor * 0.3
            )
            calibration_factor = max(0.1, min(2.0, calibration_factor))
            return base_scale * calibration_factor


class ScreeningMechanism:
    """Skips heavy noise for high-confidence predictions."""

    def __init__(self, confidence_threshold=0.9, margin_threshold=2.0):
        self.confidence_threshold = confidence_threshold
        self.margin_threshold = margin_threshold

    def should_skip_noise(self, scores):
        probs = F.softmax(scores, dim=-1)
        top1_prob = probs.max().item()
        topk = torch.topk(scores, 2, dim=-1).values
        logit_margin = (topk[..., 0] - topk[..., 1]).mean().item()
        return top1_prob > self.confidence_threshold and logit_margin > self.margin_threshold


class AdaptiveNoiseProcessor(LogitsProcessor):
    """Adaptive Gaussian noise on logits with screening and calibration."""

    def __init__(
        self,
        epsilon_base=1.0,
        alpha=10.0,
        delta=1e-5,
        enable_screening=True,
        enable_calibration=True,
        noise_amplification=2.0,
        min_sensitivity=0.5,
    ):
        self.base_scale = 0.01 / max(epsilon_base, 0.01)
        self.epsilon_base = epsilon_base
        self.accountant = RDPAccountant(alpha=alpha, delta=delta)
        self.step_count = 0
        self.noise_amplification = noise_amplification
        self.min_sensitivity = min_sensitivity
        self.min_sigma = 0.01
        self.max_sigma = 10.0
        self.calibrator = DataDependentCalibrator() if enable_calibration else None
        self.screener = ScreeningMechanism() if enable_screening else None

    def __call__(self, input_ids, scores):
        self.step_count += 1
        if self.screener and self.screener.should_skip_noise(scores):
            minimal_noise = torch.randn_like(scores) * self.min_sigma
            self.accountant.add_gaussian_step(sensitivity=0.0, sigma=self.min_sigma, noise_injected=True)
            return scores + minimal_noise

        with torch.no_grad():
            topk = torch.topk(scores, 2, dim=-1).values
            logit_margin = topk[..., 0] - topk[..., 1]
            margin = logit_margin.mean().item()
            sensitivity = max(
                self.min_sensitivity,
                min(1.0 / (1 + np.log(1 + max(margin, 1e-6))), 1.0),
            )

        if self.calibrator:
            sigma = self.calibrator.calibrate_noise_scale(scores, self.step_count, self.base_scale)
        else:
            sigma = self.base_scale

        sigma = sigma * (sensitivity / self.epsilon_base) * self.noise_amplification
        sigma = min(self.max_sigma, max(self.min_sigma, sigma))

        noise = torch.randn_like(scores) * sigma
        self.accountant.add_gaussian_step(sensitivity=sensitivity, sigma=sigma, noise_injected=True)
        return scores + noise

    def get_total_privacy_loss(self):
        return self.accountant.get_epsilon()

    def get_gamma(self):
        return self.accountant.get_gamma()


class PADDefense(BaseDefense):
    """
    Privacy-Aware Decoding: inject noise into logits during HF generation.

    Requires ``system.llm.provider: huggingface`` (or ``hf``); incompatible with vLLM/Ollama
    for logits-level hooks.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name = config.get("name", self.CANONICAL_NAME)

        self.epsilon = float(config.get("epsilon", 0.2))
        self.alpha = float(config.get("alpha", 10.0))
        self.delta = float(config.get("delta", 1e-5))
        self.enable_screening = bool(config.get("enable_screening", True))
        self.enable_calibration = bool(config.get("enable_calibration", True))
        self.noise_amplification = float(config.get("noise_amplification", 3.0))
        self.min_sensitivity = float(config.get("min_sensitivity", 0.4))
        self.noise_type = str(config.get("noise_type", "adaptive")).lower()
        self.static_noise_scale = float(config.get("static_noise_scale", 0.1))
        self.candidate_multiplier = int(config.get("candidate_multiplier", 1))

    def build_logits_processor(self):
        """Fresh processor per generation call (per-completion accounting)."""
        if self.noise_type == "static":
            return StaticNoiseProcessor(
                epsilon_base=self.epsilon,
                alpha=self.alpha,
                delta=self.delta,
                noise_scale=self.static_noise_scale,
            )
        return AdaptiveNoiseProcessor(
            epsilon_base=self.epsilon,
            alpha=self.alpha,
            delta=self.delta,
            enable_screening=self.enable_screening,
            enable_calibration=self.enable_calibration,
            noise_amplification=self.noise_amplification,
            min_sensitivity=self.min_sensitivity,
        )
