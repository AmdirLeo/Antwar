from __future__ import annotations

from dataclasses import dataclass
import math
import random

import numpy as np


@dataclass(slots=True)
class PolicyStep:
    action: int
    probability: float
    value: float


class MaskedLinearPolicy:
    def __init__(self, obs_dim: int, action_dim: int, seed: int | None = None) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        rng = np.random.default_rng(seed)
        scale = 1.0 / math.sqrt(max(obs_dim, 1))
        self.policy_weights = rng.normal(0.0, scale, size=(obs_dim, action_dim)).astype(np.float32)
        self.policy_bias = np.zeros(action_dim, dtype=np.float32)
        self.value_weights = rng.normal(0.0, scale, size=(obs_dim,)).astype(np.float32)
        self.value_bias = np.float32(0.0)
        self.rng = random.Random(seed)

    def _masked_logits(self, observation: np.ndarray, mask: np.ndarray) -> np.ndarray:
        logits = observation @ self.policy_weights + self.policy_bias
        masked = logits.astype(np.float32, copy=True)
        masked[mask <= 0] = -1e9
        return masked

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits)
        exp = np.exp(shifted)
        total = np.sum(exp)
        if total <= 0:
            probs = np.zeros_like(logits, dtype=np.float32)
            probs[0] = 1.0
            return probs
        return (exp / total).astype(np.float32)

    def step(self, observation: np.ndarray, mask: np.ndarray, explore: bool = True) -> PolicyStep:
        logits = self._masked_logits(observation, mask)
        probs = self._softmax(logits)
        value = float(observation @ self.value_weights + self.value_bias)
        valid_indices = [index for index, flag in enumerate(mask.tolist()) if flag]
        if not valid_indices:
            return PolicyStep(action=0, probability=1.0, value=value)
        if explore:
            action = self.rng.choices(range(len(probs)), weights=probs.tolist(), k=1)[0]
        else:
            action = int(np.argmax(probs))
        return PolicyStep(action=action, probability=float(probs[action]), value=value)

    def update(
        self,
        observations: np.ndarray,
        masks: np.ndarray,
        actions: np.ndarray,
        returns: np.ndarray,
        learning_rate: float = 1e-2,
        value_learning_rate: float = 5e-3,
    ) -> dict[str, float]:
        logits = observations @ self.policy_weights + self.policy_bias
        logits = logits.astype(np.float32, copy=False)
        logits[masks <= 0] = -1e9
        logits -= np.max(logits, axis=1, keepdims=True)
        exp = np.exp(logits)
        exp *= masks
        denom = np.sum(exp, axis=1, keepdims=True)
        denom[denom <= 0] = 1.0
        probs = exp / denom

        values = observations @ self.value_weights + self.value_bias
        advantages = returns - values
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(len(actions)), actions] = 1.0
        grad_logits = (one_hot - probs) * advantages[:, None]
        grad_policy_weights = observations.T @ grad_logits / max(len(actions), 1)
        grad_policy_bias = grad_logits.mean(axis=0)

        value_error = advantages
        grad_value_weights = observations.T @ value_error / max(len(actions), 1)
        grad_value_bias = float(np.mean(value_error))

        self.policy_weights += learning_rate * grad_policy_weights.astype(np.float32)
        self.policy_bias += learning_rate * grad_policy_bias.astype(np.float32)
        self.value_weights += value_learning_rate * grad_value_weights.astype(np.float32)
        self.value_bias += np.float32(value_learning_rate * grad_value_bias)

        entropy = -np.mean(np.sum(np.where(probs > 0, probs * np.log(probs + 1e-8), 0.0), axis=1))
        return {
            "policy_loss_proxy": float(-np.mean(advantages)),
            "value_loss": float(np.mean(value_error ** 2)),
            "entropy": float(entropy),
            "mean_return": float(np.mean(returns)),
        }
