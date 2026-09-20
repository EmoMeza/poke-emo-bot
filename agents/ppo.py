"""
PPO (Proximal Policy Optimization) para el ActorCritic de agents/actor_critic.py.

Tres piezas:
- Transition / RolloutBuffer: guardan lo que paso en cada decision y calculan
  las ventajas de cada trayectoria terminada.
- compute_gae: ventajas con GAE (que tan mejor salio una jugada de lo esperado).
- ppo_update: mejora la red con esos datos, sin alejar la politica de la que
  los genero (recorte del ratio).

Ver docs/como_funciona.md para la explicacion completa con ejemplos.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from agents.actor_critic import ActorCritic


@dataclass
class Transition:
    """Una decision de un agente (ambos slots juntos)."""

    obs: np.ndarray  # observacion antes de actuar, (obs_size,)
    action: np.ndarray  # [a1, a2]
    mask1: np.ndarray  # acciones legales del slot 1, (n_actions,)
    mask2: np.ndarray  # acciones legales del slot 2 dado a1, (n_actions,)
    log_prob: float  # log-prob conjunto de la accion, con la red de ese momento
    value: float  # V(s) estimado por la red en ese momento
    reward: float  # reward acumulado hasta la siguiente decision del agente


def compute_gae(
    rewards: np.ndarray, values: np.ndarray, gamma: float, lam: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ventajas GAE y retornos de UNA trayectoria terminada (la ultima decision
    es el final de la partida, asi que su siguiente valor es 0).

    delta_t = r_t + gamma * V(t+1) - V(t)          (la "sorpresa" de un paso)
    A_t     = delta_t + gamma * lam * A_{t+1}       (mezcla las sorpresas futuras)
    retorno = A_t + V(t)                            (objetivo para el critico)
    """
    advantages = np.zeros(len(rewards), dtype=np.float32)
    next_advantage = 0.0
    for t in reversed(range(len(rewards))):
        next_value = values[t + 1] if t + 1 < len(rewards) else 0.0
        delta = rewards[t] + gamma * next_value - values[t]
        next_advantage = delta + gamma * lam * next_advantage
        advantages[t] = next_advantage
    return advantages, advantages + values


@dataclass
class Batch:
    obs: torch.Tensor
    actions: torch.Tensor
    mask1: torch.Tensor
    mask2: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor

    def __len__(self) -> int:
        return len(self.obs)


class RolloutBuffer:
    def __init__(self, gamma: float = 0.99, lam: float = 0.95):
        self.gamma = gamma
        self.lam = lam
        self.clear()

    def clear(self):
        self._transitions: List[Transition] = []
        self._advantages: List[np.ndarray] = []
        self._returns: List[np.ndarray] = []

    def __len__(self) -> int:
        return len(self._transitions)

    def add_trajectory(self, trajectory: List[Transition]):
        """Agrega las decisiones de un agente en una partida terminada."""
        rewards = np.array([t.reward for t in trajectory], dtype=np.float32)
        values = np.array([t.value for t in trajectory], dtype=np.float32)
        advantages, returns = compute_gae(rewards, values, self.gamma, self.lam)
        self._transitions += trajectory
        self._advantages.append(advantages)
        self._returns.append(returns)

    def to_batch(self) -> Batch:
        def stack(field: str, dtype) -> torch.Tensor:
            return torch.as_tensor(np.array([getattr(t, field) for t in self._transitions]), dtype=dtype)

        return Batch(
            obs=stack("obs", torch.float32),
            actions=stack("action", torch.long),
            mask1=stack("mask1", torch.bool),
            mask2=stack("mask2", torch.bool),
            log_probs=stack("log_prob", torch.float32),
            values=stack("value", torch.float32),
            advantages=torch.as_tensor(np.concatenate(self._advantages)),
            returns=torch.as_tensor(np.concatenate(self._returns)),
        )


def ppo_update(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: Batch,
    epochs: int = 4,
    minibatch_size: int = 256,
    clip: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    max_grad_norm: float = 0.5,
) -> Dict[str, float]:
    """
    Mejora la red con los datos de un rollout (varias pasadas sobre los mismos datos).

    :param clip: el ratio prob_nueva/prob_vieja se recorta a [1-clip, 1+clip]: una
        accion no puede volverse mucho mas (o menos) probable en un solo update.
    :param vf_coef: peso de la perdida del critico.
    :param ent_coef: peso del bono de entropia (premia mantener la exploracion).
    :return: metricas promedio del update, para registrar.
    """
    advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)
    metrics: Dict[str, List[float]] = {
        "policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": [], "clip_fraction": []
    }

    for _ in range(epochs):
        order = torch.randperm(len(batch))
        for start in range(0, len(batch), minibatch_size):
            idx = order[start : start + minibatch_size]
            log_prob, entropy, value = model.evaluate(
                batch.obs[idx], batch.mask1[idx], batch.mask2[idx], batch.actions[idx]
            )

            log_ratio = log_prob - batch.log_probs[idx]
            ratio = log_ratio.exp()
            adv = advantages[idx]
            policy_loss = -torch.min(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv).mean()
            value_loss = ((value - batch.returns[idx]) ** 2).mean()
            entropy_mean = entropy.mean()

            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy_mean
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                metrics["policy_loss"].append(policy_loss.item())
                metrics["value_loss"].append(value_loss.item())
                metrics["entropy"].append(entropy_mean.item())
                metrics["approx_kl"].append(((ratio - 1) - log_ratio).mean().item())
                metrics["clip_fraction"].append(((ratio - 1).abs() > clip).float().mean().item())

    result = {name: float(np.mean(values)) for name, values in metrics.items()}
    variance = batch.returns.var()
    result["explained_variance"] = float(1 - (batch.returns - batch.values).var() / (variance + 1e-8))
    return result
