"""
Red actor-critic (MLP) para batallas dobles.

Recibe la observacion (observations/embedding.py) y produce:
- logits del slot 1 y del slot 2: un puntaje por cada una de las 107 acciones
  posibles de cada pokemon activo (la "politica", el actor);
- un valor V(s): cuanto reward total espera el agente desde ese estado (el
  "critico", que sirve de referencia para medir si una jugada salio mejor o
  peor de lo esperado).

Las acciones ilegales se enmascaran antes de muestrear (ver envs/vgc_env.py) y
el slot 2 se muestrea despues del slot 1, con una mascara que depende de lo que
eligio el primero.
"""

from typing import Callable, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

# Finito (no -inf): con -inf la entropia calcula 0 * -inf = NaN.
MASK_VALUE = -1e9


def masked_distribution(logits: torch.Tensor, mask: torch.Tensor) -> Categorical:
    """Distribucion sobre acciones donde las ilegales (mask=False) tienen probabilidad 0."""
    return Categorical(logits=logits.masked_fill(~mask, MASK_VALUE))


class ActorCritic(nn.Module):
    def __init__(self, obs_size: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_size, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.slot1_head = nn.Linear(hidden, n_actions)
        self.slot2_head = nn.Linear(hidden, n_actions)
        self.value_head = nn.Linear(hidden, 1)

        # Init ortogonal, estandar en PPO: la politica arranca casi uniforme
        # (ganancia chica) y el critico con escala normal.
        for module in self.trunk:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)
        for head, gain in [(self.slot1_head, 0.01), (self.slot2_head, 0.01), (self.value_head, 1.0)]:
            nn.init.orthogonal_(head.weight, gain=gain)
            nn.init.zeros_(head.bias)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.trunk(obs)
        return self.slot1_head(features), self.slot2_head(features), self.value_head(features).squeeze(-1)

    @torch.no_grad()
    def act(
        self,
        obs: np.ndarray,
        mask: np.ndarray,
        second_slot_mask_fn: Callable[[np.ndarray, int], np.ndarray],
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """
        Elige la accion de los dos slots para una observacion.

        :param obs: observacion del agente, forma (obs_size,).
        :param mask: acciones legales de los dos slots, forma (2, n_actions).
        :param second_slot_mask_fn: mascara del slot 2 dada la eleccion del slot 1
            (VGCDoublesEnv.second_slot_mask).
        :param deterministic: True elige la accion mas probable en vez de muestrear
            (para evaluar; en entrenamiento hay que muestrear para explorar).
        :return: (accion [a1, a2], mascara usada en el slot 2, log-prob conjunto, V(s)).
        """
        logits1, logits2, value = self(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
        dist1 = masked_distribution(logits1[0], torch.as_tensor(mask[0]))
        first = dist1.logits.argmax() if deterministic else dist1.sample()

        mask2 = second_slot_mask_fn(mask[1], int(first))
        dist2 = masked_distribution(logits2[0], torch.as_tensor(mask2))
        second = dist2.logits.argmax() if deterministic else dist2.sample()

        # p(a1, a2) = p(a1) * p(a2 | a1), asi que los log-prob se suman.
        log_prob = dist1.log_prob(first) + dist2.log_prob(second)
        return np.array([int(first), int(second)]), mask2, float(log_prob), float(value[0])

    def evaluate(
        self,
        obs: torch.Tensor,
        mask1: torch.Tensor,
        mask2: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Recalcula, con la red actual, log-prob, entropia y valor de jugadas ya
        hechas (lote). Usa las mascaras guardadas al jugarlas para que las
        probabilidades sean comparables con las de ese momento.

        :return: (log-prob conjunto, entropia conjunta, V(s)), cada uno de forma (lote,).
        """
        logits1, logits2, value = self(obs)
        dist1 = masked_distribution(logits1, mask1)
        dist2 = masked_distribution(logits2, mask2)
        log_prob = dist1.log_prob(actions[:, 0]) + dist2.log_prob(actions[:, 1])
        entropy = dist1.entropy() + dist2.entropy()
        return log_prob, entropy, value
