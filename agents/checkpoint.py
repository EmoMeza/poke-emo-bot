"""Guardado y carga de los pesos del ActorCritic (carpeta checkpoints/)."""

from pathlib import Path
from typing import Optional

import torch

from agents.actor_critic import ActorCritic
from envs.actions import N_ACTIONS
from observations.embedding import OBS_SIZE

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"


def save_checkpoint(
    path: Path,
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    update: int,
    win_rate: Optional[float],
):
    """Guarda pesos + optimizador (para poder retomar) + el tamano de entrada/salida
    con el que se entreno, para detectar un checkpoint incompatible al cargarlo."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "update": update,
            "eval_win_rate": win_rate,
            "obs_size": OBS_SIZE,
            "n_actions": N_ACTIONS,
        },
        path,
    )


def load_model(path: Path) -> ActorCritic:
    """Carga un ActorCritic listo para jugar desde un checkpoint."""
    checkpoint = torch.load(path)
    if (checkpoint["obs_size"], checkpoint["n_actions"]) != (OBS_SIZE, N_ACTIONS):
        raise ValueError(
            f"{path} se entreno con observacion de {checkpoint['obs_size']} y "
            f"{checkpoint['n_actions']} acciones, pero el codigo actual usa "
            f"{OBS_SIZE} y {N_ACTIONS}: hay que reentrenar."
        )
    model = ActorCritic(OBS_SIZE, N_ACTIONS)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model
