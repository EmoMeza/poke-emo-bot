"""
Entorno de entrenamiento: batallas dobles VGC (gen9championsvgc2026regmc).

Es un ParallelEnv de PettingZoo con dos agentes que pelean entre si en el
mismo servidor local (self-play): quien use este entorno decide que politica
controla a cada uno. Cada agente ve la batalla desde su propio lado, con su
propia observacion (observations/embedding.py) y su propio reward
(rewards/reward.py).

Equipos: agent1 juega siempre el equipo oficial; agent2 juega un equipo meta
distinto al azar en cada episodio (teams/data/meta/).

Ejemplo:
    env = VGCDoublesEnv()
    obs, info = env.reset()
    while env.agents:
        actions = {a: env.random_action(a) for a in env.agents}
        obs, rewards, terminated, truncated, info = env.step(actions)
"""

import numpy as np
from gymnasium.spaces import Box

from poke_env.battle import AbstractBattle
from poke_env.environment import DoublesEnv

from envs.actions import N_ACTIONS, legal_action_masks
from envs.actions import second_slot_mask as second_slot_mask_fn
from observations.embedding import OBS_SIZE
from observations.embedding import embed_battle as embed_battle_fn
from rewards.reward import calc_reward as calc_reward_fn
from teams.loader import MetaPoolTeambuilder, cargar_equipo_oficial

BATTLE_FORMAT = "gen9championsvgc2026regmc"


class VGCDoublesEnv(DoublesEnv):
    second_slot_mask = staticmethod(second_slot_mask_fn)

    def __init__(self, **kwargs):
        # strict=False: si una accion es ilegal se juega el orden por defecto
        # del servidor en vez de lanzar una excepcion que corte el entrenamiento.
        kwargs.setdefault("battle_format", BATTLE_FORMAT)
        kwargs.setdefault("strict", False)
        super().__init__(**kwargs)
        assert int(self.action_spaces[self.possible_agents[0]].nvec[0]) == N_ACTIONS

        self.agent1.update_team(cargar_equipo_oficial())
        self.agent2.update_team(MetaPoolTeambuilder())

        self.observation_spaces = {
            agent: Box(-np.inf, np.inf, shape=(OBS_SIZE,), dtype=np.float32)
            for agent in self.possible_agents
        }

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        return embed_battle_fn(battle)

    def calc_reward(self, battle: AbstractBattle) -> float:
        return calc_reward_fn(self, battle)

    def action_masks(self, agent: str) -> np.ndarray:
        """Acciones legales (2, N_ACTIONS) de los dos slots del agente (ver envs/actions.py)."""
        battle = self.battle1 if agent == self.possible_agents[0] else self.battle2
        return legal_action_masks(battle)

    def random_action(self, agent: str) -> np.ndarray:
        """Una accion legal al azar para los dos slots del agente."""
        mask = self.action_masks(agent)
        first = np.random.choice(np.flatnonzero(mask[0]))
        second = np.random.choice(np.flatnonzero(self.second_slot_mask(mask[1], first)))
        return np.array([first, second])
