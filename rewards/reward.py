"""
Calculo del reward de nuestro agente.

No reinventamos la logica de "valor de estado": reutilizamos
PokeEnv.reward_computing_helper (ya trae el manejo de mons debilitados, HP,
estados y victoria/derrota, comparando contra el turno anterior). Aca solo
centralizamos los pesos que usamos nosotros, para poder ajustarlos en un
solo lugar a medida que iteramos el entrenamiento.
"""

from poke_env.battle import AbstractBattle
from poke_env.environment.env import PokeEnv

FAINTED_VALUE = 2.0
HP_VALUE = 1.0
STATUS_VALUE = 0.3
VICTORY_VALUE = 30.0


def calc_reward(env: PokeEnv, battle: AbstractBattle) -> float:
    """
    Reward para el turno actual: diferencia de "valor de estado" respecto
    al turno anterior de esa misma batalla.

    :param env: el entorno (PokeEnv/DoublesEnv) que llama a esta funcion,
        necesario porque reward_computing_helper guarda el valor del turno
        anterior por batalla.
    :param battle: el estado actual de la batalla.
    """
    return env.reward_computing_helper(
        battle,
        fainted_value=FAINTED_VALUE,
        hp_value=HP_VALUE,
        status_value=STATUS_VALUE,
        victory_value=VICTORY_VALUE,
    )
