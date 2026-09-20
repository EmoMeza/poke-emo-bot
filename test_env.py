"""
Prueba rapida del entorno de entrenamiento (envs/vgc_env.py).

Juega N_EPISODES episodios completos en self-play donde los dos agentes eligen
acciones legales al azar (usando las mascaras del entorno) y muestra, por
episodio: cuantos turnos duro, si gano el agente con el equipo oficial, el
reward total de cada agente y cuantas veces la combinacion de acciones de los
dos slots resulto ilegal (en ese caso el entorno juega el orden por defecto).

Requisitos:
- Servidor de Pokemon Showdown corriendo en localhost:8000, con
  `noguestsecurity` y `nothrottle` en true en config/config.js (sin esto, los
  episodios seguidos se cuelgan).

Uso:
    python test_env.py
"""

import time

import numpy as np

from envs.vgc_env import VGCDoublesEnv

N_EPISODES = 3


def accion_conjunta_ilegal(env: VGCDoublesEnv, accion: np.ndarray, battle) -> bool:
    """True si la combinacion de los dos slots es ilegal aunque cada una lo sea por separado."""
    try:
        env.action_to_order(accion, battle, fake=False, strict=True)
    except AssertionError:
        return True
    return False


def main():
    env = VGCDoublesEnv()

    for episodio in range(1, N_EPISODES + 1):
        inicio = time.time()
        obs, _ = env.reset()
        agentes = list(env.agents)
        for agente in agentes:
            assert env.observation_space(agente).contains(obs[agente])

        pasos = 0
        ilegales = 0
        retornos = {agente: 0.0 for agente in agentes}

        while env.agents:
            acciones = {agente: env.random_action(agente) for agente in agentes}
            if env.agent1_to_move:
                ilegales += accion_conjunta_ilegal(env, acciones[agentes[0]], env.battle1)
            if env.agent2_to_move:
                ilegales += accion_conjunta_ilegal(env, acciones[agentes[1]], env.battle2)

            obs, rewards, _, _, _ = env.step(acciones)
            pasos += 1
            for agente in agentes:
                retornos[agente] += rewards[agente]

        print(
            f"Episodio {episodio}: {pasos} pasos en {time.time() - inicio:.0f}s | "
            f"gana oficial: {env.battle1.won} | "
            f"reward oficial {retornos[agentes[0]]:+.1f}, rival {retornos[agentes[1]]:+.1f} | "
            f"acciones conjuntas ilegales: {ilegales}"
        )

    env.close()


if __name__ == "__main__":
    main()
