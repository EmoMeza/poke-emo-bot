"""
Acciones legales en batallas dobles (gen9).

Cada slot (pokemon activo) elige entre N_ACTIONS acciones:
    0        pasar (solo cuando no hay otra opcion)
    1-6      cambiar al pokemon numero k de battle.team
    7-26     movimiento 1-4, con 5 posibles objetivos cada uno (-2..2)
    27-106   lo mismo con mega evolucion / z-move / dynamax / teracristalizacion
Ver DoublesEnv.action_to_order para el mapeo exacto a ordenes del servidor.

Estas funciones dependen solo de la batalla, no del entorno: las usan tanto el
entrenamiento (envs/vgc_env.py) como el jugador que se enfrenta a una persona
(play.py).
"""

import numpy as np

from poke_env.battle import DoubleBattle
from poke_env.environment import DoublesEnv

N_ACTIONS = 107  # 1 pasar + 6 cambios + 4 movimientos x 5 objetivos x 5 (sin gimmick + 4 gimmicks)


def legal_action_masks(battle: DoubleBattle) -> np.ndarray:
    """
    Mascara booleana (2, N_ACTIONS) con las acciones legales de cada uno de los
    dos slots en el turno actual.

    Cada turno la gran mayoria de las 107 acciones por slot son ilegales
    (movimientos sin PP, objetivos invalidos, cambios a pokemons debilitados o
    que no se llevaron...); enmascararlas evita que el agente gaste el
    entrenamiento aprendiendo a no elegirlas. "Pasar" (accion 0) solo es legal
    cuando no hay ninguna otra opcion, por ejemplo para el slot que no tiene que
    cambiar mientras el otro reemplaza a un debilitado.

    Es la mascara de cada slot por separado; las restricciones entre los dos
    slots (ver second_slot_mask) dependen de lo que elija el primero.
    """
    mask = np.zeros((2, N_ACTIONS), dtype=bool)
    for slot in range(2):
        for action in range(1, N_ACTIONS):
            try:
                DoublesEnv._action_to_order_individual(np.int64(action), battle, False, slot)
            except AssertionError:
                continue
            mask[slot, action] = True
        mask[slot, 0] = not mask[slot, 1:].any()
    return mask


def second_slot_mask(mask: np.ndarray, first_action: int) -> np.ndarray:
    """
    Mascara (N_ACTIONS,) del segundo slot, una vez elegida la accion del
    primero. Quita lo que seria ilegal en combinacion: cambiar al mismo pokemon
    y repetir mega/z/dynamax/tera (solo uno por turno). Si no queda nada, el
    segundo slot pasa (por ejemplo, cuando queda un solo pokemon para reemplazar
    a dos debilitados).

    Se usa para muestrear de a un slot: primero el slot 1 con
    legal_action_masks(battle)[0], y despues el slot 2 con esta mascara.

    :param mask: la mascara del slot 2, legal_action_masks(battle)[1].
    :param first_action: la accion elegida para el slot 1.
    """
    mask = mask.copy()
    if 1 <= first_action <= 6:
        mask[first_action] = False
    elif first_action >= 7:
        gimmick = (first_action - 7) // 20
        if gimmick > 0:
            inicio = 7 + 20 * gimmick
            mask[inicio : inicio + 20] = False
    if not mask[1:].any():
        mask[0] = True
    return mask
