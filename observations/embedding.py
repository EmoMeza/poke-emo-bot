"""
Conversion de una DoubleBattle a un vector numerico (la observacion que ve
el MLP).

No usamos one-hot de especie (serian cientos de columnas, casi todas en
cero): en vez de "quien es" cada Pokemon, describimos "como esta" (HP,
estado, boosts, tipos) y sus movimientos por sus propiedades (poder,
precision, tipo) mas el multiplicador de daño contra cada rival activo
(Pokemon.damage_multiplier ya resume toda la tabla de tipos en un numero).

Esto generaliza a equipos que el modelo nunca vio en vez de memorizar
"Salamence hace tal cosa": aprende de las propiedades del movimiento y del
matchup de tipos.
"""

from typing import List, Optional

import numpy as np

from poke_env.battle import (
    DoubleBattle,
    Field,
    Move,
    Pokemon,
    PokemonType,
    SideCondition,
    Status,
    Weather,
)
from poke_env.data import GenData

N_TYPES = len(PokemonType)
N_STATUSES = len(Status)
N_WEATHERS = len(Weather)
N_FIELDS = len(Field)
N_SIDE_CONDITIONS = len(SideCondition)

BOOST_KEYS = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]
N_BOOSTS = len(BOOST_KEYS)

STAT_KEYS = ["hp", "atk", "def", "spa", "spd", "spe"]
N_STATS = len(STAT_KEYS)
STAT_SCALE = 200.0  # normaliza stats tipicas de nivel 50 a un rango ~0-1.5

MOVES_PER_MON = 4
BENCH_SIZE = 2  # en VGC (bring 4) quedan 2 en banca ademas de los 2 activos

POKEMON_FEATURE_SIZE = 1 + N_STATUSES + N_BOOSTS + N_STATS + 2 * N_TYPES + 1  # +1: is_mega
BENCH_FEATURE_SIZE = 1 + N_STATUSES + N_STATS + 1  # +1: is_mega
MOVE_FEATURE_SIZE = 5  # base_power, accuracy, categoria, mult_vs_opp1, mult_vs_opp2

OBS_SIZE = (
    2 * POKEMON_FEATURE_SIZE  # nuestros 2 activos
    + 2 * POKEMON_FEATURE_SIZE  # activos rivales
    + 2 * BENCH_SIZE * BENCH_FEATURE_SIZE  # banca (nuestra + rival)
    + 2 * MOVES_PER_MON * MOVE_FEATURE_SIZE  # movimientos de nuestros activos
    + 2 * 3  # can_dynamax, can_mega_evolve, can_tera (por activo nuestro)
    + 2  # used_mega_evolve (nuestro) y opponent_used_mega_evolve (rival)
    + 2  # trapped (por activo nuestro)
    + 2  # force_switch (por activo nuestro)
    + N_WEATHERS
    + N_FIELDS
    + N_SIDE_CONDITIONS  # nuestro lado
    + N_SIDE_CONDITIONS  # lado rival
)


def _one_hot(index: Optional[int], size: int) -> List[float]:
    vec = [0.0] * size
    if index is not None:
        vec[index] = 1.0
    return vec


def _type_one_hot(pkm_type: Optional[PokemonType]) -> List[float]:
    return _one_hot(pkm_type.value - 1 if pkm_type is not None else None, N_TYPES)


def _status_one_hot(status: Optional[Status]) -> List[float]:
    return _one_hot(status.value - 1 if status is not None else None, N_STATUSES)


def _stat_features(mon: Pokemon, own_side: bool) -> List[float]:
    """
    Stats de HP/Atk/Def/SpA/SpD/Spe.

    Para nuestros pokemon usamos las stats reales (mon.stats), calculadas a
    partir de EVs/nature/nivel, porque las conocemos con precision. Para el
    rival usamos base_stats (el tier de la especie, sin EVs/nature): es la
    misma informacion incompleta que tendria un jugador humano, que no sabe
    la inversion exacta del rival hasta que la infiere jugando.
    """
    if own_side and all(mon.stats.get(key) is not None for key in STAT_KEYS):
        stats = mon.stats
    else:
        stats = mon.base_stats
    return [stats.get(key, 0) / STAT_SCALE for key in STAT_KEYS]


def _is_mega(mon: Pokemon, pokedex: dict) -> float:
    """
    1.0 si el pokemon esta mega evolucionado.

    poke-env no lo expone directamente y lo guarda de dos formas segun el
    lado: en nuestros pokemon la especie pasa a ser la forma mega (forme
    "Mega"/"Mega-X"/"Mega-Y" en el pokedex), y en los del rival mantiene la
    especie base pero Pokemon.mega_evolve() reemplaza sus base_stats por las
    de la mega. Se revisan ambos casos. (Un Transform tambien cambia los
    base_stats, pero no hay Ditto/Mew en nuestros equipos.)
    """
    entry = pokedex.get(mon.species)
    if entry is None:
        return 0.0
    if (entry.get("forme") or "").startswith("Mega"):
        return 1.0
    return float(mon.base_stats != entry["baseStats"])


def _pokemon_features(
    mon: Optional[Pokemon], own_side: bool, pokedex: dict
) -> List[float]:
    """HP, estado, boosts, stats, tipos y si esta mega evolucionado, de un
    Pokemon activo. Vector de ceros si el slot esta vacio (por ejemplo, un
    mon recien debilitado que aun no fue reemplazado)."""
    if mon is None:
        return [0.0] * POKEMON_FEATURE_SIZE
    features = [mon.current_hp_fraction]
    features += _status_one_hot(mon.status)
    features += [mon.boosts.get(key, 0) / 6.0 for key in BOOST_KEYS]
    features += _stat_features(mon, own_side)
    features += _type_one_hot(mon.type_1)
    features += _type_one_hot(mon.type_2)
    features += [_is_mega(mon, pokedex)]
    return features


def _bench_features(
    mon: Optional[Pokemon], own_side: bool, pokedex: dict
) -> List[float]:
    """Version resumida para pokemons que no estan en la cancha: HP, estado,
    stats (su velocidad/bulk importa igual para decidir si conviene
    meterlos) y si ya mega evoluciono (la mega se conserva al salir)."""
    if mon is None:
        return [0.0] * BENCH_FEATURE_SIZE
    return (
        [mon.current_hp_fraction]
        + _status_one_hot(mon.status)
        + _stat_features(mon, own_side)
        + [_is_mega(mon, pokedex)]
    )


def _move_features(
    move: Optional[Move], opponents: List[Optional[Pokemon]]
) -> List[float]:
    """Poder, precision, categoria y que tan efectivo es el movimiento
    contra cada uno de los dos rivales activos."""
    if move is None:
        return [0.0] * MOVE_FEATURE_SIZE
    category_value = {"PHYSICAL": 0.0, "SPECIAL": 0.5, "STATUS": 1.0}[
        move.category.name
    ]
    multipliers = [
        opp.damage_multiplier(move) / 4.0 if opp is not None else 0.0
        for opp in opponents
    ]
    return [move.base_power / 200.0, move.accuracy, category_value] + multipliers


def _pad(mons: List[Pokemon], size: int) -> List[Optional[Pokemon]]:
    padded: List[Optional[Pokemon]] = list(mons[:size])
    padded += [None] * (size - len(padded))
    return padded


def _brought(battle: DoubleBattle) -> List[Pokemon]:
    """
    Nuestros pokemon que realmente estan en la batalla (en VGC se llevan 4 de 6).

    battle.team siempre trae los 6, y los 2 no elegidos en el team preview
    quedan con HP completo para siempre: si no se filtran, se cuelan como
    banca sana. El request del servidor lista solo los llevados.
    """
    idents = {p["ident"] for p in battle.last_request.get("side", {}).get("pokemon", [])}
    brought = [mon for ident, mon in battle.team.items() if ident in idents]
    return brought or list(battle.team.values())


def _bench(mons: List[Pokemon], active: List[Optional[Pokemon]]) -> List[Pokemon]:
    """Los que no estan en la cancha, vivos primero para que el corte a
    BENCH_SIZE no deje afuera a un vivo cuando hay debilitados."""
    return sorted((mon for mon in mons if mon not in active), key=lambda mon: mon.fainted)


def embed_battle(battle: DoubleBattle) -> np.ndarray:
    """Convierte el estado actual de la batalla en el vector de entrada del
    MLP. Ver OBS_SIZE para su largo total."""
    our_active = battle.active_pokemon
    opp_active = battle.opponent_active_pokemon
    pokedex = GenData.from_gen(battle.gen).pokedex

    features: List[float] = []

    for mon in our_active:
        features += _pokemon_features(mon, True, pokedex)
    for mon in opp_active:
        features += _pokemon_features(mon, False, pokedex)

    our_bench = _bench(_brought(battle), our_active)
    opp_bench = _bench([mon for mon in battle.opponent_team.values() if mon.revealed], opp_active)
    for mon in _pad(our_bench, BENCH_SIZE):
        features += _bench_features(mon, True, pokedex)
    for mon in _pad(opp_bench, BENCH_SIZE):
        features += _bench_features(mon, False, pokedex)

    for mon in our_active:
        moves = _pad(list(mon.moves.values()), MOVES_PER_MON) if mon else [None] * MOVES_PER_MON
        for move in moves:
            features += _move_features(move, opp_active)

    features += [float(v) for v in battle.can_dynamax]
    features += [float(v) for v in battle.can_mega_evolve]
    features += [float(v) for v in battle.can_tera]
    features += [float(battle.used_mega_evolve), float(battle.opponent_used_mega_evolve)]
    features += [float(v) for v in battle.trapped]
    features += [float(v) for v in battle.force_switch]

    features += _one_hot_multi(battle.weather.keys(), Weather, N_WEATHERS)
    features += _one_hot_multi(battle.fields.keys(), Field, N_FIELDS)
    features += _one_hot_multi(battle.side_conditions.keys(), SideCondition, N_SIDE_CONDITIONS)
    features += _one_hot_multi(
        battle.opponent_side_conditions.keys(), SideCondition, N_SIDE_CONDITIONS
    )

    vector = np.array(features, dtype=np.float32)
    assert vector.shape == (OBS_SIZE,), f"esperaba {OBS_SIZE}, salio {vector.shape}"
    return vector


def _one_hot_multi(active_keys, enum_cls, size: int) -> List[float]:
    """Vector con un 1 por cada miembro del enum presente en active_keys
    (a diferencia de _one_hot, aca puede haber mas de un 1 a la vez, por
    ejemplo Trick Room + Electric Terrain juntos)."""
    vec = [0.0] * size
    for key in active_keys:
        vec[key.value - 1] = 1.0
    return vec
