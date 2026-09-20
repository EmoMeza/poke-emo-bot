"""
Parche a la data de poke-env.

poke-env 0.10.0 (la ultima version) no trae en su pokedex las megas nuevas de
Champions (Dragonite-Mega, Floette-Mega, Froslass-Mega, Scovillain-Mega...).
Cuando una de ellas mega evoluciona en batalla, poke-env falla con KeyError al
procesar el mensaje detailschange del servidor y la batalla queda colgada.

data/extra_pokedex.json contiene las especies que existen en el servidor local
de Showdown y faltan en poke-env, con las mismas claves que usa su pokedex
(baseStats, types, abilities, heightm, weightkg, ...).
"""

import json
from pathlib import Path

from poke_env.data import GenData

_EXTRA_POKEDEX_PATH = Path(__file__).parent / "data" / "extra_pokedex.json"


def aplicar_parche_pokedex(gen: int = 9) -> None:
    """Agrega al pokedex de poke-env las especies que le faltan. Es idempotente."""
    extra = json.loads(_EXTRA_POKEDEX_PATH.read_text(encoding="utf-8"))
    pokedex = GenData.from_gen(gen).pokedex
    for species_id, entry in extra.items():
        pokedex.setdefault(species_id, entry)
