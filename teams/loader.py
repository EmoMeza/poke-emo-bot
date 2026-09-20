"""
Carga de equipos (formato showdown export) para pasarselos a los Player de poke-env.

La idea: nuestro agente siempre entrena con el equipo oficial (data/oficial.txt),
mientras que el rival usa uno de varios equipos meta (data/meta/*.txt), elegido
al azar. Asi el agente no se acostumbra a jugar siempre contra el mismo equipo
rival, sino contra una muestra representativa del meta.
"""

import random
from pathlib import Path
from typing import List, Optional

from poke_env.teambuilder import ConstantTeambuilder, Teambuilder

from teams.pokedex_patch import aplicar_parche_pokedex

# Se aplica al importar: cualquier batalla que use estos equipos necesita las
# megas que le faltan a poke-env, y olvidarlo deja la batalla colgada.
aplicar_parche_pokedex()

_DATA_DIR = Path(__file__).parent / "data"
_OFICIAL_PATH = _DATA_DIR / "oficial.txt"
_META_DIR = _DATA_DIR / "meta"


def cargar_equipo_oficial() -> str:
    """Devuelve el equipo oficial (el que usa nuestro agente) como string showdown."""
    return _OFICIAL_PATH.read_text(encoding="utf-8")


def cargar_equipo_meta(indice: Optional[int] = None) -> str:
    """
    Devuelve un equipo meta (pensado para el rival), como string showdown.

    :param indice: si se especifica, devuelve ese equipo en particular (util para
        evaluar contra un equipo fijo). Si es None, elige uno al azar entre todos
        los equipos disponibles en data/meta/.
    """
    archivos = _archivos_meta()
    archivo = archivos[indice] if indice is not None else random.choice(archivos)
    return archivo.read_text(encoding="utf-8")


def _archivos_meta() -> List[Path]:
    archivos = sorted(_META_DIR.glob("*.txt"))
    if not archivos:
        raise FileNotFoundError(
            f"No hay equipos meta en {_META_DIR}. Agrega al menos un .txt ahi."
        )
    return archivos


class MetaPoolTeambuilder(Teambuilder):
    """
    Entrega un equipo meta distinto al azar cada vez que arranca una batalla.

    poke-env le pide el equipo al Teambuilder de cada Player en cada batalla
    nueva, asi que pasarselo al rival (Player.update_team) hace que rote entre
    los equipos de data/meta/ episodio a episodio, sin recrear nada.
    """

    def __init__(self):
        self._equipos = [
            ConstantTeambuilder(archivo.read_text(encoding="utf-8")).yield_team()
            for archivo in _archivos_meta()
        ]

    def yield_team(self) -> str:
        return random.choice(self._equipos)
