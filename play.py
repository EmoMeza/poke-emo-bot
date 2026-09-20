"""
Juega contra la IA entrenada.

Levanta un bot que carga los pesos guardados en checkpoints/ y acepta desafios
en el servidor local. Tu juegas desde el navegador y el bot usa el equipo oficial.

Pasos:
    1. Servidor de Pokemon Showdown corriendo en localhost:8000.
    2. python play.py                (el bot queda esperando tu desafio)
    3. Abre http://localhost:8000 en el navegador y elige un nombre cualquiera.
    4. Arma o importa un equipo valido para gen9championsvgc2026regmc (sirve
       cualquiera de teams/data/meta/*.txt: Teambuilder > Import) y escribe en
       el chat: /challenge PokeEmoBot, gen9championsvgc2026regmc
    5. Ctrl+C para salir.

Opciones:
    --checkpoint checkpoints/latest.pt   pesos a usar (por defecto pretrained.pt)
    --name NOMBRE                        nombre del bot en el servidor
    --battles N                          batallas a jugar antes de salir (por defecto 1)
    --stochastic                         la IA muestrea sus jugadas en vez de elegir
                                         siempre la mas probable (juega mas variado)
"""

import argparse
import asyncio
from pathlib import Path

from poke_env.battle import AbstractBattle
from poke_env.environment import DoublesEnv
from poke_env.player import Player
from poke_env.player.battle_order import BattleOrder
from poke_env.ps_client import AccountConfiguration

from agents.actor_critic import ActorCritic
from agents.checkpoint import CHECKPOINT_DIR, load_model
from envs.actions import legal_action_masks, second_slot_mask
from envs.vgc_env import BATTLE_FORMAT
from observations.embedding import embed_battle
from teams.loader import cargar_equipo_oficial


class TrainedPlayer(Player):
    """Jugador de poke-env que decide sus jugadas con la red entrenada."""

    def __init__(self, model: ActorCritic, deterministic: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.deterministic = deterministic

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        action, *_ = self.model.act(
            embed_battle(battle), legal_action_masks(battle), second_slot_mask, self.deterministic
        )
        return DoublesEnv.action_to_order(action, battle, fake=False, strict=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Juega contra la IA entrenada")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_DIR / "pretrained.pt")
    parser.add_argument("--name", default="PokeEmoBot")
    parser.add_argument("--battles", type=int, default=1)
    parser.add_argument("--stochastic", action="store_true")
    return parser.parse_args()


async def main():
    args = parse_args()
    if not args.checkpoint.exists():
        raise SystemExit(f"No existe {args.checkpoint}. Entrena primero con: python train.py")

    bot = TrainedPlayer(
        load_model(args.checkpoint),
        deterministic=not args.stochastic,
        account_configuration=AccountConfiguration(args.name, None),
        battle_format=BATTLE_FORMAT,
        team=cargar_equipo_oficial(),
    )
    print(f"Bot '{args.name}' listo con los pesos de {args.checkpoint.name}.")
    print(f"Abre http://localhost:8000 y desafialo en el formato {BATTLE_FORMAT}. Ctrl+C para salir.")

    await bot.accept_challenges(None, args.battles)
    print(f"Fin: la IA gano {bot.n_won_battles} de {args.battles} batalla(s).")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nSaliendo.")
