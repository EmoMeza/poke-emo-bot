"""
Entrenamiento por self-play con PPO.

Cada update: (1) se juegan partidas completas hasta juntar --steps decisiones,
con la MISMA red controlando a los dos agentes; (2) se calculan las ventajas y
se mejora la red con PPO; (3) cada --eval-every updates se mide el win rate
contra un rival que juega acciones legales al azar y se guardan los pesos en
checkpoints/ (latest.pt siempre, best.pt cuando mejora el win rate).

Requisitos: servidor de Pokemon Showdown corriendo en localhost:8000 (ver
docs/como_funciona.md para su configuracion).

Uso:
    python train.py                             # entrenamiento normal
    python train.py --updates 3 --steps 512     # prueba rapida
    tensorboard --logdir runs                   # ver las curvas
"""

import argparse
import time
from datetime import datetime
from typing import Optional

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from agents.actor_critic import ActorCritic
from agents.checkpoint import CHECKPOINT_DIR, save_checkpoint
from agents.ppo import RolloutBuffer, Transition, ppo_update
from envs.actions import N_ACTIONS
from envs.vgc_env import VGCDoublesEnv
from observations.embedding import OBS_SIZE

# El reward de ganar (30) queda en +-1 y los de HP/KO en decimas: mantiene el
# error del critico en una escala que no desestabiliza el entrenamiento.
REWARD_SCALE = 30.0


def collect_rollout(env: VGCDoublesEnv, model: ActorCritic, buffer: RolloutBuffer, min_steps: int) -> dict:
    """Juega partidas completas en self-play hasta tener al menos min_steps decisiones."""
    episodes = wins = 0
    returns = []
    while len(buffer) < min_steps:
        obs, _ = env.reset()
        agents = list(env.agents)
        trajectories = {agent: [] for agent in agents}

        while env.agents:
            to_move = {agents[0]: env.agent1_to_move, agents[1]: env.agent2_to_move}
            actions = {}
            for agent in agents:
                if not to_move[agent]:
                    continue
                mask = env.action_masks(agent)
                action, mask2, log_prob, value = model.act(obs[agent], mask, env.second_slot_mask)
                actions[agent] = action
                trajectories[agent].append(
                    Transition(obs[agent], action, mask[0], mask2, log_prob, value, 0.0)
                )

            obs, rewards, _, _, _ = env.step(actions)
            # El reward de un paso va a la ultima decision del agente, haya movido
            # o no en este paso (por ejemplo, si el rival le bajo HP mientras esperaba).
            for agent in agents:
                if trajectories[agent]:
                    trajectories[agent][-1].reward += rewards[agent] / REWARD_SCALE

        for trajectory in trajectories.values():
            if trajectory:
                buffer.add_trajectory(trajectory)
        episodes += 1
        wins += bool(env.battle1.won)
        returns.append(sum(t.reward for t in trajectories[agents[0]]) * REWARD_SCALE)

    return {"episodes": episodes, "win_rate": wins / episodes, "mean_return": float(np.mean(returns))}


def evaluate_vs_random(env: VGCDoublesEnv, model: Optional[ActorCritic], n_episodes: int) -> float:
    """
    Win rate del agente 1 (equipo oficial) contra un rival que juega acciones
    legales al azar. Con model=None el agente 1 tambien juega al azar: es la
    referencia para saber cuanto de ese win rate es ventaja de equipo.
    """
    wins = 0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        agent1, agent2 = env.agents
        while env.agents:
            actions = {}
            if env.agent1_to_move:
                if model is None:
                    actions[agent1] = env.random_action(agent1)
                else:
                    mask = env.action_masks(agent1)
                    actions[agent1] = model.act(obs[agent1], mask, env.second_slot_mask, deterministic=True)[0]
            if env.agent2_to_move:
                actions[agent2] = env.random_action(agent2)
            obs, *_ = env.step(actions)
        wins += bool(env.battle1.won)
    return wins / n_episodes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrenamiento self-play con PPO")
    parser.add_argument("--updates", type=int, default=200, help="numero de updates de PPO")
    parser.add_argument("--steps", type=int, default=2048, help="decisiones a juntar por update")
    parser.add_argument("--epochs", type=int, default=4, help="pasadas sobre los datos por update")
    parser.add_argument("--minibatch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = VGCDoublesEnv()
    model = ActorCritic(OBS_SIZE, N_ACTIONS)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    buffer = RolloutBuffer(args.gamma, args.lam)
    writer = SummaryWriter(log_dir=f"runs/{datetime.now():%Y%m%d-%H%M%S}")

    baseline = evaluate_vs_random(env, None, args.eval_episodes)
    print(f"Referencia: un agente al azar gana {baseline:.0%} contra otro al azar (ventaja de equipo).")
    writer.add_scalar("eval/win_rate_random_baseline", baseline, 0)

    best_win_rate = -1.0
    for update in range(1, args.updates + 1):
        start = time.time()
        buffer.clear()
        rollout = collect_rollout(env, model, buffer, args.steps)
        metrics = ppo_update(
            model, optimizer, buffer.to_batch(),
            epochs=args.epochs, minibatch_size=args.minibatch, clip=args.clip, ent_coef=args.ent_coef,
        )

        for name, value in {**{f"train/{k}": v for k, v in metrics.items()},
                            "selfplay/win_rate_oficial": rollout["win_rate"],
                            "selfplay/mean_return_oficial": rollout["mean_return"]}.items():
            writer.add_scalar(name, value, update)

        print(
            f"update {update}/{args.updates} | {rollout['episodes']} partidas, {len(buffer)} decisiones "
            f"| return oficial {rollout['mean_return']:+.1f} | entropia {metrics['entropy']:.2f} "
            f"| kl {metrics['approx_kl']:.4f} | error critico {metrics['value_loss']:.3f} "
            f"| {time.time() - start:.0f}s"
        )

        if update % args.eval_every == 0 or update == args.updates:
            win_rate = evaluate_vs_random(env, model, args.eval_episodes)
            writer.add_scalar("eval/win_rate_vs_random", win_rate, update)
            print(f"  -> win rate contra rival al azar: {win_rate:.0%} (referencia al azar: {baseline:.0%})")
            save_checkpoint(CHECKPOINT_DIR / "latest.pt", model, optimizer, update, win_rate)
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                save_checkpoint(CHECKPOINT_DIR / "best.pt", model, optimizer, update, win_rate)

    env.close()
    writer.close()


if __name__ == "__main__":
    main()
