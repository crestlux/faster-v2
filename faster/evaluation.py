from typing import Dict

import gym
import numpy as np


class ActionChunkWrapper:
    """Wraps an env so a flat (chunk_size * action_dim) action is executed step-by-step.

    Temporal ensemble / receding horizon:
        chunk_size  — how many actions the policy predicts at once
        exec_horizon — how many of those actions to actually execute before replanning
                       (default: chunk_size → classic open-loop chunk execution)

    With exec_horizon < chunk_size the policy is queried more frequently, giving it
    more chances to react to the current observation (closed-loop / receding horizon).
    The actor still learns to predict full chunk_size sequences from demonstrations,
    but the critic/Q-function sees exec_horizon-step returns (consistent with execution).
    """

    def __init__(self, env, chunk_size: int, exec_horizon: int = None):
        self.env = env
        self.chunk_size = chunk_size
        self.exec_horizon = exec_horizon if exec_horizon is not None else chunk_size
        assert 1 <= self.exec_horizon <= self.chunk_size, (
            f"exec_horizon={self.exec_horizon} must be in [1, chunk_size={self.chunk_size}]"
        )

    def step(self, flat_action: np.ndarray):
        actions = flat_action.reshape(self.chunk_size, -1)
        execute = actions[: self.exec_horizon]   # only the prefix
        total_reward = 0.0
        steps_taken = 0
        obs = done = info = None
        for a in execute:
            obs, reward, done, info = self.env.step(a)
            total_reward += reward
            steps_taken += 1
            if done:
                break
        info = {} if info is None else info
        info["chunk_steps"] = steps_taken
        return obs, total_reward, done, info

    def reset(self):
        return self.env.reset()

    def __getattr__(self, name):
        return getattr(self.env, name)


class SamplerPolicy:
    def __init__(self, agent):
        self.agent = agent

    def __call__(self, observations, deterministic=False, add_noise=0.0, **kwargs):
        actions = self.agent.eval_actions(observations)
        if isinstance(actions, tuple) and len(actions) == 2:
            actions, self.agent = actions
        return np.asarray(actions)


class TrajSampler:
    def __init__(self, env, max_traj_length=1000):
        self._env = env
        self.max_traj_length = max_traj_length

    def sample(self, policy, n_trajs, deterministic=False, add_noise=0.0, filter=False):
        trajs = []
        for _ in range(n_trajs):
            observation = self._env.reset()
            rewards = []
            steps = []
            traj_steps = 0
            done = False

            while not done and traj_steps < self.max_traj_length:
                action = np.asarray(policy(observation, deterministic=deterministic, add_noise=add_noise))
                observation, reward, done, info = self._env.step(action)
                info = {} if info is None else info
                rewards.append(reward)
                step_count = int(info.get("chunk_steps", 1))
                steps.append(step_count)
                traj_steps += step_count

            if filter and not np.sum(rewards) > 0:
                continue

            trajs.append({"rewards": np.asarray(rewards, dtype=np.float32), "steps": np.asarray(steps, dtype=np.int32)})

        return trajs

    @property
    def env(self):
        return self._env


def evaluate_robo(
    agent, env: gym.Env, num_episodes: int, max_traj_len: int, save_video: bool = False, return_trajs: bool = False
) -> Dict[str, float]:
    sampler = TrajSampler(env, max_traj_len)
    policy = SamplerPolicy(agent)
    trajs = sampler.sample(policy, num_episodes)
    lengths = [np.sum(t["steps"]) for t in trajs]
    returns = [np.sum(t["rewards"]) for t in trajs]
    metrics = {"return": float(np.mean(returns)), "length": float(np.mean(lengths))}
    if return_trajs:
        return trajs, metrics
    return metrics
