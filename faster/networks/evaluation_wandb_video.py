from copy import deepcopy
from typing import Dict

import gym
import numpy as np
import tqdm as tqdm_lib
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import wandb
from robomimic.config import config_factory

from faster.data.robomimic_datasets import (
    ENV_TO_HORIZON_MAP,
    IMAGE_OBS_KEYS,
    LOW_DIM_OBS_KEYS,
    PROPRIOCEPTION_KEYS,
    RobosuiteGymWrapper,
    _patch_robosuite_offscreen_context,
    get_robomimic_env,
)
from faster.evaluation import ActionChunkWrapper


def _load_video_env_meta(dataset_path):
    env_meta = deepcopy(FileUtils.get_env_metadata_from_dataset(dataset_path))
    assert "env_kwargs" in env_meta, sorted(env_meta)
    env_meta["env_kwargs"]["hard_reset"] = False
    env_meta["env_kwargs"]["has_offscreen_renderer"] = True
    return env_meta


class VideoRobosuiteGymWrapper(RobosuiteGymWrapper):
    def render(self, mode, height=None, width=None):
        if mode == "rgb_array":
            self._ensure_offscreen_context()
        return self.env.render(mode=mode, height=height, width=width)

    def _ensure_offscreen_context(self):
        robosuite_env = getattr(self.env, "env", None)
        sim = getattr(robosuite_env, "sim", None)
        if sim is None or getattr(sim, "_render_context_offscreen", None) is not None:
            return

        try:
            from robosuite.utils.binding_utils import MjRenderContextOffscreen

            render_context = MjRenderContextOffscreen(sim, device_id=-1)
            sim.add_render_context(render_context)
        except Exception:
            return


def get_video_robomimic_env(dataset_path, example_action, env_name, render_offscreen=False, use_image_obs=False):
    if not render_offscreen:
        return get_robomimic_env(dataset_path, example_action, env_name, use_image_obs=use_image_obs)

    assert env_name in ENV_TO_HORIZON_MAP, env_name
    _patch_robosuite_offscreen_context()
    config = config_factory(algo_name="iql")
    if use_image_obs:
        config.observation.modalities.obs.rgb = list(IMAGE_OBS_KEYS)
    ObsUtils.initialize_obs_utils_with_config(config)
    env_meta = _load_video_env_meta(dataset_path)
    env = EnvUtils.create_env_from_metadata(env_meta=env_meta, render=False, render_offscreen=True, use_image_obs=use_image_obs)
    obs_keys = PROPRIOCEPTION_KEYS + IMAGE_OBS_KEYS if use_image_obs else LOW_DIM_OBS_KEYS
    return VideoRobosuiteGymWrapper(env, ENV_TO_HORIZON_MAP[env_name], example_action, obs_keys=obs_keys, use_image_obs=use_image_obs)


def _as_video_env(env):
    if isinstance(env, VideoRobosuiteGymWrapper):
        return env
    if isinstance(env, RobosuiteGymWrapper):
        return VideoRobosuiteGymWrapper(env.env, env.horizon, env.action_space)
    return env


class VideoSamplerPolicy:
    def __init__(self, agent):
        self.agent = agent

    def __call__(self, observations, deterministic=False, add_noise=0.0, **kwargs):
        actions = self.agent.eval_actions(observations)
        if isinstance(actions, tuple) and len(actions) == 2:
            actions, self.agent = actions
        return np.asarray(actions)


def _render_rgb(env, height: int = 256, width: int = 256):
    frame = env.render(mode="rgb_array", height=height, width=width)
    if isinstance(frame, (list, tuple)):
        frame = frame[0]
    frame = np.asarray(frame)
    if frame.ndim == 3 and frame.shape[-1] == 4:
        frame = frame[..., :3]
    return frame.astype(np.uint8)


def _sample_eval_trajectories(env, policy, num_episodes: int, max_traj_len: int, record_video: bool):
    env = _as_video_env(env) if record_video else env
    # For action-chunked envs, step the inner env one action at a time during video
    # recording so every simulation frame is captured, not just one per chunk.
    _chunk_env = env if (record_video and isinstance(env, ActionChunkWrapper)) else None
    trajs = []
    render_error = None
    successes = 0
    pbar = tqdm_lib.tqdm(range(num_episodes), desc="eval", unit="ep", dynamic_ncols=True, leave=False)
    for episode_idx in pbar:
        observation = env.reset()
        rewards = []
        steps = []
        frames = []
        done = False
        traj_steps = 0
        should_record = record_video and episode_idx == 0

        if should_record:
            try:
                frames.append(_render_rgb(env))
            except Exception as exc:
                render_error = repr(exc)
                should_record = False

        while not done and traj_steps < max_traj_len:
            action = np.asarray(policy(observation))

            if should_record and _chunk_env is not None:
                # Step each sub-action individually so we capture one frame per
                # simulation step instead of one frame per exec_horizon steps.
                actions_2d = action.reshape(_chunk_env.chunk_size, -1)
                execute = actions_2d[: _chunk_env.exec_horizon]
                total_reward = 0.0
                steps_taken = 0
                info = {}
                for a in execute:
                    observation, sub_reward, done, info = _chunk_env.env.step(a)
                    total_reward += sub_reward
                    steps_taken += 1
                    try:
                        frames.append(_render_rgb(env))
                    except Exception as exc:
                        render_error = repr(exc)
                        should_record = False
                        break
                    if done:
                        break
                reward = total_reward
                if info is None:
                    info = {}
                info["chunk_steps"] = steps_taken
            else:
                observation, reward, done, info = env.step(action)
                if should_record:
                    try:
                        frames.append(_render_rgb(env))
                    except Exception as exc:
                        render_error = repr(exc)
                        should_record = False

            info = {} if info is None else info
            rewards.append(reward)
            step_count = int(info.get("chunk_steps", 1))
            steps.append(step_count)
            traj_steps += step_count

        ep_return = float(np.sum(rewards))
        success = ep_return > 0
        if success:
            successes += 1
        pbar.set_postfix({"ret": f"{ep_return:.2f}", "succ": f"{successes}/{episode_idx+1}"})

        traj = {
            "rewards": np.asarray(rewards, dtype=np.float32),
            "steps": np.asarray(steps, dtype=np.int32),
        }
        if should_record and frames:
            traj["frames"] = np.asarray(frames, dtype=np.uint8)
        if render_error is not None:
            traj["render_error"] = render_error
        trajs.append(traj)
    pbar.close()
    return trajs


def evaluate_robo_with_wandb_video(
    agent,
    env: gym.Env,
    num_episodes: int,
    max_traj_len: int,
    save_video: bool = False,
    return_trajs: bool = False,
) -> Dict[str, float]:
    policy = VideoSamplerPolicy(agent)
    trajs = _sample_eval_trajectories(env, policy, num_episodes, max_traj_len, record_video=save_video)
    lengths = [np.sum(t["steps"]) for t in trajs]
    returns = [np.sum(t["rewards"]) for t in trajs]
    metrics = {"return": float(np.mean(returns)), "length": float(np.mean(lengths))}

    if save_video and len(trajs) > 0 and "frames" in trajs[0]:
        video = trajs[0]["frames"].transpose(0, 3, 1, 2)
        metrics["video"] = wandb.Video(video, fps=20, format="mp4")
        metrics["video_recorded"] = 1.0
        metrics["video_num_frames"] = float(video.shape[0])
    elif save_video:
        metrics["video_recorded"] = 0.0
        metrics["video_num_frames"] = 0.0
        if len(trajs) > 0 and "render_error" in trajs[0]:
            print(f"Video render failed: {trajs[0]['render_error']}")

    if return_trajs:
        return trajs, metrics
    return metrics


def maybe_evaluate_robo_with_wandb_video(agent, env, max_traj_len, num_episodes, step, skip_initial_eval, save_video=False):
    if skip_initial_eval and step == 0:
        metrics = {"return": 0.0, "length": max_traj_len}
        if save_video:
            return metrics
        return metrics
    return evaluate_robo_with_wandb_video(
        agent,
        env,
        max_traj_len=max_traj_len,
        num_episodes=num_episodes,
        save_video=save_video,
    )
