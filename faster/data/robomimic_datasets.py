from copy import deepcopy

import numpy as np
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.config import config_factory

from faster.data.dataset import Dataset

# For low_dim, we use "object" state. For images, we use vision + proprioception only
PROPRIOCEPTION_KEYS = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")
LOW_DIM_OBS_KEYS = PROPRIOCEPTION_KEYS + ("object",)
IMAGE_OBS_KEYS = ("agentview_image", "robot0_eye_in_hand_image")
ENV_TO_HORIZON_MAP = {"lift": 400, "can": 400, "square": 400, "transport": 700, "tool_hang": 700}


def _patch_robosuite_offscreen_context():
    import robosuite.utils.binding_utils as binding_utils

    render_cls = binding_utils.MjRenderContext
    if getattr(render_cls, "_sample_rank_make_current_patch", False):
        return

    original_render = render_cls.render
    original_read_pixels = render_cls.read_pixels

    def patched_render(self, *args, **kwargs):
        self.gl_ctx.make_current()
        return original_render(self, *args, **kwargs)

    def patched_read_pixels(self, *args, **kwargs):
        self.gl_ctx.make_current()
        return original_read_pixels(self, *args, **kwargs)

    render_cls.render = patched_render
    render_cls.read_pixels = patched_read_pixels
    render_cls._sample_rank_make_current_patch = True


def _load_robomimic_env_meta(dataset_path):
    env_meta = deepcopy(FileUtils.get_env_metadata_from_dataset(dataset_path))
    assert "env_kwargs" in env_meta, sorted(env_meta)
    env_meta["env_kwargs"]["hard_reset"] = False
    return env_meta


def _reset_robomimic_playback_env(env):
    if hasattr(env, "env") and hasattr(env.env, "hard_reset"):
        env.env.hard_reset = False
    env.reset()
    state_dict = env.get_state()
    assert "states" in state_dict, sorted(state_dict)
    return env.reset_to({"states": state_dict["states"]})


class RobosuiteGymWrapper:
    def __init__(self, env, horizon, example_action, obs_keys=LOW_DIM_OBS_KEYS, use_image_obs=False):
        self.env = env
        self.horizon = horizon
        self.action_space = example_action
        self.obs_keys = obs_keys
        self.use_image_obs = use_image_obs
        self.timestep = 0
        self.returns = 0.0

    def step(self, action):
        next_obs, reward, done, _ = self.env.step(action)
        next_obs = self._process_obs(next_obs)
        success = self.env.is_success()["task"]
        self.timestep += 1
        self.returns += reward
        timeout = self.timestep >= self.horizon
        terminated = done or success or timeout
        info = None
        if terminated:
            info = {"episode": {"return": self.returns, "length": self.timestep}}
            if timeout and not success:
                info["TimeLimit.truncated"] = True
        return next_obs, reward, terminated, info

    def reset(self):
        obs = _reset_robomimic_playback_env(self.env)
        obs = self._process_obs(obs)
        self.timestep = 0
        self.returns = 0.0
        return obs

    def render(self, mode, height=None, width=None):
        return self.env.render(mode=mode, height=height, width=width)

    def _process_obs(self, obs):
        if self.use_image_obs:
            state = np.concatenate([obs[key] for key in PROPRIOCEPTION_KEYS], axis=-1)
            # Assuming agentview_image is the primary image. If using multiple, we can stack or concatenate along channel
            # We'll concatenate them along the channel dimension for simplicity
            images = []
            for key in IMAGE_OBS_KEYS:
                img = obs[key]
                if img.ndim == 3 and img.shape[0] == 3:
                    img = img.transpose(1, 2, 0)
                images.append(img)
            image = np.concatenate(images, axis=-1)
            return {"state": state, "image": image}
        else:
            return np.concatenate([obs[key] for key in self.obs_keys], axis=-1)


def process_robomimic_dataset(seq_dataset, use_image_obs=False):
    cached = seq_dataset.getitem_cache
    
    if use_image_obs:
        obs_states, next_obs_states = [], []
        obs_images, next_obs_images = [], []
    else:
        observations = []
        next_observations = []

    actions = []
    rewards = []
    terminals = []

    for item in cached:
        if use_image_obs:
            obs_states.append(np.concatenate([item["obs"][key] for key in PROPRIOCEPTION_KEYS], axis=1))
            next_obs_states.append(np.concatenate([item["next_obs"][key] for key in PROPRIOCEPTION_KEYS], axis=1))
            
            o_imgs = []
            n_imgs = []
            for key in IMAGE_OBS_KEYS:
                o_img = item["obs"][key]
                if o_img.ndim == 4 and o_img.shape[1] == 3:
                    o_img = o_img.transpose(0, 2, 3, 1)
                elif o_img.ndim == 3 and o_img.shape[0] == 3:
                    o_img = o_img.transpose(1, 2, 0)
                o_imgs.append(o_img)
                
                n_img = item["next_obs"][key]
                if n_img.ndim == 4 and n_img.shape[1] == 3:
                    n_img = n_img.transpose(0, 2, 3, 1)
                elif n_img.ndim == 3 and n_img.shape[0] == 3:
                    n_img = n_img.transpose(1, 2, 0)
                n_imgs.append(n_img)
                
            obs_images.append(np.concatenate(o_imgs, axis=-1))
            next_obs_images.append(np.concatenate(n_imgs, axis=-1))
        else:
            observations.append(np.concatenate([item["obs"][key] for key in LOW_DIM_OBS_KEYS], axis=1))
            next_observations.append(np.concatenate([item["next_obs"][key] for key in LOW_DIM_OBS_KEYS], axis=1))

        actions.append(np.asarray(item["actions"]))
        rewards.append(np.asarray(item["rewards"]))
        terminals.append(np.asarray(item["dones"]))

    actions = np.concatenate(actions).astype(np.float32)
    rewards = np.concatenate(rewards).astype(np.float32)
    terminals = np.concatenate(terminals).astype(np.float32)

    if use_image_obs:
        return {
            "observations": {
                "state": np.concatenate(obs_states).astype(np.float32),
                "image": np.concatenate(obs_images).astype(np.uint8),  # keep uint8 to save memory
            },
            "actions": actions,
            "rewards": rewards,
            "terminals": terminals,
            "next_observations": {
                "state": np.concatenate(next_obs_states).astype(np.float32),
                "image": np.concatenate(next_obs_images).astype(np.uint8),
            },
        }
    else:
        return {
            "observations": np.concatenate(observations).astype(np.float32),
            "actions": actions,
            "rewards": rewards,
            "terminals": terminals,
            "next_observations": np.concatenate(next_observations).astype(np.float32),
        }


def get_robomimic_env(dataset_path, example_action, env_name, use_image_obs=False):
    assert env_name in ENV_TO_HORIZON_MAP, env_name
    _patch_robosuite_offscreen_context()
    config = config_factory(algo_name="iql")
    if use_image_obs:
        config.observation.modalities.obs.rgb = list(IMAGE_OBS_KEYS)
    ObsUtils.initialize_obs_utils_with_config(config)
    env_meta = _load_robomimic_env_meta(dataset_path)
    env = EnvUtils.create_env_from_metadata(env_meta=env_meta, render=False, render_offscreen=use_image_obs, use_image_obs=use_image_obs)
    obs_keys = PROPRIOCEPTION_KEYS + IMAGE_OBS_KEYS if use_image_obs else LOW_DIM_OBS_KEYS
    return RobosuiteGymWrapper(env, ENV_TO_HORIZON_MAP[env_name], example_action, obs_keys=obs_keys, use_image_obs=use_image_obs)


def _episode_dones(observations, next_observations, terminals, ignore_done):
    dones = np.zeros_like(terminals, dtype=np.float32)
    # Handle both dict (image + state) and flat array (lowdim)
    if isinstance(observations, dict) and "state" in observations:
        obs_array = observations["state"]
        next_obs_array = next_observations["state"]
    else:
        obs_array = observations
        next_obs_array = next_observations

    for i in range(len(dones) - 1):
        transition_break = np.linalg.norm(obs_array[i + 1] - next_obs_array[i]) > 1e-6
        if ignore_done:
            dones[i] = float(transition_break)
        else:
            dones[i] = float(transition_break or terminals[i] == 1.0)
    dones[-1] = 1.0
    return dones


def _truncate_dataset_by_episodes(dataset_dict, num_data):
    done_indices = [-1] + [i for i, done in enumerate(dataset_dict["dones"]) if done]
    keep = []
    for i in range(len(done_indices) - 1):
        if done_indices[i] + 1 < done_indices[i + 1]:
            keep.append(done_indices[i])
    keep.append(done_indices[-1])
    total_len = keep[num_data] - keep[0]
    for key, value in dataset_dict.items():
        if isinstance(value, dict):
            for k in value.keys():
                value[k] = value[k][:total_len]
        else:
            dataset_dict[key] = value[:total_len]


def make_chunk_dataset(dataset_dict: dict, chunk_size: int, exec_horizon: int = None) -> dict:
    """Convert a step-level dataset into a chunk-level dataset.

    Each valid chunk starts at step t and spans [t, t+chunk_size).
    Chunks that would cross an episode boundary (any dones[t..t+chunk_size-2] == 1)
    are dropped so the model never trains on cross-episode transitions.

    Args:
        chunk_size:    Number of actions the policy predicts. The stored 'actions' field
                       is a flattened (chunk_size * orig_action_dim,) vector so the actor
                       can learn to predict full behavioural sequences from demonstrations.
        exec_horizon:  Number of actions actually executed before replanning
                       (temporal ensemble / receding horizon). Must satisfy
                       1 <= exec_horizon <= chunk_size.
                       Defaults to chunk_size (classic open-loop chunking).
                       Controls the Bellman backup horizon:
                         next_observations  <- obs after exec_horizon steps
                         rewards            <- sum of first exec_horizon rewards
                         masks / dones      <- status at step exec_horizon

    Returns a new dataset dict with:
      observations       <- obs at the start of the chunk
      actions            <- flattened (chunk_size * orig_action_dim,) — full prediction target
      rewards            <- sum of rewards over exec_horizon steps
      masks              <- mask at step exec_horizon
      dones              <- done flag at step exec_horizon
      next_observations  <- obs after exec_horizon steps  (RL value horizon)
    """
    if chunk_size == 1:
        return dataset_dict
    if exec_horizon is None:
        exec_horizon = chunk_size
    assert 1 <= exec_horizon <= chunk_size, (
        f"exec_horizon={exec_horizon} must be in [1, chunk_size={chunk_size}]"
    )

    dones = dataset_dict["dones"]
    N = len(dones)

    # Valid-start: no episode boundary in first chunk_size-1 steps so the full action
    # prediction sequence is drawn from a single episode.
    done_cumsum = np.concatenate([[0], np.cumsum(dones)])
    t_range = np.arange(N - chunk_size + 1, dtype=np.int64)
    mid_done_sums = done_cumsum[t_range + chunk_size - 1] - done_cumsum[t_range]
    valid_starts = t_range[mid_done_sums == 0]

    # Full action sequence: (n_chunks, chunk_size) → flattened
    action_indices = valid_starts[:, None] + np.arange(chunk_size, dtype=np.int64)[None, :]
    chunk_actions = dataset_dict["actions"][action_indices].reshape(len(valid_starts), -1).astype(np.float32)

    # RL value horizon: exec_horizon steps
    exec_indices = valid_starts[:, None] + np.arange(exec_horizon, dtype=np.int64)[None, :]
    chunk_rewards = dataset_dict["rewards"][exec_indices].sum(axis=1).astype(np.float32)
    ends = valid_starts + exec_horizon - 1   # last executed step index

    obs = dataset_dict["observations"]
    next_obs = dataset_dict["next_observations"]
    if isinstance(obs, dict):
        chunk_obs = {k: v[valid_starts] for k, v in obs.items()}
        chunk_next_obs = {k: v[ends] for k, v in next_obs.items()}
    else:
        chunk_obs = obs[valid_starts]
        chunk_next_obs = next_obs[ends]

    return {
        "observations": chunk_obs,
        "actions": chunk_actions,
        "rewards": chunk_rewards,
        "masks": dataset_dict["masks"][ends].astype(np.float32),
        "dones": dataset_dict["dones"][ends].astype(np.float32),
        "next_observations": chunk_next_obs,
    }


class RoboD4RLDataset(Dataset):
    def __init__(self, env, clip_to_eps=True, eps=1e-5, num_data=0, ignore_done=False, custom_dataset=None):
        assert custom_dataset is not None, "Public release RoboD4RLDataset only supports custom_dataset input."
        dataset = {}
        for key, value in custom_dataset.items():
            if isinstance(value, dict):
                dataset[key] = {k: np.asarray(v).copy() for k, v in value.items()}
            else:
                dataset[key] = np.asarray(value).copy()

        if clip_to_eps:
            lim = 1 - eps
            dataset["actions"] = np.clip(dataset["actions"], -lim, lim)

        dones = _episode_dones(dataset["observations"], dataset["next_observations"], dataset["terminals"], ignore_done)
        
        dataset_dict = {
            "actions": dataset["actions"].astype(np.float32),
            "rewards": dataset["rewards"].astype(np.float32),
            "masks": 1.0 - dataset["terminals"].astype(np.float32),
            "dones": dones.astype(np.float32),
        }
        
        if isinstance(dataset["observations"], dict):
            dataset_dict["observations"] = {k: v.copy() for k, v in dataset["observations"].items()}
            dataset_dict["next_observations"] = {k: v.copy() for k, v in dataset["next_observations"].items()}
        else:
            dataset_dict["observations"] = dataset["observations"].astype(np.float32)
            dataset_dict["next_observations"] = dataset["next_observations"].astype(np.float32)

        if num_data != 0:
            _truncate_dataset_by_episodes(dataset_dict, num_data)
        super().__init__(dataset_dict)
