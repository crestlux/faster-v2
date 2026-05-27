#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export DISPLAY=

ENVS=(lift can square tool_hang transport)

mkdir -p robomimic/image-only/{lift,can,square,tool_hang,transport}/ph
mkdir -p robomimic/image/{lift,can,square,tool_hang,transport}/ph
mkdir -p robomimic/low_dim/{lift,can,square,tool_hang,transport}/ph

for env in "${ENVS[@]}"; do
  echo "============================================================"
  echo "Processing ${env}..."

  input="robomimic/raw/${env}/ph/demo_v141.hdf5"

  tmp_image="robomimic/raw/${env}/ph/image.hdf5"
  tmp_lowdim="robomimic/raw/${env}/ph/low_dim.hdf5"

  image_only_output="robomimic/image-only/${env}/ph/image_v141.hdf5"
  lowdim_source="robomimic/low_dim/${env}/ph/low_dim_v141.hdf5"

  final_output="robomimic/image/${env}/ph/image_v141.hdf5"
  final_tmp="${final_output}.tmp"

  if [ ! -f "$input" ]; then
    echo "Missing input: $input" >&2
    exit 1
  fi

  # 1. Generate image-only dataset only if missing.
  if [ -f "$image_only_output" ]; then
    echo "Image-only already exists, skipping render: $image_only_output"
  else
    echo "Generating image-only dataset for ${env}..."

    rm -f "$tmp_image"

    python3 -m robomimic.scripts.dataset_states_to_obs \
      --dataset "$input" \
      --output_name image.hdf5 \
      --done_mode 2 \
      --camera_names agentview robot0_eye_in_hand \
      --camera_height 84 \
      --camera_width 84 \
      --compress

    if [ ! -f "$tmp_image" ]; then
      echo "Expected image output not found: $tmp_image" >&2
      exit 1
    fi

    mv "$tmp_image" "$image_only_output"
    echo "Saved image-only: $image_only_output"
  fi

  # 2. Use existing low_dim dataset if present. Otherwise generate only for this env.
  if [ -f "$lowdim_source" ]; then
    echo "low_dim already exists, using: $lowdim_source"
  else
    echo "low_dim missing, generating for ${env}..."

    rm -f "$tmp_lowdim"

    python3 -m robomimic.scripts.dataset_states_to_obs \
      --dataset "$input" \
      --output_name low_dim.hdf5 \
      --done_mode 2

    if [ ! -f "$tmp_lowdim" ]; then
      echo "Expected low_dim output not found: $tmp_lowdim" >&2
      exit 1
    fi

    mv "$tmp_lowdim" "$lowdim_source"
    echo "Saved low_dim: $lowdim_source"
  fi

  # 3. If final image+proprio dataset already has required keys, skip merge.
  if [ -f "$final_output" ]; then
    if FINAL_PATH="$final_output" python3 - <<'PY'
import os
import h5py

path = os.environ["FINAL_PATH"]
required = [
    "agentview_image",
    "robot0_eye_in_hand_image",
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
]

with h5py.File(path, "r") as f:
    demo = sorted(f["data"].keys())[0]
    obs = f[f"data/{demo}/obs"]
    next_obs = f[f"data/{demo}/next_obs"]

    missing = []
    for key in required:
        if key not in obs:
            missing.append(f"obs/{key}")
        if key not in next_obs:
            missing.append(f"next_obs/{key}")

    if missing:
        print("Final file exists but is missing keys:", missing)
        raise SystemExit(1)

print("Final file already valid.")
PY
    then
      echo "Final image+proprio already exists, skipping merge: $final_output"
      continue
    else
      echo "Final file is incomplete, rebuilding: $final_output"
    fi
  fi

  # 4. Merge image-only + proprioception low_dim keys.
  echo "Merging proprioception into final image dataset for ${env}..."

  rm -f "$final_tmp"

  ENV_NAME="$env" \
  IMAGE_ONLY_PATH="$image_only_output" \
  LOWDIM_PATH="$lowdim_source" \
  FINAL_TMP_PATH="$final_tmp" \
  python3 - <<'PY'
import os
import shutil
import h5py

env = os.environ["ENV_NAME"]
image_only_path = os.environ["IMAGE_ONLY_PATH"]
lowdim_path = os.environ["LOWDIM_PATH"]
final_tmp_path = os.environ["FINAL_TMP_PATH"]

low_dim_keys = [
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
]

shutil.copy2(image_only_path, final_tmp_path)

with h5py.File(final_tmp_path, "a") as fout, h5py.File(lowdim_path, "r") as flow:
    image_demos = sorted(fout["data"].keys())
    lowdim_demos = sorted(flow["data"].keys())

    if image_demos != lowdim_demos:
        raise ValueError(
            f"[{env}] demo names differ between image-only and low_dim\n"
            f"image demos: {image_demos[:5]} ... total={len(image_demos)}\n"
            f"lowdim demos: {lowdim_demos[:5]} ... total={len(lowdim_demos)}"
        )

    for demo in image_demos:
        image_T = fout[f"data/{demo}/actions"].shape[0]
        lowdim_T = flow[f"data/{demo}/actions"].shape[0]

        if image_T != lowdim_T:
            raise ValueError(
                f"[{env}] trajectory length mismatch in {demo}: "
                f"image_T={image_T}, lowdim_T={lowdim_T}"
            )

        for group_name in ["obs", "next_obs"]:
            src_path = f"data/{demo}/{group_name}"
            dst_path = f"data/{demo}/{group_name}"

            if src_path not in flow:
                raise KeyError(f"[{env}] missing source group: {src_path}")
            if dst_path not in fout:
                raise KeyError(f"[{env}] missing destination group: {dst_path}")

            src_group = flow[src_path]
            dst_group = fout[dst_path]

            for key in low_dim_keys:
                if key not in src_group:
                    raise KeyError(f"[{env}] missing low_dim key: {src_path}/{key}")

                if key in dst_group:
                    del dst_group[key]

                src_group.copy(key, dst_group)

print(f"[{env}] merged proprio keys into {final_tmp_path}")
PY

  mv "$final_tmp" "$final_output"
  echo "Saved final image+proprio dataset: $final_output"
done