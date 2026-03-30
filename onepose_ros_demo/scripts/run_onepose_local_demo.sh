#!/usr/bin/env bash
set -euo pipefail

# One-click reproducible pipeline:
# 1) SfM preprocess
# 2) colcon build
# 3) ros2 launch local_file demo
#
# Example:
#   bash scripts/run_onepose_local_demo.sh
#   bash scripts/run_onepose_local_demo.sh --object test_coffee --backend onnx
#   bash scripts/run_onepose_local_demo.sh --skip-sfm --launch-arg publish_rate_hz:=5.0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

OBJECT_NAME="test_coffee"
BACKEND="onnx"
CONDA_ENV_NAME="onepose"
SKIP_SFM=0
SKIP_BUILD=0
DRY_RUN=0
EXTRA_LAUNCH_ARGS=()

print_cmd() {
  printf '+ '
  printf '%q ' "$@"
  printf '\n'
}

run_cmd() {
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    print_cmd "$@"
    return 0
  fi
  "$@"
}

run_shell() {
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf '+ %s\n' "$*"
    return 0
  fi
  eval "$*"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --object)
      OBJECT_NAME="$2"
      shift 2
      ;;
    --backend)
      BACKEND="$2"
      shift 2
      ;;
    --conda-env)
      CONDA_ENV_NAME="$2"
      shift 2
      ;;
    --skip-sfm)
      SKIP_SFM=1
      shift
      ;;
    --skip-build)
      SKIP_BUILD=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --launch-arg)
      EXTRA_LAUNCH_ARGS+=("$2")
      shift 2
      ;;
    -h|--help)
      cat <<'EOF'
Usage: run_onepose_local_demo.sh [options]

Options:
  --object <name>        Object directory name under data/demo (default: test_coffee)
  --backend <name>       sfm_preprocess backend: onnx|torch_cpu|auto (default: onnx)
  --conda-env <name>     Conda env name (default: onepose)
  --skip-sfm             Skip sfm_preprocess step
  --skip-build           Skip colcon build step
  --dry-run              Print commands only, do not execute
  --launch-arg <k:=v>    Extra ros2 launch args (repeatable)
  -h, --help             Show this help
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

DATA_ROOT="${PKG_ROOT}/data/demo/${OBJECT_NAME}"
SEQ_DIR="${DATA_ROOT}/${OBJECT_NAME}-test"
SFM_MODEL_DIR="${DATA_ROOT}/sfm_model"
ANNOTATE_SEQ="${OBJECT_NAME}-annotate"

if [[ ! -d "${DATA_ROOT}" && "${DRY_RUN}" -eq 0 ]]; then
  echo "[ERROR] data root not found: ${DATA_ROOT}" >&2
  exit 1
elif [[ ! -d "${DATA_ROOT}" && "${DRY_RUN}" -eq 1 ]]; then
  echo "[WARN] data root not found (dry-run continues): ${DATA_ROOT}" >&2
fi

if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  run_shell "source \"${HOME}/miniconda3/etc/profile.d/conda.sh\""
  run_shell "conda activate \"${CONDA_ENV_NAME}\""
fi

run_shell "source /opt/ros/jazzy/setup.bash"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  print_cmd cd "${PKG_ROOT}"
else
  cd "${PKG_ROOT}"
fi

if [[ "${SKIP_SFM}" -eq 0 ]]; then
  echo "[INFO] Running sfm_preprocess (${BACKEND})..."
  run_cmd python "onepose_ros_demo/onnx_demo/sfm_preprocess.py" \
    --data-dir "${DATA_ROOT} ${ANNOTATE_SEQ}" \
    --outputs-dir "${SFM_MODEL_DIR}" \
    --backend "${BACKEND}"
else
  echo "[INFO] Skip sfm_preprocess."
fi

if [[ "${SKIP_BUILD}" -eq 0 ]]; then
  echo "[INFO] Building onepose_ros_demo ..."
  run_cmd colcon build --packages-select onepose_ros_demo --symlink-install
else
  echo "[INFO] Skip colcon build."
fi

run_shell "source install/setup.bash"

echo "[INFO] Launching onepose_ros_demo local_file ..."
run_cmd ros2 launch onepose_ros_demo local_file.launch.py \
  data_root:="${DATA_ROOT}" \
  seq_dir:="${SEQ_DIR}" \
  sfm_model_dir:="${SFM_MODEL_DIR}" \
  "${EXTRA_LAUNCH_ARGS[@]}"
