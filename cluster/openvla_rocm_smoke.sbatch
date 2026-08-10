#!/bin/bash
#SBATCH --job-name=panda-openvla-smoke
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/openvla-smoke-%j.out
#SBATCH --error=logs/openvla-smoke-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

module purge
module load anaconda/2024
module load rocm/6.2.4

source /apps/software/anaconda3/etc/profile.d/conda.sh
conda activate openvla-rocm
PYTHON_BIN="${CONDA_PREFIX}/bin/python"

MODEL_PATH="${OPENVLA_MODEL_PATH:-/mnt/data/sashishj/models/openvla-7b}"
ATTN_IMPL="${OPENVLA_ATTN_IMPL:-sdpa}"

"$PYTHON_BIN" - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda namespace available:", torch.cuda.is_available())
print("ROCm/HIP version:", torch.version.hip)
print("CUDA version:", torch.version.cuda)
if not torch.cuda.is_available() or not torch.version.hip:
    raise SystemExit("Expected ROCm-enabled PyTorch on an MI300X allocation")
print("device:", torch.cuda.get_device_name(0))
PY

"$PYTHON_BIN" cluster/openvla_server.py \
  --model-path "$MODEL_PATH" \
  --attn-implementation "$ATTN_IMPL" \
  --host 127.0.0.1 --port 8000 &
server_pid=$!
trap 'kill ${server_pid} 2>/dev/null || true' EXIT

for attempt in $(seq 1 60); do
  if curl --fail --silent http://127.0.0.1:8000/healthz; then
    break
  fi
  sleep 2
done

curl --fail http://127.0.0.1:8000/healthz
