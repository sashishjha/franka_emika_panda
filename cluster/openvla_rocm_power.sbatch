#!/bin/bash
# Long-running OpenVLA server job for RBCCPS MI300X allocations.
# Copy this file, replace the placeholders, and submit with sbatch.
# Uses the same conda openvla-rocm environment as openvla_rocm_smoke.sbatch.

#SBATCH --job-name=openvla-power
#SBATCH --output=logs/openvla-power-%j.out
#SBATCH --error=logs/openvla-power-%j.err
#SBATCH --partition=debug
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00

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
HOST="${OPENVLA_HOST:-127.0.0.1}"
PORT="${OPENVLA_PORT:-8000}"

echo "=== Torch / ROCm sanity check ==="
"$PYTHON_BIN" - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch.cuda.is_available():", torch.cuda.is_available())
print("HIP:", torch.version.hip)
print("device_count:", torch.cuda.device_count())
print("device:", torch.cuda.get_device_name(0))
if not torch.cuda.is_available() or not torch.version.hip:
    raise SystemExit("Expected ROCm-enabled PyTorch on an MI300X allocation")
PY

"$PYTHON_BIN" cluster/openvla_server.py \
    --model-path "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --attn-implementation "$ATTN_IMPL" &

SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

echo "=== Waiting for server to become healthy ==="
for attempt in $(seq 1 60); do
    if curl --fail --silent "http://$HOST:$PORT/healthz"; then
        echo
        echo "OpenVLA server is healthy."
        break
    fi
    sleep 2
done

curl --fail "http://$HOST:$PORT/healthz"
echo
echo "OpenVLA server is running on $HOST:$PORT"

# Keep the SLURM job alive while the server runs.
wait "$SERVER_PID"