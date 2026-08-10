# RBCCPS OpenVLA / ROCm deployment

The MacBook runs MuJoCo, camera rendering, the action adapter, and the
low-level FL+SMC controller.  RBCCPS runs only OpenVLA inference on an AMD
MI300X allocation.

## Before submitting a job

Obtain these site-specific values from RBCCPS documentation or support:

- SLURM partition and project account;
- GPU request syntax and CPU/memory limits;
- supported ROCm/PyTorch module or Apptainer/Singularity image;
- access to Hugging Face model weights and a persistent cache location;
- compute-node networking policy and permitted SSH tunnel route.

Do not substitute a CUDA environment. PyTorch's ROCm/HIP build intentionally
uses the `torch.cuda` namespace, but it must report a non-empty
`torch.version.hip` value in the allocated job.

## Smoke test

1. Copy `openvla_rocm_smoke.sbatch` to a user-owned file.
2. Replace each `<...>` placeholder.
3. Run `sbatch <your-file>.sbatch`.
4. Confirm the log reports a HIP version, an AMD GPU name, and a successful
   `/healthz` response.
5. If `flash_attention_2` fails, rerun using `--attn-implementation sdpa`.
   Do not silently change action semantics or model weights while debugging.

## Start the service

Inside an interactive GPU allocation or approved persistent SLURM job:

```bash
conda activate openvla-rocm
python cluster/openvla_server.py \
    --model-path /mnt/data/sashishj/models/openvla-7b \
    --host 127.0.0.1 --port 8000 \
    --attn-implementation sdpa
```

The service intentionally binds to loopback. If policy inference must be
requested from the MacBook, use an RBCCPS-approved SSH tunnel rather than
opening a public port:

```bash
ssh -N -L 8000:127.0.0.1:8000 <user>@<rbccps-login-host>
```

The exact tunnel route can differ when a compute node is not directly
reachable; follow RBCCPS policy.

## Client contract

`POST /act` receives:

```json
{
  "image": {
    "encoding": "raw-rgb-base64",
    "shape": [224, 224, 3],
    "data": "..."
  },
  "instruction": "move toward the visible target",
  "unnorm_key": "optional-dataset-statistics-key"
}
```

It returns:

```json
{"action": [dx, dy, dz, dRx, dRy, dRz, gripper]}
```

The action must be de-normalized with the correct training-dataset statistics
before it reaches the simulator. The local adapter interprets the first three
components as a world-frame Cartesian displacement in metres. It currently
holds orientation fixed and maps the seventh component to the existing Panda
gripper actuator.
