# RGB + OpenVLA integration

> [!IMPORTANT]
> **OpenVLA cannot run on the MacBook.** The model (~14 GB, requires an AMD
> Instinct MI300X + ROCm) runs exclusively on the **RBCCPS cluster**.  
> The local machine runs only: MuJoCo simulation, camera rendering, the action
> adapter, and the FL+SMC torque controller.

## Architecture

```text
MacBook (local)                          RBCCPS cluster
──────────────────────────────           ──────────────────────────────
MuJoCo sim + FL+SMC controller           OpenVLA-7B inference
  │                                        (AMD MI300X + ROCm)
  │── 224×224 RGB + instruction ──────────▶ openvla_server.py (:8000)
  │◀──────────────── 7-D action ──────────
  │
  └── Cartesian action adapter
        └── FL+SMC torque output
```

The MacBook never loads model weights. Only an RGB image and a language
instruction cross the network boundary; MuJoCo state stays local.

The legacy analytic-trajectory controller runs entirely offline (no cluster
needed):

```zsh
source .venv/bin/activate
python fl_smc_controller.py        # modes 1–4: no RBCCPS required
```

## Local setup (simulation + Gemini VLA, no OpenVLA)

```zsh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-local.txt
python fl_smc_controller.py
```

Modes 1–4 (analytic trajectories + Gemini shape selection) run fully
offline. Mode 5 requires the RBCCPS tunnel below.

## Connecting to the RBCCPS OpenVLA service (mode 5)

1. Submit the SLURM job on RBCCPS:
   ```bash
   sbatch cluster/openvla_rocm_power.sbatch
   ```
2. Once the server is healthy, open an SSH tunnel from your MacBook:
   ```zsh
   ssh -N -L 8000:127.0.0.1:8000 <user>@<rbccps-login-host>
   ```
3. Set the endpoint and run mode 5:
   ```zsh
   export OPENVLA_ENDPOINT=http://127.0.0.1:8000/act
   export OPENVLA_UNNORM_KEY=<dataset-statistics-key>
   export OPENVLA_POLICY_HZ=5
   python fl_smc_controller.py 5
   ```

`OPENVLA_UNNORM_KEY` must match the dataset statistics used during
fine-tuning. Do not reuse a BridgeData key for a Panda-specific model.

## Local smoke test (no RBCCPS, validates plumbing only)

Mode 5 also offers a **constant-action smoke test** that exercises the full
camera → async runner → adapter → controller path with a hard-coded action,
without loading any model weights. Select "Local constant-action smoke test"
when prompted. This confirms the integration plumbing works before connecting
to the cluster.

## Current action support

The policy wire action is:

```text
[world_dx_m, world_dy_m, world_dz_m, dRx, dRy, dRz, gripper_open]
```

The first three values drive the existing translational FL+SMC controller;
the seventh drives the gripper actuator. Rotation is validated and logged
but intentionally held until a 6-D pose controller is implemented.

For safety, the default runtime adapter constrains policy targets to a local
workspace of ±0.25 m in X/Y and ±0.20 m in Z around the reset end-effector
pose. Expand those bounds only after validating the target task and
camera/action coordinate frame.

## Dataset collection (offline, no RBCCPS required)

Use the analytic FL+SMC trajectory tracker as an offline teacher:

```zsh
python collect_vla_dataset.py --output-dir dataset/collected --episodes-per-shape 4
```

The collector writes policy-safe RGB images, language instructions, and
action labels in an inspectable `.npz` staging format. Review and convert
these data to the selected OpenVLA fine-tuning format before uploading to
RBCCPS for training. See [dataset/README.md](dataset/README.md) and
[cluster/README.md](cluster/README.md).
