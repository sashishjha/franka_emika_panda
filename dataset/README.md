# VLA dataset staging format

`collect_vla_dataset.py` writes one compressed `.npz` file per teacher rollout.
Each file contains the RGB policy observation, instruction, and a seven-value
action label:

```text
[world_dx_m, world_dy_m, world_dz_m, dRx, dRy, dRz, gripper_open]
```

The current teacher has no orientation or gripper trajectory, so its final
four labels are `[0, 0, 0, 1]`. This is intentionally explicit: the present
FL+SMC controller controls Cartesian translation only.

The `metadata_json` field records the action convention and sample rate. Do
not train directly until the dataset has been reviewed and converted to the
selected OpenVLA fine-tuning input format with matching normalization stats.

Generated demonstrations are ignored by Git and should be stored in an
RBCCPS-accessible data location for training.
