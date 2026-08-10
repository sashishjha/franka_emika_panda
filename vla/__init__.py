"""Boundaries for visual-language-action integration.

The modules in this package deliberately keep high-level policy inference
separate from the deterministic MuJoCo controller.  They have no OpenVLA
runtime dependency, so the simulator can continue to run on a development
machine while inference runs remotely.
"""

from .action_adapter import ActionAdapterConfig, CartesianActionAdapter, CartesianReference
from .camera import PolicyCamera, PolicyCameraConfig
from .policy import AsyncPolicyRunner, ConstantPolicy, RemoteOpenVLAPolicy

__all__ = [
    "ActionAdapterConfig",
    "AsyncPolicyRunner",
    "CartesianActionAdapter",
    "CartesianReference",
    "ConstantPolicy",
    "PolicyCamera",
    "PolicyCameraConfig",
    "RemoteOpenVLAPolicy",
]
