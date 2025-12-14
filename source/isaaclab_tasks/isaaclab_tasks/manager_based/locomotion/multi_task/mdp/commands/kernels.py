import enum
import torch
from isaaclab.utils.math import quat_error_magnitude


class ACTIVATION_KERNEL_ID(enum.IntEnum):
    TANH = 0
    LESS = 1
    GREATER = 2


class METRIC_KERNEL_ID(enum.IntEnum):
    GEOMETRIC = 0
    QUATERNION = 1


class STATE_KERNEL_ID(enum.IntEnum):
    JOINT_POS = 0
    BODY_VEL = 1
    BODY_POS = 2
    BODY_QUAT = 3


# --- activation kernels (error -> reward/predicate) ---
def tanh_kernel(error, std):
    return 1.0 - torch.tanh(error / std)


def less_kernel(error, threshold):
    return error < threshold


def greater_kernel(error, threshold):
    return error > threshold


ACTIVATION_KERNELS = (tanh_kernel, less_kernel, greater_kernel)


# --- metric kernels (x_cur, x_target -> scalar error) ---
def geometric_error(x_cur, x_target):
    return torch.linalg.vector_norm(x_cur - x_target, dim=-1)


def quaternion_error(q_cur, q_target):
    return quat_error_magnitude(q_cur, q_target)


METRIC_KERNELS = (geometric_error, quaternion_error)
