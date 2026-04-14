import warp as wp


# --------------------------------------------------------------------------- #
# Elementwise Warp ops
# --------------------------------------------------------------------------- #
@wp.func
def _tanh_op(x: float, std: float):
    return 1.0 - wp.tanh(x / std)


@wp.func
def _less_op(x: float, threshold: float):
    return x < threshold


@wp.func
def _greater_op(x: float, threshold: float):
    return x > threshold


# --------------------------------------------------------------------------- #
# Kernels
# --------------------------------------------------------------------------- #
@wp.kernel
def tanh_kernel_scalar_std(
    x: wp.array(dtype=wp.float32),
    std: float,
    out: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    out[tid] = _tanh_op(x[tid], std)


@wp.kernel
def tanh_kernel_array_std(
    x: wp.array(dtype=wp.float32),
    std: wp.array(dtype=wp.float32),
    out: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    out[tid] = _tanh_op(x[tid], std[tid])


@wp.kernel
def less_kernel_scalar_threshold(
    x: wp.array(dtype=wp.float32),
    threshold: float,
    out: wp.array(dtype=wp.bool),
):
    tid = wp.tid()
    out[tid] = _less_op(x[tid], threshold)


@wp.kernel
def less_kernel_array_threshold(
    x: wp.array(dtype=wp.float32),
    threshold: wp.array(dtype=wp.float32),
    out: wp.array(dtype=wp.bool),
):
    tid = wp.tid()
    out[tid] = _less_op(x[tid], threshold[tid])


@wp.kernel
def greater_kernel_scalar_threshold(
    x: wp.array(dtype=wp.float32),
    threshold: float,
    out: wp.array(dtype=wp.bool),
):
    tid = wp.tid()
    out[tid] = _greater_op(x[tid], threshold)


@wp.kernel
def greater_kernel_array_threshold(
    x: wp.array(dtype=wp.float32),
    threshold: wp.array(dtype=wp.float32),
    out: wp.array(dtype=wp.bool),
):
    tid = wp.tid()
    out[tid] = _greater_op(x[tid], threshold[tid])
