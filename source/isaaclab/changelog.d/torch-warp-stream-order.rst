Added a shared stream scope for mixed Torch/Warp operations, preserving device ordering with the caller and the restored Warp stream without host synchronization.
