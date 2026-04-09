from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg
from isaaclab.utils import configclass

@configclass
class SuccessEstimatorPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):

    success_estimator_learning_rate: float = 1e-4
    """Learning rate for the success estimator optimizer. Only used with :class:`SuccessEstimatorPPO`."""

    success_loss_coef: float = 1.0
    """Loss coefficient for the success estimator. Only used with :class:`SuccessEstimatorPPO`."""

