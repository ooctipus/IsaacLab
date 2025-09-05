from isaaclab.utils import configclass


@configclass
class PbtCfg:
    enabled: bool = False

    policy_idx: int = 0
    """policy index in a population: should always be specified explicitly! Each run in a population should have a unique idx from [0..N-1]"""

    num_policies: int = 8
    """total number of policies in the population, the total number of learners. Override through CLI!"""

    directory: str = ""
    """directory where to store pbt related info"""

    workspace: str = "pbt_workspace"
    """suffix of the workspace dir name inside train_dir, used to distinguish different PBT runs with the same experiment name. Recommended to specify a unique name"""

    interval_steps: int = 100
    """Interval in env steps between PBT iterations (checkpointing, mutation, etc.)"""

    replace_fraction: float = 0.4
    """Fraction of the underperforming policies whose weights are to be replaced by better performing policies
    This is rounded up, i.e. for 8 policies and fraction 0.3 we replace ceil(0.3*8)=3 worst policies"""

    replace_threshold_frac_std: float = 0.1
    """Replace an underperforming policy only if its reward is lower by at least this fraction of standard deviation
    within the population."""

    replace_threshold_frac_absolute: float = 0.1
    """Replace an underperforming policy only if its reward is lower by at least this fraction of the absolute value
    of the objective of a better policy"""

    mutation_rate: float = 0.25
    """Probability to mutate a certain parameter"""

    change_min: float = 1.1
    """min values for the mutation of a parameter
    The mutation is performed by multiplying or dividing (randomly) the parameter value by a value sampledfrom [change_min, change_max]
    """

    change_max: float = 2.0
    """max values for the mutation of a parameter
    The mutation is performed by multiplying or dividing (randomly) the parameter value by a value sampled from [change_min, change_max]
    """

    mutation: dict[str, str] = {}
