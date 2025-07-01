from __future__ import annotations
from isaaclab.utils import configclass
from dataclasses import MISSING

from .success_monitor import SuccessMonitor


@configclass
class SuccessMonitorCfg:
    
    class_type: type[SuccessMonitor] = SuccessMonitor
    
    monitored_history_len: int = 100
    """The total length of success entry recorded, monitoring table size: (num_monitored_data, monitored_history_len)"""

    num_monitored_data: int = MISSING
    """Number of success monitored. monitoring table size: (num_monitored_data, monitored_history_len)"""
    
    device: str = "cpu"
    """The device used to maintain success table data structure"""
