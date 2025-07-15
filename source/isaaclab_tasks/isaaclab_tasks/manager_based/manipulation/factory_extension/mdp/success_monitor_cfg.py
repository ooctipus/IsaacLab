from __future__ import annotations
from isaaclab.utils import configclass
from dataclasses import MISSING

from .success_monitor import SuccessMonitor, SequentialDataSuccessMonitor


@configclass
class SuccessMonitorCfg:
    
    class_type: type[SuccessMonitor] = SuccessMonitor
    
    monitored_history_len: int = 100
    """The total length of success entry recorded, monitoring table size: (num_monitored_data, monitored_history_len)"""

    num_monitored_data: int = MISSING
    """Number of success monitored. monitoring table size: (num_monitored_data, monitored_history_len)"""
    
    device: str = "cpu"
    """The device used to maintain success table data structure"""


@configclass
class SequentialDataSuccessMonitorCfg(SuccessMonitorCfg):

    class_type: type[SequentialDataSuccessMonitor] = SequentialDataSuccessMonitor 

    success_rate_report_chunk_size: int = MISSING
    
    episode_length_list: list[int] = MISSING



