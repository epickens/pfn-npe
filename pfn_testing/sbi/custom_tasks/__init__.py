"""Custom SBI task implementations."""

from pfn_testing.sbi.custom_tasks.ar1_ts_t50 import AR1TimeSeriesT50Task
from pfn_testing.sbi.custom_tasks.lotka_volterra_raw import LotkaVolterraRawTask
from pfn_testing.sbi.custom_tasks.ou import OUTask
from pfn_testing.sbi.custom_tasks.sir_raw import SIRRawTask
from pfn_testing.sbi.custom_tasks.solar_dynamo import SolarDynamoTask

__all__ = [
    "AR1TimeSeriesT50Task",
    "LotkaVolterraRawTask",
    "OUTask",
    "SIRRawTask",
    "SolarDynamoTask",
]
