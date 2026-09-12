"""Policy-independent multi-VNR workload traces for matched experiments."""

from dataclasses import dataclass
from typing import Any, List

from .scenarios import DynamicSAGINScenario
from .topology import RunningService


@dataclass(frozen=True)
class WorkloadEvent:
    time_step: int
    service: RunningService


def generate_workload_trace(
    scenario: DynamicSAGINScenario,
    config: Any,
    start_time: int,
    steps: int,
) -> List[WorkloadEvent]:
    """Generate every arrival before policy execution, including later rejections."""
    events: List[WorkloadEvent] = []
    arrival_index = 1
    span = config.max_virtual_nodes - config.min_virtual_nodes + 1
    for time_step in range(start_time + 1, start_time + steps + 1):
        draw = ((time_step * 37 + arrival_index * 17) % 100) / 100.0
        if draw >= config.arrival_probability:
            continue
        service_id = f"svc-{arrival_index:04d}"
        virtual_nodes = config.min_virtual_nodes + ((arrival_index * 7 + time_step) % span)
        events.append(WorkloadEvent(time_step, scenario.build_service(service_id, virtual_nodes, time_step)))
        arrival_index += 1
    return events
