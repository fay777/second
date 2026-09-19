from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ReconfigurationRecord:
    time_step: int
    service_id: str
    risk_score: float
    migrated: bool
    scope: str
    estimated_cost: float
    realized_cost: float
    disruption_time: float
    post_risk: float
    risk_reduction: float
    node_migrations: int
    link_reroutes: int
    node_migration_cost: float
    link_reroute_cost: float
    accepted: bool
    sla_violated: bool


@dataclass
class MetricsTracker:
    records: List[ReconfigurationRecord] = field(default_factory=list)

    def add(self, record: ReconfigurationRecord) -> None:
        self.records.append(record)

    def summary(self) -> Dict[str, float]:
        if not self.records:
            return {
                "num_records": 0.0,
                "migrations": 0.0,
                "avg_risk": 0.0,
                "avg_estimated_cost": 0.0,
                "avg_realized_cost": 0.0,
                "total_realized_cost": 0.0,
                "avg_disruption": 0.0,
                "total_disruption": 0.0,
                "avg_post_risk": 0.0,
                "avg_risk_reduction": 0.0,
                "accepted_reconfigurations": 0.0,
                "node_migrations": 0.0,
                "link_reroutes": 0.0,
                "avg_node_migration_cost": 0.0,
                "avg_link_reroute_cost": 0.0,
                "total_node_migration_cost": 0.0,
                "total_link_reroute_cost": 0.0,
                "sla_violations": 0.0,
            }
        count = len(self.records)
        return {
            "num_records": float(count),
            "migrations": float(sum(1 for record in self.records if record.migrated)),
            "avg_risk": sum(record.risk_score for record in self.records) / count,
            "avg_estimated_cost": sum(record.estimated_cost for record in self.records) / count,
            "avg_realized_cost": sum(record.realized_cost for record in self.records) / count,
            "total_realized_cost": sum(record.realized_cost for record in self.records),
            "avg_disruption": sum(record.disruption_time for record in self.records) / count,
            "total_disruption": sum(record.disruption_time for record in self.records),
            "avg_post_risk": sum(record.post_risk for record in self.records) / count,
            "avg_risk_reduction": sum(record.risk_reduction for record in self.records) / count,
            "accepted_reconfigurations": float(sum(1 for record in self.records if record.accepted)),
            "node_migrations": float(sum(record.node_migrations for record in self.records)),
            "link_reroutes": float(sum(record.link_reroutes for record in self.records)),
            "avg_node_migration_cost": sum(record.node_migration_cost for record in self.records) / count,
            "avg_link_reroute_cost": sum(record.link_reroute_cost for record in self.records) / count,
            "total_node_migration_cost": sum(record.node_migration_cost for record in self.records),
            "total_link_reroute_cost": sum(record.link_reroute_cost for record in self.records),
            "sla_violations": float(sum(1 for record in self.records if record.sla_violated)),
        }
