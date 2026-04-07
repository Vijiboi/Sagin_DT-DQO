from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

from env.models import SlotResult


def write_run_outputs(output_root: str, slot_results: list[SlotResult], summary: dict[str, object]) -> Path:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    _write_slot_metrics(run_dir / "slot_metrics.csv", slot_results)
    _write_assignments(run_dir / "assignments.csv", slot_results)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return run_dir


def _write_slot_metrics(path: Path, slot_results: list[SlotResult]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "slot",
                "average_delay",
                "average_energy",
                "mission_cost",
                "fidelity_cost",
                "sync_trigger_count",
                "coordination_trigger_count",
                "one_hot_valid",
                "resource_feasible",
                "local_execution_count",
            ]
        )
        for slot_result in slot_results:
            writer.writerow(
                [
                    slot_result.slot,
                    round(slot_result.average_delay, 6),
                    round(slot_result.average_energy, 6),
                    round(slot_result.mission_cost, 6),
                    round(slot_result.fidelity_cost, 6),
                    slot_result.sync_trigger_count,
                    slot_result.coordination_trigger_count,
                    int(slot_result.one_hot_valid),
                    int(slot_result.resource_feasible),
                    slot_result.local_execution_count,
                ]
            )


def _write_assignments(path: Path, slot_results: list[SlotResult]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "slot",
                "task_id",
                "owner_ap_id",
                "destination_id",
                "local_cost",
                "delay_cost",
                "energy_cost",
                "mission_cost",
                "fidelity_cost",
            ]
        )
        for slot_result in slot_results:
            for assignment in slot_result.assignments:
                writer.writerow(
                    [
                        slot_result.slot,
                        assignment.task_id,
                        assignment.owner_ap_id,
                        assignment.destination_id,
                        round(assignment.local_cost, 6),
                        round(assignment.delay_cost, 6),
                        round(assignment.energy_cost, 6),
                        round(assignment.mission_cost, 6),
                        round(assignment.fidelity_cost, 6),
                    ]
                )
