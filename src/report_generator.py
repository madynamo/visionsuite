"""
report_generator.py
--------------------
Collects the results produced by the individual detection/classification
modules into a single structured run report (JSON + CSV), timestamped, so
that every CLI invocation leaves an auditable record.

This is the piece that ties the three functional modules together into one
coherent workflow, and supports the Usability and Reliability non-functional
requirements (a user, or an automated evaluator, always gets a machine
readable summary of what happened, even if some modules were skipped).
"""

import csv
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List

from src.logger_setup import get_logger

logger = get_logger(__name__)


@dataclass
class RunReport:
    command: str
    input_path: str
    output_dir: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    results: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def add_result(self, key: str, value: Any) -> None:
        self.results[key] = value

    def add_error(self, message: str) -> None:
        logger.error(message)
        self.errors.append(message)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "command": self.command,
            "input_path": self.input_path,
            "output_dir": self.output_dir,
            "timestamp": self.timestamp,
            "results": self.results,
            "errors": self.errors,
            "status": "SUCCESS" if not self.errors else "COMPLETED_WITH_ERRORS",
        }

    def save_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        logger.info("Run report (JSON) saved to '%s'.", path)

    def append_csv(self, path: str) -> None:
        """Append a single summary row to a running CSV log of all runs."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        file_exists = os.path.isfile(path)

        row = {
            "timestamp": self.timestamp,
            "command": self.command,
            "input_path": self.input_path,
            "status": "SUCCESS" if not self.errors else "COMPLETED_WITH_ERRORS",
            "results_summary": json.dumps(self.results),
        }

        with open(path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
        logger.info("Run summary appended to CSV log '%s'.", path)
