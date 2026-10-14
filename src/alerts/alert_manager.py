from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List


@dataclass
class AlertEvent:
    event_type: str
    timestamp: str
    details: Dict[str, object] = field(default_factory=dict)


class AlertManager:
    def __init__(self):
        self.events: List[AlertEvent] = []
        self.current_state = "NORMAL"
        self.last_alert_time = None

    def trigger_potential_fall(self, confidence=0.0):
        if self.current_state == "POTENTIAL_FALL":
            return "POTENTIAL_FALL"

        self.current_state = "POTENTIAL_FALL"
        self.last_alert_time = datetime.now().isoformat(timespec="seconds")
        self.events.append(
            AlertEvent(
                event_type="POTENTIAL_FALL",
                timestamp=self.last_alert_time,
                details={"confidence": float(confidence), "status": "waiting_for_verification"},
            )
        )
        return self.current_state

    def request_help(self):
        self.current_state = "HELP_REQUESTED"
        timestamp = datetime.now().isoformat(timespec="seconds")
        self.events.append(AlertEvent(event_type="HELP_REQUESTED", timestamp=timestamp, details={}))
        return self.current_state

    def cancel_alert(self):
        self.current_state = "NORMAL"
        timestamp = datetime.now().isoformat(timespec="seconds")
        self.events.append(AlertEvent(event_type="CANCELLED", timestamp=timestamp, details={}))
        return self.current_state

    def reset(self):
        self.current_state = "NORMAL"
        self.events.clear()
        self.last_alert_time = None
        return self.current_state
