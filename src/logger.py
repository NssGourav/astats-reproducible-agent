import json
import numpy as np
from datetime import datetime

class NumpyEncoder(json.JSONEncoder):
    """Custom encoder to handle NumPy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

class StatsLogger:
    def __init__(self):
        self.logs = []
        self.start_time = datetime.now()

    def log_step(self, step_name, description, metadata=None):
        """Log a high-level workflow step."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "step": step_name,
            "description": description,
            "metadata": metadata or {}
        }
        self.logs.append(log_entry)
        print(f"[{step_name}] {description}")

    def log_decision(self, decision_type, chosen_action, reason, alternatives=None, assumptions_checked=None, metadata=None):
        """Specifically log decisions with justification - critical for reproducibility."""
        decision_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "DECISION",
            "decision_type": decision_type,
            "chosen_action": chosen_action,
            "reason": reason,
            "alternatives": alternatives or [],
            "assumptions_checked": assumptions_checked or {},
            "metadata": metadata or {}
        }
        self.logs.append(decision_entry)
        print(f"--- DECISION: {decision_type} ---")
        print(f"Chosen: {chosen_action}")
        print(f"Reason: {reason}")
        if assumptions_checked:
            print(f"Assumptions: {assumptions_checked}")
        print("-" * 30)

    def save_log(self, filename="workflow_log.json"):
        output = {
            "session_start": self.start_time.isoformat(),
            "session_end": datetime.now().isoformat(),
            "workflow": self.logs
        }
        with open(filename, 'w') as f:
            json.dump(output, f, indent=4, cls=NumpyEncoder)
        print(f"Workflow log saved to {filename}")

logger = StatsLogger()
