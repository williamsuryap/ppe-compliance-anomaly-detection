"""
Anomaly scoring engine for PPE compliance monitoring.
Tracks compliance over time and flags anomalous drops.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from collections import deque
import time


@dataclass
class AnomalyAlert:
    """An anomaly alert event."""
    timestamp: float
    severity: str           # "INFO", "WARNING", "CRITICAL"
    message: str
    compliance_score: float
    delta: float            # Change from previous window
    details: Dict = field(default_factory=dict)

    @property
    def severity_emoji(self) -> str:
        return {"INFO": "ℹ️", "WARNING": "⚠️", "CRITICAL": "🚨"}.get(self.severity, "❓")

    @property
    def time_str(self) -> str:
        return time.strftime("%H:%M:%S", time.localtime(self.timestamp))


class AnomalyScorer:
    """
    Temporal anomaly detection for PPE compliance.

    Monitors compliance scores over time using a rolling window
    and flags sudden drops in compliance as anomalies.
    """

    def __init__(self, window_size: int = 10,
                 warning_threshold: float = 15.0,
                 critical_threshold: float = 25.0):
        """
        Args:
            window_size: Number of recent scores to keep in rolling window
            warning_threshold: % drop to trigger WARNING
            critical_threshold: % drop to trigger CRITICAL
        """
        self.window_size = window_size
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.score_history: deque = deque(maxlen=window_size * 2)
        self.alerts: List[AnomalyAlert] = []
        self.inspection_count = 0

    def update(self, compliance_score: float,
               details: Optional[Dict] = None) -> Optional[AnomalyAlert]:
        """
        Add a new compliance score and check for anomalies.

        Args:
            compliance_score: Current overall compliance score (0-100)
            details: Optional dict with additional inspection details

        Returns:
            AnomalyAlert if an anomaly was detected, None otherwise
        """
        self.inspection_count += 1
        now = time.time()
        self.score_history.append(compliance_score)

        if len(self.score_history) < 3:
            return None

        # Calculate rolling average of previous scores
        prev_scores = list(self.score_history)[:-1]
        recent_avg = np.mean(prev_scores[-self.window_size:])
        delta = compliance_score - recent_avg

        alert = None
        if delta <= -self.critical_threshold:
            alert = AnomalyAlert(
                timestamp=now,
                severity="CRITICAL",
                message=f"Severe compliance drop detected: {delta:+.1f}% from rolling avg",
                compliance_score=compliance_score,
                delta=delta,
                details=details or {},
            )
        elif delta <= -self.warning_threshold:
            alert = AnomalyAlert(
                timestamp=now,
                severity="WARNING",
                message=f"Compliance degradation: {delta:+.1f}% from rolling avg",
                compliance_score=compliance_score,
                delta=delta,
                details=details or {},
            )
        elif compliance_score < 50:
            alert = AnomalyAlert(
                timestamp=now,
                severity="WARNING",
                message=f"Low overall compliance: {compliance_score:.1f}%",
                compliance_score=compliance_score,
                delta=delta,
                details=details or {},
            )

        if alert:
            self.alerts.append(alert)

        return alert

    def get_trend(self) -> Dict:
        """Get compliance trend data."""
        scores = list(self.score_history)
        if not scores:
            return {"current": 0, "average": 0, "trend": "stable", "scores": []}

        current = scores[-1]
        avg = np.mean(scores)

        if len(scores) < 3:
            trend = "stable"
        else:
            recent = np.mean(scores[-3:])
            older = np.mean(scores[:-3]) if len(scores) > 3 else recent
            if recent > older + 5:
                trend = "improving"
            elif recent < older - 5:
                trend = "declining"
            else:
                trend = "stable"

        return {
            "current": round(current, 1),
            "average": round(avg, 1),
            "min": round(min(scores), 1),
            "max": round(max(scores), 1),
            "trend": trend,
            "inspections": self.inspection_count,
            "alerts_count": len(self.alerts),
            "scores": scores,
        }

    def get_alerts_table(self) -> List[Dict]:
        """Get alerts formatted for display."""
        rows = []
        for alert in reversed(self.alerts[-20:]):
            rows.append({
                "Time": alert.time_str,
                "Severity": f"{alert.severity_emoji} {alert.severity}",
                "Score": f"{alert.compliance_score:.1f}%",
                "Change": f"{alert.delta:+.1f}%",
                "Message": alert.message,
            })
        return rows

    def simulate_shift(self, num_inspections: int = 24,
                       base_compliance: float = 85.0) -> List[Dict]:
        """
        Simulate a full shift of inspections for demo purposes.

        Args:
            num_inspections: Number of inspection points
            base_compliance: Starting compliance level

        Returns:
            List of inspection results with timestamps
        """
        results = []
        rng = np.random.RandomState(42)
        self.reset()

        for i in range(num_inspections):
            # Simulate compliance patterns
            hour = (8 + i * (8 / num_inspections))  # 8 AM to 4 PM
            
            # Lunch dip pattern
            if 11.5 <= hour <= 13:
                drop = rng.uniform(15, 30)
            # End-of-shift fatigue
            elif hour >= 15:
                drop = rng.uniform(5, 15)
            else:
                drop = rng.uniform(-5, 8)

            score = max(0, min(100, base_compliance - drop + rng.normal(0, 3)))
            
            alert = self.update(score, {"hour": round(hour, 1)})

            results.append({
                "inspection": i + 1,
                "time": f"{int(hour):02d}:{int((hour % 1) * 60):02d}",
                "score": round(score, 1),
                "alert": alert.severity if alert else None,
                "alert_msg": alert.message if alert else "—",
            })

        return results

    def reset(self):
        """Reset scorer state."""
        self.score_history.clear()
        self.alerts.clear()
        self.inspection_count = 0
