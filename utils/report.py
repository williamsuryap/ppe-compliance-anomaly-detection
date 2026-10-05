"""
Inspection report generator for PPE compliance.
Generates structured reports from detection results.
"""

import time
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class InspectionReport:
    """A complete inspection report."""
    timestamp: float
    total_workers: int
    compliant_count: int
    partial_count: int
    non_compliant_count: int
    overall_score: float
    compliance_rate: float
    violations: List[Dict]
    per_item_stats: Dict

    @property
    def time_str(self) -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.timestamp))

    def to_dict(self) -> dict:
        return {
            "timestamp": self.time_str,
            "total_workers": self.total_workers,
            "compliant": self.compliant_count,
            "partial": self.partial_count,
            "non_compliant": self.non_compliant_count,
            "overall_score": self.overall_score,
            "compliance_rate": self.compliance_rate,
            "violations": self.violations,
            "per_item_stats": self.per_item_stats,
        }


class ReportGenerator:
    """
    Generates structured inspection reports from worker detections.
    """

    def __init__(self):
        self.report_history: List[InspectionReport] = []

    def generate(self, workers: list) -> InspectionReport:
        """
        Generate an inspection report from worker detections.

        Args:
            workers: List of WorkerDetection objects

        Returns:
            InspectionReport with full compliance breakdown
        """
        now = time.time()

        if not workers:
            report = InspectionReport(
                timestamp=now, total_workers=0,
                compliant_count=0, partial_count=0, non_compliant_count=0,
                overall_score=0, compliance_rate=0,
                violations=[], per_item_stats={}
            )
            self.report_history.append(report)
            return report

        compliant = 0
        partial = 0
        non_compliant = 0
        violations = []
        item_present = {"hardhat": 0, "vest": 0, "goggles": 0}
        total = len(workers)

        for w in workers:
            level = w.ppe_status.level
            if level == "compliant":
                compliant += 1
            elif level == "partial":
                partial += 1
            else:
                non_compliant += 1

            # Track per-item stats
            if w.ppe_status.hardhat:
                item_present["hardhat"] += 1
            if w.ppe_status.vest:
                item_present["vest"] += 1
            if w.ppe_status.goggles:
                item_present["goggles"] += 1

            # Record violations
            missing = w.ppe_status.items_missing
            if missing:
                violations.append({
                    "worker_id": w.worker_id,
                    "score": w.ppe_status.score,
                    "missing_items": ", ".join(missing),
                    "severity": "CRITICAL" if len(missing) >= 2 else "WARNING",
                })

        per_item_stats = {
            item: {
                "present": count,
                "missing": total - count,
                "rate": round(count / total * 100, 1) if total > 0 else 0,
            }
            for item, count in item_present.items()
        }

        overall_score = sum(w.ppe_status.score for w in workers) / total
        compliance_rate = round(compliant / total * 100, 1) if total > 0 else 0

        report = InspectionReport(
            timestamp=now,
            total_workers=total,
            compliant_count=compliant,
            partial_count=partial,
            non_compliant_count=non_compliant,
            overall_score=round(overall_score, 1),
            compliance_rate=compliance_rate,
            violations=violations,
            per_item_stats=per_item_stats,
        )
        self.report_history.append(report)
        return report

    def get_workers_table(self, workers: list) -> List[Dict]:
        """Format worker data as a table for display."""
        rows = []
        for w in workers:
            ppe = w.ppe_status
            rows.append({
                "Worker": f"W{w.worker_id}",
                "Hardhat": "✅" if ppe.hardhat else "❌",
                "Vest": "✅" if ppe.vest else "❌",
                "Goggles": "✅" if ppe.goggles else "❌",
                "Score": f"{ppe.score:.0f}%",
                "Status": {"compliant": "✅ OK", "partial": "⚠️ Partial",
                            "non_compliant": "🚨 Violation"}[ppe.level],
            })
        return rows

    def get_violations_table(self, workers: list) -> List[Dict]:
        """Get only violation entries for display."""
        rows = []
        for w in workers:
            missing = w.ppe_status.items_missing
            if missing:
                rows.append({
                    "Worker": f"W{w.worker_id}",
                    "Missing Items": ", ".join(m.capitalize() for m in missing),
                    "Score": f"{w.ppe_status.score:.0f}%",
                    "Severity": "🚨 CRITICAL" if len(missing) >= 2 else "⚠️ WARNING",
                })
        return rows

    def get_summary_html(self, report: InspectionReport) -> str:
        """Generate an HTML summary card."""
        score_color = "#22c55e" if report.overall_score >= 80 else \
                      "#f59e0b" if report.overall_score >= 50 else "#ef4444"

        return f"""
        <div style="background: #111118; border-radius: 16px; padding: 20px;
                    border: 1px solid #222;">
            <div style="display: flex; justify-content: space-between; align-items: center;
                        margin-bottom: 16px;">
                <h3 style="margin: 0; color: white; font-size: 18px;">
                    📋 Inspection Report
                </h3>
                <span style="color: #888; font-size: 12px;">{report.time_str}</span>
            </div>

            <div style="display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 16px;">
                <div style="background: {score_color}22; border: 1px solid {score_color}44;
                            border-radius: 12px; padding: 16px; flex: 1; min-width: 120px;
                            text-align: center;">
                    <div style="font-size: 28px; font-weight: 800; color: {score_color};">
                        {report.overall_score:.0f}%
                    </div>
                    <div style="font-size: 11px; color: #888; text-transform: uppercase;
                                letter-spacing: 1px;">Overall Score</div>
                </div>

                <div style="background: #22c55e22; border: 1px solid #22c55e44;
                            border-radius: 12px; padding: 16px; flex: 1; min-width: 100px;
                            text-align: center;">
                    <div style="font-size: 24px; font-weight: 700; color: #22c55e;">
                        {report.compliant_count}
                    </div>
                    <div style="font-size: 11px; color: #888;">Compliant</div>
                </div>

                <div style="background: #f59e0b22; border: 1px solid #f59e0b44;
                            border-radius: 12px; padding: 16px; flex: 1; min-width: 100px;
                            text-align: center;">
                    <div style="font-size: 24px; font-weight: 700; color: #f59e0b;">
                        {report.partial_count}
                    </div>
                    <div style="font-size: 11px; color: #888;">Partial</div>
                </div>

                <div style="background: #ef444422; border: 1px solid #ef444444;
                            border-radius: 12px; padding: 16px; flex: 1; min-width: 100px;
                            text-align: center;">
                    <div style="font-size: 24px; font-weight: 700; color: #ef4444;">
                        {report.non_compliant_count}
                    </div>
                    <div style="font-size: 11px; color: #888;">Violations</div>
                </div>
            </div>

            <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                <span style="background: #1a1a24; border-radius: 6px; padding: 4px 10px;
                             font-size: 12px; color: #aaa;">
                    👷 {report.total_workers} Workers
                </span>
                <span style="background: #1a1a24; border-radius: 6px; padding: 4px 10px;
                             font-size: 12px; color: #aaa;">
                    📊 {report.compliance_rate:.0f}% Compliance Rate
                </span>
                <span style="background: #1a1a24; border-radius: 6px; padding: 4px 10px;
                             font-size: 12px; color: #aaa;">
                    ⚠️ {len(report.violations)} Violations
                </span>
            </div>
        </div>"""
