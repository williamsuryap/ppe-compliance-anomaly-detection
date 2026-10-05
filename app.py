"""
PPE Compliance & Anomaly Detection — Gradio App
=================================================
Automated industrial safety inspection system for detecting
Personal Protective Equipment compliance and anomalies.

Run:  python app.py
Visit: http://localhost:7860
"""

import gradio as gr
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import io
import time

from utils.ppe_detector import PPEDetector, draw_compliance_overlay, compute_overall_compliance
from utils.anomaly_scorer import AnomalyScorer
from utils.report import ReportGenerator
from assets.sample_scene import generate_industrial_scene

# ── Global instances ──────────────────────────────────────────────────────────
reporter = ReportGenerator()
anomaly_scorer = AnomalyScorer()


# ── Helper functions ──────────────────────────────────────────────────────────

def make_compliance_gauge(score: float) -> np.ndarray:
    """Create a circular gauge chart for compliance score."""
    fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=100, subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor("#0f1117")

    # Gauge parameters
    theta_range = np.pi * 1.5  # 270 degrees
    start_angle = np.pi * 0.75
    score_frac = score / 100.0

    # Background arc
    theta_bg = np.linspace(start_angle, start_angle + theta_range, 100)
    ax.plot(theta_bg, [1] * 100, color="#222", linewidth=18, solid_capstyle="round")

    # Score arc
    if score_frac > 0:
        theta_score = np.linspace(start_angle, start_angle + theta_range * score_frac,
                                   int(100 * score_frac) + 2)
        color = "#22c55e" if score >= 80 else "#f59e0b" if score >= 50 else "#ef4444"
        ax.plot(theta_score, [1] * len(theta_score), color=color,
                linewidth=18, solid_capstyle="round")

    ax.set_ylim(0, 1.5)
    ax.set_rticks([])
    ax.set_thetagrids([])
    ax.spines["polar"].set_visible(False)
    ax.grid(False)

    # Center text
    ax.text(0, 0, f"{score:.0f}%", ha="center", va="center",
            fontsize=28, fontweight="bold", color="white",
            transform=ax.transAxes)
    ax.text(0, -0.12, "Compliance", ha="center", va="center",
            fontsize=10, color="#888", transform=ax.transAxes)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return np.array(Image.open(buf))


def make_timeline_chart(results: list) -> np.ndarray:
    """Create a compliance timeline chart."""
    fig, ax = plt.subplots(figsize=(10, 4), dpi=100)
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    times = [r["time"] for r in results]
    scores = [r["score"] for r in results]

    # Color zones
    ax.axhspan(80, 100, alpha=0.1, color="#22c55e")
    ax.axhspan(50, 80, alpha=0.1, color="#f59e0b")
    ax.axhspan(0, 50, alpha=0.1, color="#ef4444")

    # Threshold lines
    ax.axhline(y=80, color="#22c55e", linestyle="--", alpha=0.4, linewidth=1)
    ax.axhline(y=50, color="#f59e0b", linestyle="--", alpha=0.4, linewidth=1)

    # Main line
    ax.plot(times, scores, color="#06b6d4", linewidth=2.5, marker="o",
            markersize=5, markerfacecolor="#06b6d4", markeredgecolor="white",
            markeredgewidth=1, zorder=5)

    # Fill under curve
    ax.fill_between(times, scores, alpha=0.15, color="#06b6d4")

    # Highlight alerts
    for r in results:
        if r["alert"] == "CRITICAL":
            idx = results.index(r)
            ax.plot(r["time"], r["score"], "v", color="#ef4444",
                    markersize=12, zorder=10)
        elif r["alert"] == "WARNING":
            ax.plot(r["time"], r["score"], "v", color="#f59e0b",
                    markersize=10, zorder=10)

    ax.set_xlabel("Time", fontsize=10, color="#aaa", labelpad=8)
    ax.set_ylabel("Compliance Score (%)", fontsize=10, color="#aaa", labelpad=8)
    ax.set_title("Shift Compliance Timeline", fontsize=14, fontweight="bold",
                 color="white", pad=12)
    ax.set_ylim(0, 105)
    ax.tick_params(colors="#888", labelsize=8, rotation=45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333")
    ax.spines["bottom"].set_color("#333")

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return np.array(Image.open(buf))


def make_item_chart(per_item: dict) -> np.ndarray:
    """Create a per-item compliance bar chart."""
    fig, ax = plt.subplots(figsize=(5, 3), dpi=100)
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    items = list(per_item.keys())
    rates = [per_item[item]["rate"] for item in items]
    colors = ["#f59e0b", "#22c55e", "#06b6d4"]

    bars = ax.barh(items, rates, color=colors[:len(items)], height=0.5,
                   edgecolor="white", linewidth=0.5)

    for bar, rate in zip(bars, rates):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{rate:.0f}%", ha="left", va="center", fontsize=11,
                fontweight="bold", color="white")

    ax.set_xlim(0, 110)
    ax.set_xlabel("Compliance Rate (%)", fontsize=9, color="#aaa")
    ax.set_title("Per-Item Compliance", fontsize=12, fontweight="bold",
                 color="white", pad=10)
    ax.tick_params(colors="#888", labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333")
    ax.spines["bottom"].set_color("#333")

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return np.array(Image.open(buf))


# ── Tab 1: Single Image Inspection ───────────────────────────────────────────

def inspect_single(image, sensitivity, use_mock):
    """Run PPE inspection on a single image."""
    if image is None:
        image = generate_industrial_scene(seed=42)
    else:
        if isinstance(image, Image.Image):
            image = np.array(image)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    det = PPEDetector(use_mock=use_mock)
    workers = det.detect(image, sensitivity=sensitivity)
    annotated = draw_compliance_overlay(image, workers)
    stats = compute_overall_compliance(workers)

    report = reporter.generate(workers)
    summary_html = reporter.get_summary_html(report)

    # Workers table
    workers_data = reporter.get_workers_table(workers)
    workers_df = pd.DataFrame(workers_data) if workers_data else pd.DataFrame()

    # Gauge + item chart
    gauge = make_compliance_gauge(stats["overall_score"])
    item_chart = make_item_chart(report.per_item_stats) if report.per_item_stats else None

    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    return annotated_rgb, summary_html, workers_df, gauge, item_chart


# ── Tab 2: Batch Inspection ──────────────────────────────────────────────────

def inspect_batch(images, sensitivity, use_mock):
    """Run PPE inspection on multiple images."""
    if images is None or len(images) == 0:
        # Generate 5 sample scenes
        images_bgr = [generate_industrial_scene(seed=i * 17 + 7) for i in range(5)]
    else:
        images_bgr = []
        for img in images:
            if isinstance(img, Image.Image):
                img = np.array(img)
            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            elif img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            images_bgr.append(img)

    det = PPEDetector(use_mock=use_mock)
    all_results = []

    for idx, img in enumerate(images_bgr):
        workers = det.detect(img, sensitivity=sensitivity)
        stats = compute_overall_compliance(workers)
        all_results.append({
            "Image": f"Image {idx + 1}",
            "Workers": stats["total_workers"],
            "Compliant": stats["compliant"],
            "Partial": stats["partial"],
            "Violations": stats["non_compliant"],
            "Score": f"{stats['overall_score']:.0f}%",
            "Rate": f"{stats['compliance_rate']:.0f}%",
        })

    df = pd.DataFrame(all_results)

    # Overall summary
    all_scores = [float(r["Score"].replace("%", "")) for r in all_results]
    avg_score = np.mean(all_scores) if all_scores else 0
    total_workers = sum(r["Workers"] for r in all_results)
    total_violations = sum(r["Violations"] for r in all_results)

    summary_html = f"""
    <div style="background: #111118; border-radius: 16px; padding: 20px;
                border: 1px solid #222;">
        <h3 style="color: white; margin: 0 0 12px 0;">📦 Batch Inspection Results</h3>
        <div style="display: flex; gap: 12px; flex-wrap: wrap;">
            <div style="background: #06b6d422; border-radius: 10px; padding: 14px;
                        flex: 1; text-align: center; border: 1px solid #06b6d444;">
                <div style="font-size: 24px; font-weight: 700; color: #06b6d4;">
                    {len(images_bgr)}</div>
                <div style="font-size: 11px; color: #888;">Images</div>
            </div>
            <div style="background: #8b5cf622; border-radius: 10px; padding: 14px;
                        flex: 1; text-align: center; border: 1px solid #8b5cf644;">
                <div style="font-size: 24px; font-weight: 700; color: #8b5cf6;">
                    {total_workers}</div>
                <div style="font-size: 11px; color: #888;">Workers</div>
            </div>
            <div style="background: {'#22c55e' if avg_score >= 80 else '#f59e0b' if avg_score >= 50 else '#ef4444'}22;
                        border-radius: 10px; padding: 14px; flex: 1; text-align: center;
                        border: 1px solid {'#22c55e' if avg_score >= 80 else '#f59e0b' if avg_score >= 50 else '#ef4444'}44;">
                <div style="font-size: 24px; font-weight: 700;
                            color: {'#22c55e' if avg_score >= 80 else '#f59e0b' if avg_score >= 50 else '#ef4444'};">
                    {avg_score:.0f}%</div>
                <div style="font-size: 11px; color: #888;">Avg Score</div>
            </div>
            <div style="background: #ef444422; border-radius: 10px; padding: 14px;
                        flex: 1; text-align: center; border: 1px solid #ef444444;">
                <div style="font-size: 24px; font-weight: 700; color: #ef4444;">
                    {total_violations}</div>
                <div style="font-size: 11px; color: #888;">Violations</div>
            </div>
        </div>
    </div>"""

    return df, summary_html


# ── Tab 3: Anomaly Timeline ──────────────────────────────────────────────────

def run_anomaly_timeline(num_inspections, base_compliance):
    """Simulate a shift and show anomaly timeline."""
    scorer = AnomalyScorer(warning_threshold=12, critical_threshold=22)
    results = scorer.simulate_shift(
        num_inspections=int(num_inspections),
        base_compliance=float(base_compliance),
    )

    timeline_chart = make_timeline_chart(results)

    # Alerts table
    alerts = scorer.get_alerts_table()
    alerts_df = pd.DataFrame(alerts) if alerts else pd.DataFrame(
        columns=["Time", "Severity", "Score", "Change", "Message"]
    )

    # Results table
    results_df = pd.DataFrame(results)

    trend = scorer.get_trend()
    trend_emoji = {"improving": "📈", "declining": "📉", "stable": "➡️"}
    trend_color = {"improving": "#22c55e", "declining": "#ef4444", "stable": "#f59e0b"}

    summary_html = f"""
    <div style="background: #111118; border-radius: 16px; padding: 20px;
                border: 1px solid #222;">
        <h3 style="color: white; margin: 0 0 12px 0;">
            {trend_emoji.get(trend['trend'], '📊')} Shift Analysis
        </h3>
        <div style="display: flex; gap: 10px; flex-wrap: wrap;">
            <div style="background: #1a1a24; border-radius: 8px; padding: 12px;
                        flex: 1; min-width: 100px;">
                <div style="font-size: 20px; font-weight: 700; color: #06b6d4;">
                    {trend['average']:.1f}%</div>
                <div style="font-size: 10px; color: #888;">Average</div>
            </div>
            <div style="background: #1a1a24; border-radius: 8px; padding: 12px;
                        flex: 1; min-width: 100px;">
                <div style="font-size: 20px; font-weight: 700;
                            color: {trend_color.get(trend['trend'], '#888')};">
                    {trend['trend'].capitalize()}</div>
                <div style="font-size: 10px; color: #888;">Trend</div>
            </div>
            <div style="background: #1a1a24; border-radius: 8px; padding: 12px;
                        flex: 1; min-width: 100px;">
                <div style="font-size: 20px; font-weight: 700; color: #ef4444;">
                    {trend['alerts_count']}</div>
                <div style="font-size: 10px; color: #888;">Alerts</div>
            </div>
            <div style="background: #1a1a24; border-radius: 8px; padding: 12px;
                        flex: 1; min-width: 100px;">
                <div style="font-size: 20px; font-weight: 700; color: #22c55e;">
                    {trend['max']:.0f}%</div>
                <div style="font-size: 10px; color: #888;">Peak</div>
            </div>
        </div>
    </div>"""

    return timeline_chart, alerts_df, summary_html, results_df


# ── Gradio UI ─────────────────────────────────────────────────────────────────

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.amber,
    secondary_hue=gr.themes.colors.red,
    neutral_hue=gr.themes.colors.gray,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
).set(
    body_background_fill="#0a0a0f",
    body_background_fill_dark="#0a0a0f",
    block_background_fill="#111118",
    block_background_fill_dark="#111118",
    block_border_width="0px",
    block_shadow="0 4px 24px rgba(0,0,0,0.3)",
    input_background_fill="#1a1a24",
    input_background_fill_dark="#1a1a24",
    button_primary_background_fill="linear-gradient(135deg, #f59e0b, #ef4444)",
    button_primary_background_fill_hover="linear-gradient(135deg, #fbbf24, #f87171)",
    button_primary_text_color="white",
)

CSS = """
.gradio-container { max-width: 1300px !important; }
footer { display: none !important; }
.gr-button-primary { border-radius: 10px !important; font-weight: 600 !important; }
"""

with gr.Blocks(theme=THEME, css=CSS, title="PPE Compliance & Anomaly Detection") as app:
    gr.HTML("""
    <div style="text-align: center; padding: 20px 0 10px 0;">
        <h1 style="background: linear-gradient(135deg, #f59e0b, #ef4444);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                   font-size: 2.2rem; margin: 0; font-weight: 800;">
            🦺 PPE Compliance & Anomaly Detection
        </h1>
        <p style="color: #888; margin-top: 6px; font-size: 0.95rem;">
            Automated industrial safety inspection — Hardhat · Vest · Goggles
        </p>
    </div>
    """)

    with gr.Tabs():
        # ── Tab 1: Single Image Inspection ────────────────────────────────
        with gr.Tab("🔍 Single Inspection", id="single_tab"):
            with gr.Row():
                with gr.Column(scale=1):
                    img_input = gr.Image(label="Upload Workplace Image", type="pil",
                                         height=320)
                    sensitivity = gr.Radio(["low", "medium", "high"], value="medium",
                                           label="Detection Sensitivity")
                    mock_toggle = gr.Checkbox(value=True, label="Mock Mode")
                    inspect_btn = gr.Button("🔍 Run Inspection", variant="primary", size="lg")

                with gr.Column(scale=1):
                    img_output = gr.Image(label="Compliance Overlay", height=320)
                    summary_html = gr.HTML(label="Report")

            with gr.Row():
                workers_table = gr.Dataframe(label="Worker Details", interactive=False)
                with gr.Column():
                    gauge_img = gr.Image(label="Compliance Gauge", height=220)
                    item_chart = gr.Image(label="Per-Item Breakdown")

            inspect_btn.click(
                inspect_single,
                inputs=[img_input, sensitivity, mock_toggle],
                outputs=[img_output, summary_html, workers_table, gauge_img, item_chart],
            )

        # ── Tab 2: Batch Inspection ───────────────────────────────────────
        with gr.Tab("📦 Batch Inspection", id="batch_tab"):
            with gr.Row():
                with gr.Column(scale=1):
                    batch_input = gr.Gallery(label="Upload Images (or leave empty for demo)",
                                             type="pil", columns=3, height=250)
                    batch_sens = gr.Radio(["low", "medium", "high"], value="medium",
                                          label="Sensitivity")
                    batch_mock = gr.Checkbox(value=True, label="Mock Mode")
                    batch_btn = gr.Button("📦 Run Batch Inspection", variant="primary",
                                          size="lg")

                with gr.Column(scale=1):
                    batch_summary = gr.HTML(label="Batch Summary")
                    batch_table = gr.Dataframe(label="Results per Image", interactive=False)

            batch_btn.click(
                inspect_batch,
                inputs=[batch_input, batch_sens, batch_mock],
                outputs=[batch_table, batch_summary],
            )

        # ── Tab 3: Anomaly Timeline ───────────────────────────────────────
        with gr.Tab("📈 Anomaly Timeline", id="anomaly_tab"):
            gr.HTML("""
            <div style="background: linear-gradient(135deg, #f59e0b22, #ef444422);
                        border-radius: 12px; padding: 16px; margin-bottom: 16px;
                        border: 1px solid #f59e0b44;">
                <p style="color: #ccc; margin: 0; font-size: 0.9rem;">
                    <strong style="color: #fbbf24;">📊 Shift Simulator</strong> —
                    Simulates a full 8-hour work shift with realistic compliance
                    patterns, detecting anomalies like lunch-break drops and
                    end-of-shift fatigue.
                </p>
            </div>
            """)

            with gr.Row():
                num_inspections = gr.Slider(8, 48, value=24, step=1,
                                             label="Inspection Points")
                base_compliance = gr.Slider(50, 100, value=85, step=1,
                                             label="Base Compliance Level (%)")
                timeline_btn = gr.Button("📈 Simulate Shift", variant="primary", size="lg")

            timeline_chart = gr.Image(label="Compliance Timeline")
            timeline_summary = gr.HTML(label="Shift Analysis")

            with gr.Row():
                alerts_table = gr.Dataframe(label="🚨 Anomaly Alerts", interactive=False)
                results_table = gr.Dataframe(label="📋 Inspection Log", interactive=False)

            timeline_btn.click(
                run_anomaly_timeline,
                inputs=[num_inspections, base_compliance],
                outputs=[timeline_chart, alerts_table, timeline_summary, results_table],
            )

    gr.HTML("""
    <div style="text-align: center; padding: 16px; color: #555; font-size: 0.8rem;">
        Built with Gradio • OpenCV • YOLOv8 | © 2024 William Surya
    </div>
    """)

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
