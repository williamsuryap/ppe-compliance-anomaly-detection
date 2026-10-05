"""
Synthetic industrial scene generator for PPE demo.
Creates workplace scenes with worker silhouettes and PPE indicators.
"""

import cv2
import numpy as np
from typing import Tuple


def generate_industrial_scene(width: int = 1280, height: int = 720,
                               seed: int = None) -> np.ndarray:
    """
    Generate a synthetic industrial/construction scene.

    Creates a scene with concrete floor, structural elements,
    and ambient industrial details.

    Args:
        width: Frame width
        height: Frame height
        seed: Random seed for reproducibility

    Returns:
        BGR image as numpy array
    """
    if seed is not None:
        np.random.seed(seed)

    frame = np.zeros((height, width, 3), dtype=np.uint8)

    # Background: industrial gray-brown gradient
    for y in range(height):
        ratio = y / height
        r = int(45 + ratio * 25)
        g = int(50 + ratio * 20)
        b = int(55 + ratio * 15)
        frame[y, :] = (b, g, r)

    # Concrete floor
    floor_y = int(height * 0.55)
    cv2.rectangle(frame, (0, floor_y), (width, height), (75, 78, 80), -1)

    # Floor texture
    noise = np.random.randint(-5, 5, (height - floor_y, width, 3), dtype=np.int16)
    floor_region = frame[floor_y:height].astype(np.int16)
    frame[floor_y:height] = np.clip(floor_region + noise, 0, 255).astype(np.uint8)

    # Steel beams / structural elements
    beam_color = (60, 62, 65)
    for i in range(3):
        bx = int(width * (0.15 + i * 0.35))
        cv2.rectangle(frame, (bx, 0), (bx + 20, floor_y), beam_color, -1)
        cv2.rectangle(frame, (bx - 30, floor_y - 8), (bx + 50, floor_y), beam_color, -1)

    # Horizontal beam
    cv2.rectangle(frame, (0, int(height * 0.08)), (width, int(height * 0.08) + 15),
                  beam_color, -1)

    # Safety signage area (yellow/black stripes)
    stripe_y = int(height * 0.50)
    stripe_h = 20
    stripe_w = 30
    for x in range(0, width, stripe_w * 2):
        cv2.rectangle(frame, (x, stripe_y), (x + stripe_w, stripe_y + stripe_h),
                      (0, 200, 255), -1)  # Yellow
        cv2.rectangle(frame, (x + stripe_w, stripe_y),
                      (x + stripe_w * 2, stripe_y + stripe_h),
                      (30, 30, 30), -1)  # Black

    # Equipment / machinery blocks
    for _ in range(np.random.randint(2, 5)):
        ex = np.random.randint(50, width - 150)
        ey = np.random.randint(floor_y + 20, height - 80)
        ew = np.random.randint(40, 120)
        eh = np.random.randint(30, 70)
        color = (
            np.random.randint(50, 90),
            np.random.randint(55, 95),
            np.random.randint(45, 85),
        )
        cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), color, -1)
        cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (40, 40, 40), 1)

    # Draw synthetic worker silhouettes
    num_workers = np.random.randint(4, 8)
    worker_colors = [
        (180, 140, 100), (160, 130, 90), (170, 135, 95),
        (150, 120, 85), (165, 125, 88),
    ]

    for i in range(num_workers):
        wx = np.random.randint(80, width - 80)
        wy = np.random.randint(floor_y - 30, height - 140)
        body_h = np.random.randint(80, 130)
        body_w = int(body_h * 0.35)
        head_r = int(body_w * 0.6)

        skin = worker_colors[i % len(worker_colors)]

        # Body (work clothes - dark blue/gray)
        clothes = (np.random.randint(60, 100), np.random.randint(50, 80),
                   np.random.randint(40, 60))
        cv2.rectangle(frame, (wx - body_w // 2, wy),
                      (wx + body_w // 2, wy + body_h), clothes, -1)

        # Head
        cv2.circle(frame, (wx, wy - head_r), head_r, skin, -1)

        # Random PPE indicators
        has_hardhat = np.random.random() > 0.35
        has_vest = np.random.random() > 0.30

        if has_hardhat:
            # Yellow hardhat
            hat_pts = np.array([
                [wx - head_r - 3, wy - head_r],
                [wx + head_r + 3, wy - head_r],
                [wx + head_r + 5, wy - head_r - 8],
                [wx - head_r - 5, wy - head_r - 8],
            ])
            cv2.fillPoly(frame, [hat_pts], (0, 200, 255))

        if has_vest:
            # Orange hi-vis vest
            vest_y1 = wy + 5
            vest_y2 = wy + int(body_h * 0.6)
            cv2.rectangle(frame, (wx - body_w // 2 - 2, vest_y1),
                          (wx + body_w // 2 + 2, vest_y2), (0, 140, 255), -1)
            # Reflective stripes
            cv2.line(frame, (wx - body_w // 2, vest_y1 + 10),
                     (wx + body_w // 2, vest_y1 + 10), (200, 200, 200), 2)
            cv2.line(frame, (wx - body_w // 2, vest_y2 - 10),
                     (wx + body_w // 2, vest_y2 - 10), (200, 200, 200), 2)

        # Legs
        leg_w = body_w // 3
        cv2.rectangle(frame, (wx - body_w // 3, wy + body_h),
                      (wx - body_w // 3 + leg_w, wy + body_h + body_h // 3),
                      clothes, -1)
        cv2.rectangle(frame, (wx + body_w // 6, wy + body_h),
                      (wx + body_w // 6 + leg_w, wy + body_h + body_h // 3),
                      clothes, -1)

    # Ambient lighting effect
    overlay = frame.copy()
    cv2.circle(overlay, (width // 3, 0), int(width * 0.4), (90, 95, 100), -1)
    cv2.addWeighted(overlay, 0.05, frame, 0.95, 0, frame)

    return frame
