"""Video loading / saving. Everything downstream assumes (T, H, W, 3) uint8 RGB."""
from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np


def read_video(path: str, max_frames: Optional[int] = None,
               resize: Optional[tuple[int, int]] = None) -> np.ndarray:
    """Read a video file to (T, H, W, 3) uint8 RGB.

    resize: (width, height) if given.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"cannot open video: {path}")
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if resize is not None:
            bgr = cv2.resize(bgr, resize, interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if max_frames is not None and len(frames) >= max_frames:
            break
    cap.release()
    if not frames:
        raise ValueError(f"no frames decoded from {path}")
    return np.stack(frames, axis=0)


def write_video(path: str, video: np.ndarray, fps: float = 24.0) -> None:
    """Write (T, H, W, 3) uint8 RGB to disk as mp4v."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    t, h, w, _ = video.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for f in video:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()


def to_gray(video: np.ndarray) -> np.ndarray:
    """(T,H,W,3) uint8 -> (T,H,W) float32 in [0,1]."""
    return np.stack(
        [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in video], axis=0
    ).astype(np.float32) / 255.0


def normalise(video: np.ndarray, size: tuple[int, int], n_frames: int) -> np.ndarray:
    """Fix resolution and frame count so every measurement is made identically.

    size: (width, height). Frames are centre-cropped in time (not resampled)
    to avoid introducing temporal interpolation artefacts of our own.
    """
    w, h = size
    if video.shape[0] < n_frames:
        raise ValueError(f"clip has {video.shape[0]} frames, need {n_frames}")
    start = (video.shape[0] - n_frames) // 2
    video = video[start:start + n_frames]
    if (video.shape[2], video.shape[1]) != (w, h):
        video = np.stack(
            [cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA) for f in video],
            axis=0,
        )
    return video
