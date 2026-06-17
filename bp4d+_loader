"""
data/bp4d_loader.py
===================
Dataloader for the BP4D+ dataset.
Handles thermal video frames and physiological ground truth data
(heart rate, respiration rate).
"""

import os
import glob
import cv2
import numpy as np


class BP4DDataset:

    def __init__(self, root_dir, subjects=None, tasks=None,
                 warmup_seconds=0, fps=25.0):
        """
        Initialize the BP4D+ dataset loader.

        Args:
            root_dir:        Path to the BP4D+ root folder.
            subjects:        List of subject IDs, e.g. ["F001", "F002"].
                             If None, all available subjects are loaded.
            tasks:           List of task IDs, e.g. ["T1", "T8"].
                             If None, all available tasks are loaded.
            warmup_seconds:  Number of seconds to skip at the beginning
                             of each video (camera warmup period).
            fps:             Fallback framerate if the video metadata
                             does not contain a valid FPS value.
        """
        self.root_dir = root_dir
        self.thermal_dir = os.path.join(root_dir, "Thermal")
        self.physio_dir = os.path.join(root_dir, "Physiology")
        self.warmup_seconds = warmup_seconds
        self.default_fps = fps

        # ── Discover subjects ──
        # If no subjects are specified, scan the Thermal directory
        # and use all subdirectories as subject IDs.
        if subjects is None:
            self.subjects = sorted([
                d for d in os.listdir(self.thermal_dir)
                if os.path.isdir(os.path.join(self.thermal_dir, d))
            ])
        else:
            self.subjects = subjects

        # ── Collect all valid (subject, task) pairs ──
        # A pair is valid only if both a thermal video and
        # a corresponding physiology directory exist.
        self.samples = []
        for subj in self.subjects:
            subj_thermal = os.path.join(self.thermal_dir, subj)
            wmv_files = sorted(glob.glob(
                os.path.join(subj_thermal, "*.wmv")
            ))
            for wmv_path in wmv_files:
                # Extract task name from filename (e.g. "T1.wmv" -> "T1")
                task = os.path.splitext(
                    os.path.basename(wmv_path)
                )[0]

                # Skip tasks not in the requested list
                if tasks is not None and task not in tasks:
                    continue

                # Only include if ground truth physiology exists
                physio_path = os.path.join(
                    self.physio_dir, subj, task
                )
                if os.path.isdir(physio_path):
                    self.samples.append((subj, task, wmv_path))

        print(f"BP4DDataset: {len(self.subjects)} subjects, "
              f"{len(self.samples)} samples")

    def __len__(self):
        """Return the total number of (subject, task) samples."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Load and return a single sample by index.

        Returns:
            dict with keys:
                - frames:       np.ndarray (N, H, W, 3), float32
                - fps:          float, video framerate
                - subject:      str, subject ID (e.g. "F001")
                - task:         str, task ID (e.g. "T1")
                - hr_bpm:       float, mean heart rate in BPM
                - rr_bpm:       float, median respiration rate in BPM
                - pulse_rate:   np.ndarray, full heart rate signal
                - resp_rate:    np.ndarray, full respiration rate signal
        """
        subj, task, wmv_path = self.samples[idx]

        # Load thermal video frames
        frames, fps = self._load_frames(wmv_path)

        # Load physiological ground truth
        physio_dir = os.path.join(self.physio_dir, subj, task)
        hr_bpm, rr_bpm, pulse_rate, resp_rate = self._load_physiology(physio_dir)

        return {
            "frames":       frames,
            "fps":          fps,
            "subject":      subj,
            "task":         task,
            "hr_bpm":       hr_bpm,
            "rr_bpm":       rr_bpm,
            "pulse_rate":   pulse_rate,
            "resp_rate":    resp_rate,
        }

    def _load_frames(self, video_path):
        """
        Read all frames from a thermal video file.

        Converts frames to float32 and removes warmup frames
        at the beginning if warmup_seconds > 0.

        Args:
            video_path: Full path to the .wmv video file.

        Returns:
            frames: np.ndarray of shape (N, H, W, 3), dtype float32
            fps:    float, framerate of the video
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        # Try to read FPS from video metadata, use fallback otherwise
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = self.default_fps

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        if not frames:
            raise ValueError(f"Video is empty: {video_path}")

        frames = np.array(frames, dtype=np.float32)

        # Remove warmup frames (thermal cameras need time to stabilize)
        warmup_frames = int(self.warmup_seconds * fps)
        if warmup_frames > 0 and warmup_frames < len(frames):
            frames = frames[warmup_frames:]

        return frames, fps

    @staticmethod
    def _load_physiology(physio_dir):
        """
        Load heart rate and respiration rate ground truth.

        Heart rate:       mean of all values (nanmean)
        Respiration rate: median of all values (nanmedian),
                          since median is more robust against outliers.

        Args:
            physio_dir: Path to the physiology folder for one
                        subject/task pair.

        Returns:
            hr_bpm:      float, mean heart rate in BPM (or NaN)
            rr_bpm:      float, median respiration rate in BPM (or NaN)
            pulse_rate:  np.ndarray, full heart rate signal (or None)
            resp_rate:   np.ndarray, full respiration rate signal (or None)
        """
        pr_path = os.path.join(physio_dir, "Pulse Rate_BPM.txt")
        rr_path = os.path.join(physio_dir, "Respiration Rate_BPM.txt")

        # ── Heart rate ──
        if os.path.exists(pr_path):
            pulse_rate = np.loadtxt(pr_path)
            hr_bpm = float(np.nanmean(pulse_rate))
        else:
            pulse_rate = None
            hr_bpm = float("nan")

        # ── Respiration rate ──
        if os.path.exists(rr_path):
            resp_rate = np.loadtxt(rr_path)
            rr_bpm = float(np.nanmedian(resp_rate))
        else:
            resp_rate = None
            rr_bpm = float("nan")

        return hr_bpm, rr_bpm, pulse_rate, resp_rate
