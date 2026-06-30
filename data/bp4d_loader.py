"""
data/bp4d_loader.py
===================
BP4D+ dataset loader. Inherits from BaseLoader.
Only implements BP4D-specific logic.

Optimised: VideoCapture is cached for streaming,
not re-opened for every single frame.
"""

import os
import glob
import cv2
import numpy as np
from data.base_loader import BaseLoader


class BP4DDataset(BaseLoader):

    def __init__(self, root_dir, subjects=None,
                 tasks=None, warmup_seconds=0, fps=25.0,
                 cache_dir="cache",
                 force_preprocess=False):
        self.thermal_dir = os.path.join(
            root_dir, "Thermal")
        self.physio_dir = os.path.join(
            root_dir, "Physiology")
        self.tasks = tasks

        # Video cache for streaming
        self._video_cache_path = None
        self._video_cache_cap = None

        super().__init__(
            root_dir=root_dir,
            subjects=subjects,
            warmup_seconds=warmup_seconds,
            fps=fps,
            cache_dir=cache_dir,
            force_preprocess=force_preprocess,
        )

    # ─────────────────────────────────────────────────────
    # Video cache – open once, reuse
    # ─────────────────────────────────────────────────────

    def _get_video_cap(self, wmv_path):
        """
        Get cached VideoCapture. Opens file only once.
        """
        if self._video_cache_path != wmv_path:
            # Close old one
            if self._video_cache_cap is not None:
                self._video_cache_cap.release()

            self._video_cache_cap = cv2.VideoCapture(
                wmv_path)
            if not self._video_cache_cap.isOpened():
                raise IOError(
                    f"Cannot open video: {wmv_path}")
            self._video_cache_path = wmv_path

        return self._video_cache_cap

    def _release_video_cache(self):
        """Release cached VideoCapture to free RAM."""
        if self._video_cache_cap is not None:
            self._video_cache_cap.release()
            self._video_cache_cap = None
            self._video_cache_path = None

    # ─────────────────────────────────────────────────────
    # BP4D-specific implementations
    # ─────────────────────────────────────────────────────

    def _discover_samples(self, subjects):
        """Find all (subject, task, video_path) combos."""
        if subjects is None:
            subjects = sorted([
                d for d in os.listdir(self.thermal_dir)
                if os.path.isdir(
                    os.path.join(self.thermal_dir, d))
            ])

        samples = []
        for subj in subjects:
            subj_thermal = os.path.join(
                self.thermal_dir, subj)
            wmv_files = sorted(glob.glob(
                os.path.join(subj_thermal, "*.wmv")
            ))
            for wmv_path in wmv_files:
                task = os.path.splitext(
                    os.path.basename(wmv_path))[0]

                if (self.tasks is not None
                        and task not in self.tasks):
                    continue

                physio_path = os.path.join(
                    self.physio_dir, subj, task)
                if os.path.isdir(physio_path):
                    samples.append(
                        (subj, task, wmv_path))

        return samples

    def _load_frames(self, sample_info):
        """Load all frames from .wmv video file."""
        subj, task, wmv_path = sample_info

        cap = cv2.VideoCapture(wmv_path)
        if not cap.isOpened():
            raise IOError(
                f"Cannot open video: {wmv_path}")

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        if not frames:
            raise ValueError(
                f"Video is empty: {wmv_path}")

        return np.array(frames, dtype=np.float16)

    def _load_single_frame(self, sample_info, frame_idx):
        """
        Load one frame from video (for streaming).
        Uses cached VideoCapture – file opened only ONCE.
        """
        subj, task, wmv_path = sample_info

        cap = self._get_video_cap(wmv_path)

        # Check if we need to seek
        current_pos = int(
            cap.get(cv2.CAP_PROP_POS_FRAMES))
        if current_pos != frame_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        ret, frame = cap.read()

        if not ret:
            # Try reopening (some codecs need this)
            self._release_video_cache()
            cap = self._get_video_cap(wmv_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                raise IOError(
                    f"Cannot read frame {frame_idx} "
                    f"from {wmv_path}")

        return frame.astype(np.float16)

    def _get_total_frames(self, sample_info):
        """Get frame count from video metadata."""
        subj, task, wmv_path = sample_info

        cap = self._get_video_cap(wmv_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return total

    def _get_fps(self, sample_info):
        """Get FPS from video metadata."""
        subj, task, wmv_path = sample_info

        cap = self._get_video_cap(wmv_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        return fps if fps > 0 else self.default_fps

    def _load_ground_truth(self, sample_info, fps):
        """
        Load HR/RR ground truth from physiology files.

        Returns BPM values AND raw waveforms.
        """
        subj, task, wmv_path = sample_info
        physio_dir = os.path.join(
            self.physio_dir, subj, task)

        result = {
            "hr_bpm":         float("nan"),
            "rr_bpm":         float("nan"),
            "pulse_rate":     None,
            "resp_rate":      None,
            "bp_waveform":    None,
            "resp_waveform":  None,
            "physio_fps":     1000.0,
        }

        # ── Pulse Rate BPM ──
        pr_path = os.path.join(
            physio_dir, "Pulse Rate_BPM.txt")
        if os.path.exists(pr_path):
            pulse_rate = np.loadtxt(pr_path)
            result["pulse_rate"] = pulse_rate
            result["hr_bpm"] = float(
                np.nanmean(pulse_rate))

        # ── Respiration Rate BPM ──
        rr_path = os.path.join(
            physio_dir, "Respiration Rate_BPM.txt")
        if os.path.exists(rr_path):
            resp_rate = np.loadtxt(rr_path)
            result["resp_rate"] = resp_rate
            result["rr_bpm"] = float(
                np.nanmedian(resp_rate))

        # ── Raw BP waveform ──
        bp_path = os.path.join(
            physio_dir, "BP_mmHg.txt")
        if os.path.exists(bp_path):
            result["bp_waveform"] = np.loadtxt(bp_path)

        # ── Raw Respiration waveform ──
        resp_path = os.path.join(
            physio_dir, "Resp_Volts.txt")
        if os.path.exists(resp_path):
            result["resp_waveform"] = np.loadtxt(
                resp_path)

        # ── Physiology sampling rate ──
        if result["pulse_rate"] is not None:
            total_frames = self._get_total_frames(
                sample_info)
            video_duration = total_frames / fps
            if video_duration > 0:
                result["physio_fps"] = (
                    len(result["pulse_rate"])
                    / video_duration
                )

        return result

    def _get_subject(self, sample_info):
        return sample_info[0]

    def _get_task(self, sample_info):
        return sample_info[1]

    def __del__(self):
        """Clean up video cache on deletion."""
        self._release_video_cache()
