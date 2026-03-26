"""Helpers for optional background audio playback when replaying recorded videos."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import sounddevice as sd

try:
    # MoviePy keeps the dependency footprint simple for local development.
    from moviepy import VideoFileClip
except Exception:  # pragma: no cover
    VideoFileClip = None


@dataclass
class AudioPlaybackHandle:
    """Small handle used to stop playback and release the decoded clip cleanly."""

    clip: object  # VideoFileClip
    thread: threading.Thread
    sr: int

    def stop(self):
        """Stop playback and release the underlying movie resource."""
        try:
            sd.stop()
        finally:
            try:
                if self.clip is not None:
                    self.clip.close()
            except Exception:
                pass


def start_audio_playback(video_path: Union[str, int], sr: int = 44100) -> Optional[AudioPlaybackHandle]:
    """Start video audio in a background thread when the source is a recorded file."""
    if isinstance(video_path, int):
        return None

    if VideoFileClip is None:
        raise RuntimeError("moviepy is not installed. Run: pip install moviepy")

    clip = VideoFileClip(video_path)

    if clip.audio is None:
        clip.close()
        return None

    audio_np = clip.audio.to_soundarray(fps=sr).astype(np.float32)

    def _play():
        sd.play(audio_np, sr)
        sd.wait()

    t = threading.Thread(target=_play, daemon=True)
    t.start()

    return AudioPlaybackHandle(clip=clip, thread=t, sr=sr)
