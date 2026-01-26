# preprocessing/mp4_audio_playback.py

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import sounddevice as sd

try:
    # moviepy is "pip-only" and avoids system ffmpeg installs in many setups
    from moviepy import VideoFileClip
except Exception:  # pragma: no cover
    VideoFileClip = None


@dataclass
class AudioPlaybackHandle:
    clip: object  # VideoFileClip
    thread: threading.Thread
    sr: int

    def stop(self):
        try:
            sd.stop()
        finally:
            try:
                # Close moviepy resources
                if self.clip is not None:
                    self.clip.close()
            except Exception:
                pass


def start_audio_playback(video_path: Union[str, int], sr: int = 44100) -> Optional[AudioPlaybackHandle]:
    """
    Start MP4 audio playback asynchronously.
    - If video_path is int (webcam), returns None (no audio playback).
    - If moviepy isn't installed/available, raises RuntimeError with a clear message.
    """
    if isinstance(video_path, int):
        return None  # live camera source -> no audio track to play

    if VideoFileClip is None:
        raise RuntimeError("moviepy is not installed. Run: pip install moviepy")

    clip = VideoFileClip(video_path)

    if clip.audio is None:
        clip.close()
        return None  # MP4 has no audio track

    # Decode to numpy array (N x C), float32
    audio_np = clip.audio.to_soundarray(fps=sr).astype(np.float32)

    def _play():
        sd.play(audio_np, sr)
        sd.wait()

    t = threading.Thread(target=_play, daemon=True)
    t.start()

    return AudioPlaybackHandle(clip=clip, thread=t, sr=sr)
