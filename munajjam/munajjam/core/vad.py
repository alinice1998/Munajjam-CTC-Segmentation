import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import List, Tuple

class SileroVAD:
    def __init__(self, sampling_rate: int = 16000):
        self.sampling_rate = sampling_rate
        # Load the Silero VAD model from torch hub
        self.model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True
        )
        self.model.eval()
        (self.get_speech_timestamps,
         self.save_audio,
         self.read_audio,
         self.VADIterator,
         self.collect_chunks) = utils

    def read_audio_ffmpeg(self, file: str | Path, sr: int = 16000) -> torch.Tensor:
        import subprocess
        try:
            out = subprocess.check_output([
                "ffmpeg", "-i", str(file),
                "-f", "s16le", "-acodec", "pcm_s16le",
                "-ac", "1", "-ar", str(sr), "-"
            ], stderr=subprocess.DEVNULL)
            wav = np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
            return torch.from_numpy(wav).unsqueeze(0)
        except Exception as e:
            # Fallback to torchaudio if ffmpeg fails
            return self.read_audio(str(file), sampling_rate=sr)

    def split_audio(self, audio_path: str | Path) -> Tuple[torch.Tensor, List[dict]]:
        """
        Reads audio and returns the waveform and a list of speech timestamps.
        Each timestamp dict contains 'start' and 'end' in samples.
        """
        wav = self.read_audio_ffmpeg(audio_path, sr=self.sampling_rate)
        # get speech timestamps
        speech_timestamps = self.get_speech_timestamps(
            wav, 
            self.model, 
            sampling_rate=self.sampling_rate,
            min_speech_duration_ms=250,
            min_silence_duration_ms=100
        )
        return wav, speech_timestamps

    def merge_short_chunks(self, speech_timestamps: List[dict], max_duration_samples: int) -> List[dict]:
        """
        Merges short speech chunks into longer segments (up to a max duration)
        to optimize wav2vec2 inference and reduce chunking overhead.
        """
        if not speech_timestamps:
            return []

        merged = []
        current_chunk = speech_timestamps[0].copy()

        for i in range(1, len(speech_timestamps)):
            next_chunk = speech_timestamps[i]
            # Calculate duration if we merge them
            duration_if_merged = next_chunk['end'] - current_chunk['start']
            
            if duration_if_merged <= max_duration_samples:
                # Merge
                current_chunk['end'] = next_chunk['end']
            else:
                # Push the current chunk and start a new one
                merged.append(current_chunk)
                current_chunk = next_chunk.copy()

        merged.append(current_chunk)
        return merged
