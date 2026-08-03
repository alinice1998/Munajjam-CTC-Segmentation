from enum import Enum
from munajjam.transcription.base import BaseTranscriber
from munajjam.transcription.whisperx import Whisperx
from munajjam.transcription.sherpa import SherpaTranscriber


class TranscriberBackend(str, Enum):
    WHISPERX = "whisperx"
    SHERPA_ONNX = "sherpa_onnx"


class WhisperFactory:
    @staticmethod
    def get_transcriber(
        backend: TranscriberBackend, model_name: str, device: str = "cuda"
    ) -> BaseTranscriber:
        if backend == TranscriberBackend.WHISPERX:
            return Whisperx(model_name, device)
        elif backend == TranscriberBackend.SHERPA_ONNX:
            return SherpaTranscriber(use_q8=True)
        else:
            raise ValueError(f"Unsupported backend: {backend}")
