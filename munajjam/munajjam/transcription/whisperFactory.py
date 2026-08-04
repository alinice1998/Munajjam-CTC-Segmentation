from enum import Enum
from munajjam.transcription.base import BaseTranscriber
from munajjam.transcription.whisperx import Whisperx
from munajjam.transcription.sherpa import SherpaTranscriber
from munajjam.transcription.ctc import CTCTranscriber


class TranscriberBackend(str, Enum):
    WHISPERX = "whisperx"
    SHERPA_ONNX = "sherpa_onnx"
    CTC_SEGMENTATION = "ctc_segmentation"


class WhisperFactory:
    @staticmethod
    def get_transcriber(
        backend: TranscriberBackend, model_name: str, device: str = "cuda"
    ) -> BaseTranscriber:
        if backend == TranscriberBackend.WHISPERX:
            return Whisperx(model_name, device)
        elif backend == TranscriberBackend.SHERPA_ONNX:
            return SherpaTranscriber(use_q8=True)
        elif backend == TranscriberBackend.CTC_SEGMENTATION:
            return CTCTranscriber(model_name=model_name, device=device)
        else:
            raise ValueError(f"Unsupported backend: {backend}")
