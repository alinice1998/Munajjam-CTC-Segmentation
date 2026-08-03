from enum import Enum
from typing import Literal

from munajjam.transcription.whisper import WhisperTranscriber
from munajjam.transcription.ctc import CTCTranscriber


class WhisperBackend(Enum):
    OPENAI = "openai"
    FASTERWHISPER = "fasterwhisper"
    CTC = "ctc"


class WhisperFactory:
    def create_whisper(
        self,
        backend: WhisperBackend,
        model_name: str,
        device: Literal["auto", "cpu", "cuda", "mps"] = "cuda",
        compute_type: str = "float32",
    ) -> WhisperTranscriber | CTCTranscriber:
        if backend == WhisperBackend.FASTERWHISPER:
            return WhisperTranscriber(
                model_id=model_name, device=device, model_type="faster-whisper"
            )
        elif backend == WhisperBackend.OPENAI:
            return WhisperTranscriber(model_id=model_name, device=device, model_type="transformers")
        elif backend == WhisperBackend.CTC:
            return CTCTranscriber(model_name=model_name, device=device, compute_type=compute_type)
        else:
            raise ValueError(f"Unsupported backend: {backend}")
