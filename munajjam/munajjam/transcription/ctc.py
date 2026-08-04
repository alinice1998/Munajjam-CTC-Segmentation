import gc
import re
from pathlib import Path
from typing import Any, List

import torch
import torchaudio
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from ctc_segmentation import (
    prepare_text,
    prepare_token_list,
    ctc_segmentation,
    determine_utterance_segments,
    CtcSegmentationParameters
)

from munajjam.config import get_settings
from munajjam.data import load_surah_ayahs
from munajjam.models import Segment, SegmentType, WordTimestamp
from munajjam.transcription.base import BaseTranscriber


class CTCTranscriber(BaseTranscriber):
    def __init__(self, model_name: str = "jonatasgrosman/wav2vec2-large-xlsr-53-arabic", device: str = "cuda", compute_type: str = "float32"):
        self.model_name = model_name
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Loading wav2vec2 model {self.model_name}...")
        self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        

        # Get vocabulary for ctc-segmentation
        self.vocab = self.processor.tokenizer.get_vocab()
        # ctc-segmentation expects token list mapped to indices
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        # Usually blank is 0 in HF wav2vec2, but let's get it dynamically
        self.blank_id = self.processor.tokenizer.pad_token_id
        if self.blank_id is None:
            self.blank_id = self.vocab.get("<pad>", 0)

    def _normalize_arabic(self, text: str) -> str:
        text = re.sub(r"[\u064B-\u065F\u06D6-\u06DC\u06DF-\u06E8\u06EA-\u06ED]", "", text)
        text = re.sub(r"[أإآٱ]", "ا", text)
        text = re.sub(r"[^\u0621-\u064A\s]", "", text)
        return text.strip()

    def read_audio_ffmpeg(self, file: str | Path, sr: int = 16000) -> torch.Tensor:
        import subprocess
        import torchaudio
        try:
            out = subprocess.check_output([
                "ffmpeg", "-i", str(file),
                "-f", "s16le", "-acodec", "pcm_s16le",
                "-ac", "1", "-ar", str(sr), "-"
            ], stderr=subprocess.DEVNULL)
            wav = np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
            return torch.from_numpy(wav).unsqueeze(0)
        except Exception:
            wav, current_sr = torchaudio.load(str(file))
            if current_sr != sr:
                transform = torchaudio.transforms.Resample(orig_freq=current_sr, new_freq=sr)
                wav = transform(wav)
            if wav.size(0) > 1:
                wav = wav.mean(dim=0, keepdim=True)
            return wav

    def transcribe(
        self,
        audio_path: str | Path,
        *,
        surah_id: int,
        batch_size: int = 16,
    ) -> list[Segment]:
        
        # 1. Load Reference Text
        ayahs = load_surah_ayahs(surah_id)
        if not ayahs:
            return []

        # Normalization and flattening
        ref_words = []
        ayah_boundaries = []
        current_word_idx = 0
        
        for ayah in ayahs:
            words = ayah.text.split()
            ayah_start_word_idx = current_word_idx
            for w in words:
                norm_w = self._normalize_arabic(w)
                if norm_w:
                    ref_words.append(norm_w)
                    current_word_idx += 1
            ayah_boundaries.append({
                "ayah_number": ayah.ayah_number,
                "start_idx": ayah_start_word_idx,
                "end_idx": current_word_idx - 1
            })

        if not ref_words:
            return []

        # 2. Load Audio
        wav = self.read_audio_ffmpeg(audio_path, sr=16000)
        total_samples = wav.shape[1]
        
        # 3. Extract Logits using sliding window to prevent OOM and VAD dropping speech
        window_size = 15 * 16000
        overlap = 2 * 16000
        step = window_size - overlap
        
        all_logits = []
        with torch.no_grad():
            start = 0
            while start < total_samples:
                end = min(start + window_size, total_samples)
                chunk_wav = wav[0, start:end].unsqueeze(0).to(self.device)
                
                inputs = self.processor(chunk_wav.squeeze().cpu().numpy(), sampling_rate=16000, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                logits = self.model(**inputs).logits
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                
                start_frame = start // 320
                end_frame = end // 320
                
                if start > 0:
                    discard_front = (overlap // 2) // 320
                    log_probs = log_probs[discard_front:]
                    start_frame += discard_front
                    
                if end < total_samples:
                    discard_back = (overlap // 2) // 320
                    log_probs = log_probs[:-discard_back]
                    end_frame -= discard_back
                
                all_logits.append((log_probs, start_frame))
                start += step

        # Reconstruct full_log_probs
        total_frames = total_samples // 320 + 1
        num_vocab = len(self.vocab)
        full_log_probs = np.full((total_frames, num_vocab), -100.0, dtype=np.float32)
        full_log_probs[:, self.blank_id] = 0.0

        for lp, start_frame in all_logits:
            num_frames = lp.shape[0]
            end_frame = min(start_frame + num_frames, total_frames)
            actual_frames = end_frame - start_frame
            full_log_probs[start_frame:end_frame] = lp[:actual_frames]

        # 4. CTC Segmentation
        # Prepare text and tokens
        text_str = " ".join(ref_words)
        
        # We need to map vocab characters correctly.
        # ctc_segmentation can take character list
        char_list = [self.inv_vocab[i] for i in range(num_vocab)]
        
        config = CtcSegmentationParameters()
        config.char_list = char_list
        config.blank = self.blank_id
        config.index_duration = 0.02  # Explicitly set to wav2vec2's 20ms frame shift
        # Replace spaces with model's word boundary token if any, or just use space
        word_boundary = self.processor.tokenizer.word_delimiter_token
        if word_boundary is None:
            word_boundary = "|"
            
        config.replace_spaces_with_character = word_boundary

        # We only need one run with the list of words to get word-level timings


        # 5. Build Segments and apply gap filling
        final_segments = []
        # Generate word-level segments directly

        ground_truth_mat, utt_begin_indices = prepare_text(config, ref_words)
        timings, char_probs, state_list = ctc_segmentation(
            config, full_log_probs, ground_truth_mat
        )
        word_segments = determine_utterance_segments(
            config, utt_begin_indices, char_probs, timings, ref_words
        )

        # We rely purely on the timings output by the model (via determine_utterance_segments)
        # without any aggressive gap filling or forcing the last word to the end of the audio.
            
        # Build Ayah segments
        for ayah_bnd in ayah_boundaries:
            a_start = ayah_bnd["start_idx"]
            a_end = ayah_bnd["end_idx"]
            
            if a_start >= len(word_segments) or a_end >= len(word_segments):
                continue
                
            ayah_start_time = word_segments[a_start][0]
            ayah_end_time = word_segments[a_end][1]
            
            words_ts = []
            for w_idx in range(a_start, a_end + 1):
                w_start, w_end, w_conf = word_segments[w_idx]
                
                # Clip confidence to be between 0.0 and 1.0 to avoid Pydantic validation errors
                w_conf = max(0.0, min(1.0, float(w_conf)))
                
                words_ts.append(WordTimestamp(
                    word=ref_words[w_idx],
                    start=w_start,
                    end=w_end,
                    probability=w_conf
                ))
                
            seg = Segment(
                id=ayah_bnd["ayah_number"],
                surah_id=surah_id,
                start=ayah_start_time,
                end=ayah_end_time,
                text=" ".join([w.word for w in words_ts]),
                words=words_ts
            )
            final_segments.append(seg)

        return final_segments
