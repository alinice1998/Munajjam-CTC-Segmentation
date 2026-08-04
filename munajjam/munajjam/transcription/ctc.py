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
    ctc_segmentation,
    determine_utterance_segments,
    CtcSegmentationParameters
)

from munajjam.data import load_surah_ayahs
from munajjam.models import Segment, SegmentType, WordTimestamp
from munajjam.transcription.base import BaseTranscriber
from munajjam.core.vad import SileroVAD
from munajjam.core.dp_core import align_segments_dp_with_constraints
from munajjam.core.arabic import normalize_arabic, detect_segment_type

class CTCTranscriber(BaseTranscriber):
    def __init__(self, model_name: str = "jonatasgrosman/wav2vec2-large-xlsr-53-arabic", device: str = "cuda", compute_type: str = "float32"):
        self.model_name = model_name
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Loading wav2vec2 model {self.model_name} on {self.device}...")
        self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        
        print("Loading SileroVAD...")
        self.vad = SileroVAD(sampling_rate=16000)

        # Get vocabulary for ctc-segmentation
        self.vocab = self.processor.tokenizer.get_vocab()
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.blank_id = self.processor.tokenizer.pad_token_id
        if self.blank_id is None:
            self.blank_id = self.vocab.get("<pad>", 0)

    def _normalize_arabic(self, text: str) -> str:
        return normalize_arabic(text)

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

        # 2. VAD: Split audio into speech chunks
        wav, speech_timestamps = self.vad.split_audio(audio_path)
        # Merge short chunks to have decent context (e.g. up to 15 seconds)
        merged_timestamps = self.vad.merge_short_chunks(speech_timestamps, max_duration_samples=15 * 16000)
        
        if not merged_timestamps:
            return []
            
        # 3. ASR: Transcribe each chunk
        transcribed_segments = []
        for chunk in merged_timestamps:
            start_sample = chunk['start']
            end_sample = chunk['end']
            
            chunk_wav = wav[0, start_sample:end_sample].unsqueeze(0).to(self.device)
            inputs = self.processor(chunk_wav.squeeze().cpu().numpy(), sampling_rate=16000, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                logits = self.model(**inputs).logits
                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = self.processor.batch_decode(predicted_ids)[0]
                
            seg_type, _ = detect_segment_type(transcription)
            transcribed_segments.append(Segment(
                id=0,
                surah_id=surah_id,
                start=start_sample / 16000.0,
                end=end_sample / 16000.0,
                text=transcription,
                type=seg_type
            ))
            
        # 4. DP Alignment: Map transcribed segments to Ayahs
        alignment_results = align_segments_dp_with_constraints(
            segments=transcribed_segments,
            ayahs=ayahs,
            silences_ms=None,
            max_segments_per_ayah=10
        )
        
        if not alignment_results:
            return []
            
        # 5. CTC Segmentation on each aligned Ayah
        final_segments = []
        num_vocab = len(self.vocab)
        char_list = [self.inv_vocab[i] for i in range(num_vocab)]
        
        config = CtcSegmentationParameters()
        config.char_list = char_list
        config.blank = self.blank_id
        config.index_duration = 0.02
        word_boundary = self.processor.tokenizer.word_delimiter_token
        if word_boundary is None:
            word_boundary = "|"
        config.replace_spaces_with_character = word_boundary
        
        for result in alignment_results:
            ayah = result.ayah
            ayah_start_s = result.start_time
            ayah_end_s = result.end_time
            
            # Allow a tiny padding for boundary safety (0.2s)
            padding = 0.2
            padded_start_s = max(0.0, ayah_start_s - padding)
            padded_end_s = min(wav.shape[1] / 16000.0, ayah_end_s + padding)
            
            start_sample = int(padded_start_s * 16000)
            end_sample = int(padded_end_s * 16000)
            
            chunk_wav = wav[0, start_sample:end_sample].unsqueeze(0).to(self.device)
            
            # Get logits
            inputs = self.processor(chunk_wav.squeeze().cpu().numpy(), sampling_rate=16000, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                logits = self.model(**inputs).logits
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                
            # Prepare reference words for this Ayah
            ref_words = [w for w in ayah.text.split() if self._normalize_arabic(w)]
            if not ref_words:
                continue
                
            # Run ctc_segmentation
            try:
                ground_truth_mat, utt_begin_indices = prepare_text(config, ref_words)
                timings, char_probs, state_list = ctc_segmentation(
                    config, log_probs, ground_truth_mat
                )
                word_segments = determine_utterance_segments(
                    config, utt_begin_indices, char_probs, timings, ref_words
                )
            except Exception as e:
                print(f"Warning: CTC segmentation failed for ayah {ayah.ayah_number}: {e}")
                # Fallback: distribute evenly
                dur = padded_end_s - padded_start_s
                step = dur / len(ref_words)
                word_segments = [(i*step, (i+1)*step, 0.5) for i in range(len(ref_words))]
            
            # Gap filling within Ayah
            for i in range(len(word_segments) - 1):
                word_segments[i] = (word_segments[i][0], word_segments[i+1][0], word_segments[i][2])
            
            chunk_duration = (end_sample - start_sample) / 16000.0
            if len(word_segments) > 0:
                last_idx = len(word_segments) - 1
                word_segments[last_idx] = (word_segments[last_idx][0], chunk_duration, word_segments[last_idx][2])
                
            # Convert to absolute timestamps
            words_ts = []
            for w_idx in range(len(ref_words)):
                w_start, w_end, w_conf = word_segments[w_idx]
                w_conf = max(0.0, min(1.0, float(w_conf)))
                
                # Undo padding offset
                abs_start = padded_start_s + w_start
                abs_end = padded_start_s + w_end
                
                # Clamp to actual ayah boundaries just in case
                abs_start = max(ayah_start_s, abs_start)
                
                words_ts.append(WordTimestamp(
                    word=ref_words[w_idx],
                    start=abs_start,
                    end=abs_end,
                    probability=w_conf
                ))
                
            seg = Segment(
                id=ayah.ayah_number,
                surah_id=surah_id,
                start=ayah_start_s,
                end=ayah_end_s,
                text=" ".join([w.word for w in words_ts]),
                words=words_ts
            )
            final_segments.append(seg)
            
        return final_segments
