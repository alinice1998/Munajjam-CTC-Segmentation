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
    determine_utterance_segments
)

from munajjam.config import get_settings
from munajjam.data import load_surah_ayahs
from munajjam.models import Segment, SegmentType, WordTimestamp
from munajjam.transcription.base import BaseTranscriber
from munajjam.core.vad import SileroVAD

class CTCTranscriber(BaseTranscriber):
    def __init__(self, model_name: str = "jonatasgrosman/wav2vec2-large-xlsr-53-arabic", device: str = "cuda", compute_type: str = "float32"):
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        
        print(f"Loading wav2vec2 model {self.model_name}...")
        self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        
        self.vad = SileroVAD(sampling_rate=16000)

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

        # 2. VAD Splitting
        wav, speech_timestamps = self.vad.split_audio(audio_path)
        # Merge short chunks (e.g. max 15 seconds = 15 * 16000 samples)
        merged_chunks = self.vad.merge_short_chunks(speech_timestamps, max_duration_samples=15 * 16000)

        # Ensure wav is 1D for slicing
        if wav.dim() == 2:
            wav = wav[0]

        # 3. Extract Logits for all chunks
        # To avoid OOM, we process chunk by chunk
        all_logits = []
        with torch.no_grad():
            for chunk in merged_chunks:
                start_sample = chunk['start']
                end_sample = chunk['end']
                chunk_wav = wav[start_sample:end_sample].unsqueeze(0).to(self.device)
                
                inputs = self.processor(chunk_wav.squeeze().cpu().numpy(), sampling_rate=16000, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                logits = self.model(**inputs).logits
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                all_logits.append((log_probs.squeeze(0).cpu().numpy(), start_sample, end_sample))

        # We need to map reference text to ctc_segmentation format
        # CTC-segmentation usually takes the entire unsegmented audio logits.
        # But since we chunked it, we can create a sparse large logit matrix or pad it with blanks.
        # Actually, creating one large array of logits padded with blanks for silence is the easiest way.
        
        total_samples = wav.shape[0]
        # Calculate expected frames. 1 frame = 320 samples for wav2vec2 (usually 16000 / 50 = 320)
        # We can just construct a full length log_probs matrix filled with blank probability = 1.0 (log = 0.0), others = -inf
        num_vocab = len(self.vocab)
        total_frames = total_samples // 320 + 1
        full_log_probs = np.full((total_frames, num_vocab), -100.0, dtype=np.float32)
        full_log_probs[:, self.blank_id] = 0.0

        for lp, start_sample, end_sample in all_logits:
            start_frame = start_sample // 320
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
        
        config = ctc_segmentation.CtcSegmentationParameters()
        config.char_list = char_list
        config.blank = self.blank_id
        # Replace spaces with model's word boundary token if any, or just use space
        word_boundary = self.processor.tokenizer.word_delimiter_token
        if word_boundary is None:
            word_boundary = "|"
            
        config.replace_spaces_with_character = word_boundary

        ground_truth_mat, utt_begin_indices = prepare_text(config, text_str)
        
        timings, char_probs, state_list = ctc_segmentation(
            config, full_log_probs, ground_truth_mat
        )
        
        segments_timing = determine_utterance_segments(
            config, utt_begin_indices, char_probs, timings, text_str
        )

        # 5. Build Segments and apply gap filling
        final_segments = []
        # segments_timing gives (start_time, end_time, confidence) for each utterance (here we can treat each word as an utterance if we split by word, 
        # but prepare_text splits by space natively if we pass a list of strings instead of single string).
        
        # Let's re-run prepare_text with list of words to get word-level timings
        ground_truth_mat, utt_begin_indices = prepare_text(config, ref_words)
        timings, char_probs, state_list = ctc_segmentation(
            config, full_log_probs, ground_truth_mat
        )
        word_segments = determine_utterance_segments(
            config, utt_begin_indices, char_probs, timings, ref_words
        )

        # Apply gap filling: stretch word end to next word start
        for i in range(len(word_segments) - 1):
            word_segments[i] = (word_segments[i][0], word_segments[i+1][0], word_segments[i][2])
            
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
                words_ts.append(WordTimestamp(
                    word=ref_words[w_idx],
                    start=w_start,
                    end=w_end,
                    probability=w_conf
                ))
                
            seg = Segment(
                id=ayah_bnd["ayah_number"],
                start=ayah_start_time,
                end=ayah_end_time,
                text=" ".join([w.word for w in words_ts]),
                words=words_ts
            )
            final_segments.append(seg)

        return final_segments
