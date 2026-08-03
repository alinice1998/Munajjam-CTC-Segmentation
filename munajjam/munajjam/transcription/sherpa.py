import re
from pathlib import Path
from typing import Any
import numpy as np
import soundfile as sf
from rapidfuzz import fuzz
import sherpa_onnx

from munajjam.data import load_surah_ayahs
from munajjam.models import Segment, SegmentType, WordTimestamp
from munajjam.transcription.base import BaseTranscriber
from munajjam.core.downloader import ensure_fastconformer_model


class SherpaTranscriber(BaseTranscriber):
    def __init__(self, use_q8: bool = True):
        self.use_q8 = use_q8
        self.recognizer = None

    def _normalize_arabic(self, text: str) -> str:
        text = re.sub(r"[\u064B-\u065F\u06D6-\u06DC\u06DF-\u06E8\u06EA-\u06ED]", "", text)
        text = re.sub(r"[أإآٱ]", "ا", text)
        text = re.sub(r"[^\u0621-\u064A\s]", "", text)
        return text.strip()

    def _read_audio(self, file_path: str | Path, target_sr: int = 16000) -> np.ndarray:
        import subprocess
        try:
            out = subprocess.check_output([
                "ffmpeg", "-i", str(file_path),
                "-f", "s16le", "-acodec", "pcm_s16le",
                "-ac", "1", "-ar", str(target_sr), "-"
            ], stderr=subprocess.DEVNULL)
            return np.frombuffer(out, np.int16).astype(np.float32) / 32768.0
        except Exception:
            wav, current_sr = sf.read(str(file_path))
            if len(wav.shape) > 1:
                wav = wav.mean(axis=1)
            # Not resampling with scipy here to keep dependencies low, assuming 16kHz for fallback
            # but ffmpeg should succeed 99% of the time.
            return wav.astype(np.float32)

    def transcribe(
        self,
        audio_path: str | Path,
        *,
        surah_id: int,
        batch_size: int = 16,
    ) -> list[Segment]:
        ayahs = load_surah_ayahs(surah_id)
        if not ayahs:
            return []

        ref_words = []
        for ayah in ayahs:
            for w in ayah.text.split():
                ref_words.append(w)

        if not self.recognizer:
            model_path, tokens_path = ensure_fastconformer_model(use_q8=self.use_q8)
            print(f"Loading Sherpa-ONNX model {model_path}...")
            self.recognizer = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
                model=model_path,
                tokens=tokens_path,
                num_threads=2,
            )

        audio_samples = self._read_audio(audio_path)
        stream = self.recognizer.create_stream()
        stream.accept_waveform(16000, audio_samples)
        self.recognizer.decode_stream(stream)
        
        result = stream.result

        # Reconstruct words from SentencePiece tokens
        extracted_words: list[dict[str, Any]] = []
        current_word = ""
        current_start = -1.0
        
        # Tokens usually start with   (U+2581) to denote word start
        for token, ts in zip(result.tokens, result.timestamps):
            if token.startswith(" ") or token.startswith(" "):
                if current_word:
                    extracted_words.append({
                        "word": current_word.replace(" ", "").replace(" ", "").strip(),
                        "start": current_start,
                        "end": ts,
                        "confidence": 0.9 # Sherpa-ONNX offline doesn't easily expose token confidences
                    })
                current_word = token
                current_start = ts
            else:
                current_word += token
                
        if current_word:
            # Estimate last word end
            last_end = result.timestamps[-1] + 0.5 if result.timestamps else 0.5
            extracted_words.append({
                "word": current_word.replace(" ", "").replace(" ", "").strip(),
                "start": current_start,
                "end": last_end,
                "confidence": 0.9
            })

        # --- Dynamic Programming Alignment ---
        n = len(ref_words)
        m = len(extracted_words)
        dp = np.zeros((n + 1, m + 1))

        for i in range(1, n + 1):
            rw = self._normalize_arabic(ref_words[i - 1])
            for j in range(1, m + 1):
                ew = self._normalize_arabic(extracted_words[j - 1]["word"])
                match_score = fuzz.ratio(rw, ew) / 100.0
                if match_score < 0.6:
                    match_score = -1.0
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1] + match_score)

        mapped_alignments: list[dict[str, Any] | None] = [None] * n
        i, j = n, m
        while i > 0 and j > 0:
            rw = self._normalize_arabic(ref_words[i - 1])
            ew = self._normalize_arabic(extracted_words[j - 1]["word"])
            match_score = fuzz.ratio(rw, ew) / 100.0

            if match_score >= 0.6 and dp[i][j] == dp[i - 1][j - 1] + match_score:
                mapped_alignments[i - 1] = extracted_words[j - 1]
                i -= 1
                j -= 1
            elif dp[i][j] == dp[i - 1][j]:
                i -= 1
            else:
                j -= 1

        # Interpolate unmapped words
        w_alignments: list[dict[str, Any]] = []
        for k in range(n):
            if mapped_alignments[k]:
                w_alignments.append(
                    {
                        "word": ref_words[k],
                        "start": mapped_alignments[k]["start"],
                        "end": mapped_alignments[k]["end"],
                        "confidence": mapped_alignments[k]["confidence"],
                    }
                )
            else:
                prev_end = w_alignments[-1]["end"] if w_alignments else 0.0
                w_alignments.append(
                    {
                        "word": ref_words[k],
                        "start": prev_end,
                        "end": prev_end + 0.1,
                        "confidence": 0.0,
                    }
                )

        final_alignments = w_alignments

        try:
            total_duration = sf.info(str(audio_path)).duration
        except Exception:
            total_duration = final_alignments[-1]["end"] + 2.0

        # Gap filling logic
        ayah_boundary_indices = set()
        w_idx = 0
        for ayah in ayahs:
            w_idx += len(ayah.text.split())
            ayah_boundary_indices.add(w_idx - 1)

        for k in range(len(final_alignments)):
            if k > 0:
                if final_alignments[k]["start"] < final_alignments[k - 1]["end"]:
                    final_alignments[k]["start"] = final_alignments[k - 1]["end"]

            if k < len(final_alignments) - 1:
                next_start = final_alignments[k + 1]["start"]
                current_end = final_alignments[k]["end"]
                gap = next_start - current_end

                if gap > 0:
                    if k in ayah_boundary_indices:
                        if gap <= 0.3:
                            start_buffer = min(gap, 0.1)
                            final_alignments[k + 1]["start"] = round(next_start - start_buffer, 3)
                            final_alignments[k]["end"] = round(next_start - start_buffer, 3)
                        elif gap >= 0.4:
                            final_alignments[k + 1]["start"] = round(next_start - 0.2, 3)
                            final_alignments[k]["end"] = round(next_start - 0.2, 3)
                        else:
                            mid = gap / 2.0
                            final_alignments[k + 1]["start"] = round(next_start - mid, 3)
                            final_alignments[k]["end"] = round(next_start - mid, 3)
                    else:
                        if gap > 0.1:
                            final_alignments[k]["end"] = round(next_start - 0.1, 3)
            else:
                final_alignments[k]["end"] = round(total_duration, 3)

            if final_alignments[k]["end"] <= final_alignments[k]["start"]:
                final_alignments[k]["end"] = round(final_alignments[k]["start"] + 0.1, 3)

        # Build segments
        word_idx = 0
        segments = []

        for ayah in ayahs:
            ayah_words_count = len(ayah.text.split())
            ayah_alignments = final_alignments[word_idx : word_idx + ayah_words_count]
            word_idx += ayah_words_count

            if not ayah_alignments:
                continue

            words = []
            avg_conf = 0.0
            for wa in ayah_alignments:
                words.append(
                    WordTimestamp(
                        word=wa["word"],
                        start=wa["start"],
                        end=wa["end"],
                        probability=wa["confidence"],
                    )
                )
                avg_conf += wa["confidence"]

            if words:
                avg_conf /= len(words)
                segments.append(
                    Segment(
                        id=ayah.ayah_number,
                        surah_id=surah_id,
                        start=words[0].start,
                        end=words[-1].end,
                        text=ayah.text,
                        type=SegmentType.AYAH,
                        words=words,
                        confidence=avg_conf,
                    )
                )

        return segments
