import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

audio = torch.randn(1, 16000 * 15)

inputs = processor(audio.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
logits = model(**inputs).logits
print(f"Audio samples: {audio.shape[1]}")
print(f"Expected frames (samples/320): {audio.shape[1] / 320}")
print(f"Actual frames: {logits.shape[1]}")
