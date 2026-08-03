from transformers import Wav2Vec2Processor
model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"
processor = Wav2Vec2Processor.from_pretrained(model_name)
vocab = processor.tokenizer.get_vocab()
print("pad_token_id:", processor.tokenizer.pad_token_id)
print("word_delimiter_token_id:", processor.tokenizer.word_delimiter_token_id)
print("vocab '<pad>':", vocab.get('<pad>'))
print("vocab '[PAD]':", vocab.get('[PAD]'))
