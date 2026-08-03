import numpy as np
from ctc_segmentation import CtcSegmentationParameters, prepare_text

config = CtcSegmentationParameters()
config.char_list = ['<pad>', 'a', 'b', 'c', '|']
config.blank = 0
config.replace_spaces_with_character = '|'

text = ['a', 'b', 'a', 'c']
ground_truth_mat, utt_begin_indices = prepare_text(config, text)
print('utt_begin_indices:', utt_begin_indices)
print('len(text):', len(text))
