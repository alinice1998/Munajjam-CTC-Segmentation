import ctc_segmentation
config = ctc_segmentation.CtcSegmentationParameters()
config.char_list = ['<pad>', 'a', 'b', 'c', '|']
config.blank = 0
config.replace_spaces_with_character = '|'
text = ['a', 'b', 'a', 'c']
_, indices = ctc_segmentation.prepare_text(config, text)
print('Indices:', indices)
