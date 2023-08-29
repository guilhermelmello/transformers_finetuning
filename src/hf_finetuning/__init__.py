from mappers import TextMapper


def get_dataset_mapper(tokenizer, text_pairs=False):
    if text_pairs:
        return TextMapper.textpair2token(tokenizer)
    else:
        return TextMapper.text2token(tokenizer)
