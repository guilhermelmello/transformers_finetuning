"""
"""


class TextMapper:
    @staticmethod
    def text2token(tokenizer):
        def mapper(batch):
            return tokenizer(
                text=batch['text'],
                truncation=True,
                padding=False
            )
        return mapper

    @staticmethod
    def textpair2token(tokenizer):
        def mapper(batch):
            return tokenizer(
                text=batch['text'],
                text_pair=batch['text_pair'],
                truncation=True,
                padding=False
            )
        return mapper
