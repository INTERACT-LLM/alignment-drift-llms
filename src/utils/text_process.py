import polars as pl
import emoji
import re

# for emohi clearning
def _remove_emojis(text: str) -> str:
    emoji_less_text = emoji.replace_emoji(text)  # remove emojis
    return emoji_less_text

def _remove_emojis_from_df(df: pl.DataFrame, column: str) -> pl.DataFrame:
    cleaned_df = df.with_columns(
         pl.col(column)
        .map_elements(
            lambda x: _remove_emojis(x),
            return_dtype=pl.Utf8,  # ensure the return type is string
        ).alias(column))

    return cleaned_df

# for surprisal 
def _split_text_into_sents(text: str) -> list[str]:
    sents = re.split(r"([.?!])", text)  # capture sent delimiters
    result = []

    for i in range(0, len(sents), 2):  # process in pairs
        sentence = sents[i].strip()
        if i + 1 < len(sents):  # if there is a delimiter, attach it
            sentence += sents[i + 1]
        if sentence:
            result.append(sentence)

    return result