import re

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