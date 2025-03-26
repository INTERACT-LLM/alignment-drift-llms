from text_process import _split_text_into_sents

from lingua import Language, LanguageDetectorBuilder

LANGUAGES = [Language.ENGLISH, Language.SPANISH]

test_text_with_english = "Me alegra saber que estás disfrutando de la clase. A mí también me parece divertida hoy, especialmente porque vamos a hablar sobre las festividades en España. ¿Sabías que la fiesta más famosa es el Carnaval? (I'm glad you're enjoying the class. I find it fun today too, especially because we're going to talk about festivals in Spain. Did you know that the most famous party is Carnival?)"

test_text_without_english = "Me alegra saber que estás disfrutando de la clase. A mí también me parece divertida hoy, especialmente porque vamos a hablar sobre las festividades en España. ¿Sabías que la fiesta más famosa es el Carnaval?"

def _detect_lang(
    text: list[str] | str,
    languages_to_consider: list[Language] = LANGUAGES,
    language_threshold: tuple[Language, float] = (Language.ENGLISH, 0.90),
) -> bool:
    """
    Returns true if language threshold (specific language, confidence that X contains language)
    """
    if not isinstance(text, list):
        sents = _split_text_into_sents(text)
    else:
        sents = text

    detector = LanguageDetectorBuilder.from_languages(*languages_to_consider).build()

    for sent in sents:
        confidence_values = detector.compute_language_confidence_values(sent)

        for confidence in confidence_values:
            if (
                confidence.language == language_threshold[0]
                and confidence.value >= language_threshold[1]
            ):
                print(
                    f"[INFO]: Text contains at least one sentence with {language_threshold[0].name} (confidence of {language_threshold[1]})"
                )
                return True  # If any sentence meets the threshold, return True

    return False  # If no sentences meet the threshold, return False


if __name__ == "__main__":
    # Case 1: Text with both Spanish and English
    print("Test 1: Detection with English")
    detection_with_english = _detect_lang(text=test_text_with_english)
    print(detection_with_english, "\n")

    # Case 2: Text fully in Spanish (no English present)
    print("Test 2: Detection without English")
    detection_without_english = _detect_lang(text=test_text_without_english)
    print(detection_without_english, "\n")