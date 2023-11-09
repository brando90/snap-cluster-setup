"""
"""

def token_edit_levenstein_similarity_normalized(text1: str, text2: str) -> float:
    """
    Compute the normalized levenstein distance between two texts.

    ref: https://stackoverflow.com/questions/55487618/looking-for-python-library-which-can-perform-levenshtein-other-edit-distance-at/77393697#77393697
    """
    import nltk
    return 1 - nltk.edit_distance(text1, text2) / max(len(text1), len(text2))