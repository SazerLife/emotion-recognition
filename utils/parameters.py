from string import punctuation


SEED = 12345
LABELS_NAMES = ["anger", "disgust", "fear", "happy", "sad", "surprise"]
PUNCTUATION = set(
    list(punctuation)
    + ["``", "...", "''", "«", "»", "…", "”", "”", "“", "-", "–", ".."]
)
