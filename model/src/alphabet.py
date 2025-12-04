# model/src/alphabet.py

"""
Defines the unified multilingual alphabet for the OCR model.
We start with Latin + digits + punctuation + starter sets for
Arabic/Urdu, Hindi (Devanagari), and Bengali.

You can extend or prune these sets later as needed.
"""

# Latin letters + digits
latin_lower = "abcdefghijklmnopqrstuvwxyz"
latin_upper = latin_lower.upper()
digits = "0123456789"

# Common punctuation & whitespace
punct = " .,!?:;'-\"()[]{}@#%&+/=*_<>|\\^~`"

# Arabic + Urdu characters (starter subset; can be expanded)
arabic_urdu = (
    "ابتثجحخدذرزسشصضطظعغفقكلمنهوىي"
    "آأؤئءةٹڈڑںی"
)

# Hindi (Devanagari) starter subset
devanagari = (
    "ँंःअआइईउऊऋएऐओऔ"
    "कखगघङचछजझञटठडढणतथदधन"
    "पफबभमयरलवशषसह"
    "ािीुूृेैोौ्"
)

# Bengali (Bangla) starter subset
bengali = (
    "ঁংঃঅআইঈউঊএঐওঔ"
    "কখগঘঙচছজঝঞটঠডঢণতথদধন"
    "পফবভমযরলশষসহ"
    "ািীুূৃেৈোৌ্"
)

# Build combined alphabet
_raw_alphabet = (
    latin_lower
    + latin_upper
    + digits
    + punct
    + arabic_urdu
    + devanagari
    + bengali
)

# Deduplicate while preserving order
seen = set()
alphabet_chars = []
for ch in _raw_alphabet:
    if ch not in seen:
        seen.add(ch)
        alphabet_chars.append(ch)

ALPHABET = "".join(alphabet_chars)

# CTC "blank" token index = len(ALPHABET)
BLANK_INDEX = len(ALPHABET)

# Helper maps
CHAR_TO_IDX = {ch: i for i, ch in enumerate(ALPHABET)}
IDX_TO_CHAR = {i: ch for i, ch in enumerate(ALPHABET)}


def text_to_indices(text: str):
    """
    Convert a string to a list of character indices.
    Unknown characters are skipped.
    """
    return [CHAR_TO_IDX[ch] for ch in text if ch in CHAR_TO_IDX]


def indices_to_text(indices):
    """
    Convert a list of indices back to a string.
    Ignores BLANK_INDEX and unknown indices.
    """
    chars = []
    for idx in indices:
        if idx == BLANK_INDEX:
            continue
        ch = IDX_TO_CHAR.get(idx)
        if ch is not None:
            chars.append(ch)
    return "".join(chars)


if __name__ == "__main__":
    print("Alphabet length:", len(ALPHABET))
    print(ALPHABET)
    print("Blank index:", BLANK_INDEX)
