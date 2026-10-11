
import re
import html

HEADER_PATTERNS = [
    r"^\s*TITLE\s*:.*$",
    r"^\s*AUTHOR\s*:.*$",
    r"^\s*SOURCE\s*:.*$",
    r"^\s*URL\s*:.*$",
    r"^\s*LINK\s*:.*$",
]

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
MULTISPACE_RE = re.compile(r"\s+")

def remove_metadata_headers(text: str) -> str:
    lines = text.splitlines()
    cleaned_lines = []

    for line in lines:
        is_header = False

        for pat in HEADER_PATTERNS:
            if re.match(pat, line, flags=re.IGNORECASE):
                is_header = True
                break

        if not is_header:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def normalize_text_basic(text: str) -> str:
    text = html.unescape(text)
    text = remove_metadata_headers(text)

    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)

    replacements = {
        "’": "'",
        "‘": "'",
        "`": "'",
        "´": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
        "\u00a0": " ",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", text)
    text = text.lower()
    text = MULTISPACE_RE.sub(" ", text).strip()

    return text


def split_into_sentences(text: str):
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()

    raw_sentences = re.split(r"(?<=[.!?])\s+", text)

    sentences = []

    for s in raw_sentences:
        s = s.strip()
        if len(s) >= 20:
            sentences.append(s)

    return sentences


def tokenize(text: str):
    text = re.sub(r"\d+", " ", text)

    tokens = re.findall(
        r"[^\W\d_]+(?:'[^\W\d_]+)?",
        text,
        flags=re.UNICODE
    )

    cleaned_tokens = []

    for tok in tokens:
        tok = tok.strip("'").lower()

        if len(tok) >= 2:
            cleaned_tokens.append(tok)

    return cleaned_tokens


POSSIBLE_PREFIXES = [
    "jing",
    "nong",
]


def khasi_rule_based_lemmatize_token(token: str) -> str:
    t = token.lower().strip("'")
    t = re.sub(r"'+", "'", t)

    if len(t) <= 4:
        return t

    for pref in POSSIBLE_PREFIXES:
        if t.startswith(pref) and len(t) >= len(pref) + 5:
            stem = t[len(pref):]

            if len(stem) >= 5:
                return stem

    return t


def preprocess_text_for_svd(text: str) -> str:
    text = normalize_text_basic(text)
    tokens = tokenize(text)
    lemmas = [khasi_rule_based_lemmatize_token(tok) for tok in tokens]

    return " ".join(lemmas)


def preprocess_sentence_for_cbow(sentence: str) -> str:
    sentence = normalize_text_basic(sentence)
    tokens = tokenize(sentence)
    lemmas = [khasi_rule_based_lemmatize_token(tok) for tok in tokens]

    return " ".join(lemmas)
