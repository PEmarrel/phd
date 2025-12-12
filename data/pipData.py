from typing import Callable, Optional, List
import unicodedata
import string
from nltk.tokenize import word_tokenize

def pipe_data(
    language: str,
    dataseteur: Callable[..., object],
    window_size: int = 3,
    nb_neg: int = 3,
    subsample_thresh: float = 1.0,
    vocab_size_limit: Optional[int] = None,
    power:float =0.75,
    file: Optional[str] = None,
    files: Optional[list[str]] = None,
    sentences: Optional[List[List[str]]] = None,
    remove_accent: bool = True,
    remove_ponct: bool = True,
    keep_accent: bool = True,
    contraction_map: Optional[dict] = None,
    stop_words:List[str] = []
) -> object:
    """
    """
    if contraction_map is None:
        contraction_map = {
            "n't": " n't", "'re": " 're", "'ve": " 've", "'ll": " 'll",
            "'d": " 'd", "'s": " 's", "'m": " 'm"
        }

    if all([(files is None), (file is None), (sentences is None)],):
        raise AssertionError("One of files, file or sentence must not be None")

    if language not in {"english", "french", None}:
        raise ValueError("language must be 'english' or 'french' or None")

    def remove_accents(text: str) -> str:
        nk = unicodedata.normalize("NFKD", text)
        return "".join(ch for ch in nk if not unicodedata.combining(ch))

    keep = {"'", "’"} if keep_accent else set()
    base_punct = set(string.punctuation)
    extra_punct = set('“”‘’—–…«»')
    punct_to_remove = (base_punct | extra_punct) - keep
    TRANSL_TABLE = str.maketrans('', '', ''.join(sorted(punct_to_remove)))

    tokens_by_sentence: List[List[str]] = []

    if files is not None:
        for name_file in files:
            with open(name_file, encoding="utf-8") as f:
                for line in f:
                    s = line.strip().lower()
                    if not s:
                        continue
                    if remove_accent:
                        s = remove_accents(s)
                    for k, v in contraction_map.items():
                        s = s.replace(k, v)
                    if remove_ponct:
                        s = s.translate(TRANSL_TABLE)
                    s2 = [word for word in s.split() if word not in stop_words]
                    s = " ".join(s2)
                    toks = word_tokenize(s, language=language)
                    if toks:
                        tokens_by_sentence.append(toks)

    elif file is not None:
        with open(file, encoding="utf-8") as f:
            for line in f:
                s = line.strip().lower()
                if not s:
                    continue
                if remove_accent:
                    s = remove_accents(s)
                for k, v in contraction_map.items():
                    s = s.replace(k, v)
                if remove_ponct:
                    s = s.translate(TRANSL_TABLE)
                s2 = [word for word in s.split() if word not in stop_words]
                s = " ".join(s2)                
                if not s:
                    continue
                toks = word_tokenize(s, language=language)
                if toks:
                    tokens_by_sentence.append(toks)
    else:
        tokens_by_sentence = [list(s) for s in sentences if s]

    return dataseteur(
        sentences=tokens_by_sentence,
        window_size=window_size,
        nb_neg=nb_neg,
        subsample_thresh=subsample_thresh,
        power=power,
        vocab_size_limit=vocab_size_limit
    )