import re
from typing import Callable, Dict, Optional, List, Set
import unicodedata
import string

import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

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

def read_file(file_path: str) -> List[List[str]]:
    with open(file_path, 'r', encoding='utf-8') as f:
        sentences = [line.strip().split() for line in f if line.strip()]
    return sentences

def remove_accents(text: str) -> str:
    """Normalizes text to remove accents (e.g., 'café' -> 'cafe')."""
    nk = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nk if not unicodedata.combining(ch))

def prepare_data(
    file_path: str,
    language: str,
    remove_accent: bool = True,
    remove_punct: bool = True,
    keep_apostrophes: bool = True,
    contraction_map: Optional[Dict[str, str]] = None,
    stop_words: Optional[List[str]] = None
    ) -> List[List[str]]:
    if contraction_map is None:
        contraction_map = {}

    stop_words_set: Set[str] = set(stop_words) if stop_words else set()

    punctuation_chars = set(string.punctuation) | set('“”‘’—–…«»')
    if keep_apostrophes:
        punctuation_chars -= {"'", "’"}
    
    punct_trans_table = str.maketrans({c: " " for c in punctuation_chars})

    tokens_by_sentence: List[List[str]] = []

    with open(file_path, encoding="utf-8") as f:
        for line in f:
            s = line.strip().lower()
            if not s:
                continue
            if remove_accent:
                s = remove_accents(s)
            for k, v in contraction_map.items():
                s = s.replace(k, v)
            if remove_punct:
                s = s.translate(punct_trans_table)
            toks = word_tokenize(s, language=language)
            if stop_words_set:
                toks = [t for t in toks if t not in stop_words_set]
            if toks:
                tokens_by_sentence.append(toks)

    return tokens_by_sentence

def prepare_data_with_intonation(
    file_path: str,
    language: str,
    remove_accent: bool = True,
    remove_punct: bool = True,
    keep_apostrophes: bool = True,
    contraction_map: Optional[Dict[str, str]] = None,
    stop_words: Optional[List[str]] = None,
    break_line: bool = True,
    expand_is_contraction: bool = True
    ) -> List[List[str]]:
    """
    Read a file in this format :
    word (str) intonation (int)
    
    :param file_path: 
    :type file_path: str
    :param language: Description
    :type language: str
    :param remove_accent: Description
    :type remove_accent: bool
    :param remove_punct: Description
    :type remove_punct: bool
    :param keep_apostrophes: Description
    :type keep_apostrophes: bool
    :param contraction_map: Description
    :type contraction_map: Optional[Dict[str, str]]
    :param stop_words: Description
    :type stop_words: Optional[List[str]]
    :param break_line: Description
    :type break_line: bool
    :param expand_is_contraction: Description
    :type expand_is_contraction: bool
    :return: Description
    :rtype: List[List[str]]
    """

    sentence_split_re = re.compile(r'[\.!\?]+')
    
    contraction_re = None
    if contraction_map:
        pattern = "|".join(re.escape(k) for k in sorted(contraction_map.keys(), reverse=True))
        contraction_re = re.compile(f"({pattern})")

    punctuation_chars = set(string.punctuation)
    if keep_apostrophes or expand_is_contraction:
        punctuation_chars -= {"'", "’"}
    
    punct_trans_table = str.maketrans({c: " " for c in punctuation_chars})
    stop_words_set: Set[str] = set(stop_words) if stop_words else set()
    tokens_by_sentence: List[List[str]] = []
    
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            sub_lines = sentence_split_re.split(line.strip().lower()) if break_line else [line.strip().lower()]
            
            for s in sub_lines:
                if not s: continue
                
                if contraction_re:
                    s = contraction_re.sub(lambda m: contraction_map[m.group(0)], s)
                    
                s = s.replace("-", "")
                s = s.replace("—", " ")
                
                if remove_accent:
                    s = remove_accents(s) 

                if remove_punct:
                    s = s.translate(punct_trans_table)

                toks = word_tokenize(s, language=language)

                if expand_is_contraction and language == 'english':
                    tagged = nltk.pos_tag(toks)
                    new_toks = []
                    for word, tag in tagged:
                        if tag == 'POS': continue # Remove possession
                        elif word in ["'s", "’s"] and tag == 'VBZ':
                            new_toks.append("is")
                        else:
                            new_toks.append(word)
                    toks = new_toks

                clean_toks = []
                for t in toks:
                    t_stripped = t.strip("'’")
                    if t_stripped and t_stripped not in stop_words_set:
                        clean_toks.append(t_stripped)
                
                if clean_toks:
                    tokens_by_sentence.append(clean_toks)

    return tokens_by_sentence

def separate_text_intonation(data:List[List[str]]):
    texts = []
    intonations = []
    for sentence in data:
        intonation = sentence[1::2]
        text = sentence[::2]
        if all(t.isalpha() for t in text) and all(t.isdigit() for t in intonation):
            texts.append(text)
            intonations.append(list(map(int, intonation)))
        else:
            print("Warning: Mismatched text and intonation in sentence:", sentence)
            print("Extracted text:", text)
            print("Extracted intonation:", intonation)
            for t in text:
                if not t.isalpha():
                    print(" Non-alpha text token:", t)
            for i in intonation:
                if not i.isdigit():
                    print(" Non-digit intonation token:", i)
            
    return texts, intonations
