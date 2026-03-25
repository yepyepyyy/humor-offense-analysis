import os
import re
import math
from typing import Dict, List

import pandas as pd

import config


# =========================
# Regex patterns
# =========================
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_RE = re.compile(r"(?<!\w)@[A-Za-z0-9_]+")
HASHTAG_RE = re.compile(r"(?<!\w)#[^\s#]+")
EN_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
CHINESE_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")
DIGIT_RE = re.compile(r"\d")
UPPER_RE = re.compile(r"[A-Z]")
LETTER_RE = re.compile(r"[A-Za-z]")
ELLIPSIS_RE = re.compile(r"\.\.\.|……")
MULTI_SPACE_RE = re.compile(r" {2,}")
BLANK_LINE_RE = re.compile(r"\n\s*\n")
PUNCT_RE = re.compile(r"[!！?？,，.。:：;；'\"“”‘’()\[\]{}…—\-]")
REPEATED_PUNCT_RE = re.compile(r"([!！?？.,，。:：;；…])\1+")
ELONGATION_RE = re.compile(r"(.)\1{2,}")
EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FAFF"
    "\u2600-\u26FF"
    "\u2700-\u27BF"
    "]+",
    flags=re.UNICODE,
)

LAUGHTER_PATTERNS = [
    re.compile(r"\b(?:haha|hahaha|hehe|hehehe|lol|lmao|rofl|xd+)\b", re.IGNORECASE),
    re.compile(r"哈+|呵+|hhh+|233+"),
]
NEGATION_PATTERNS = [
    re.compile(r"\b(?:no|not|never|none|nothing|n't)\b", re.IGNORECASE),
    re.compile(r"不|没|無|无|别|莫|休想"),
]
INTENSIFIER_PATTERNS = [
    re.compile(r"\b(?:very|so|too|really|extremely|super)\b", re.IGNORECASE),
    re.compile(r"太|很|超级|超|非常|特别|极其|真?的?太"),
]
INTERJECTION_PATTERNS = [
    re.compile(r"\b(?:oh|ah|wow|omg|ugh|hey|yo)\b", re.IGNORECASE),
    re.compile(r"啊|呀|哇|欸|诶|唉|额|呃|哎"),
]


DEFAULT_TEXT_CANDIDATES = ["tweet", "text", "content", "sentence"]


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def count_pattern_matches(pattern: re.Pattern, text: str) -> int:
    return len(pattern.findall(text))


def max_char_repeat(text: str) -> int:
    if not text:
        return 0
    best = 1
    current = 1
    for i in range(1, len(text)):
        if text[i] == text[i - 1]:
            current += 1
            best = max(best, current)
        else:
            current = 1
    return best


def simple_tokens(text: str) -> List[str]:
    en_words = [w.lower() for w in EN_WORD_RE.findall(text)]
    zh_chars = CHINESE_CHAR_RE.findall(text)
    return en_words + zh_chars


def count_from_patterns(patterns: List[re.Pattern], text: str) -> int:
    total = 0
    for pattern in patterns:
        total += len(pattern.findall(text))
    return total


def detect_text_col(df: pd.DataFrame, text_col: str = None) -> str:
    columns = list(df.columns)

    if text_col is not None:
        if text_col not in columns:
            raise KeyError(f"text_col='{text_col}' 不在表头里，当前列：{columns}")
        return text_col

    for candidate in DEFAULT_TEXT_CANDIDATES:
        if candidate in columns:
            return candidate

    raise ValueError(f"Could not find text column. Available columns: {columns}")


def extract_basic_lf(text: str) -> Dict[str, float]:
    text = "" if text is None or (isinstance(text, float) and math.isnan(text)) else str(text)

    char_count = len(text)
    char_count_no_space = len(re.sub(r"\s", "", text))
    space_count = text.count(" ")
    newline_count = text.count("\n")
    blank_line_count = count_pattern_matches(BLANK_LINE_RE, text)
    multiple_space_count = count_pattern_matches(MULTI_SPACE_RE, text)

    digit_count = count_pattern_matches(DIGIT_RE, text)
    punctuation_count = count_pattern_matches(PUNCT_RE, text)
    exclamation_count = text.count("!") + text.count("！")
    question_count = text.count("?") + text.count("？")
    ellipsis_count = count_pattern_matches(ELLIPSIS_RE, text)
    comma_count = text.count(",") + text.count("，")
    period_count = text.count(".") + text.count("。")
    repeated_punct_count = count_pattern_matches(REPEATED_PUNCT_RE, text)

    hashtag_count = count_pattern_matches(HASHTAG_RE, text)
    mention_count = count_pattern_matches(MENTION_RE, text)
    url_count = count_pattern_matches(URL_RE, text)
    emoji_count = count_pattern_matches(EMOJI_RE, text)

    chinese_char_count = count_pattern_matches(CHINESE_CHAR_RE, text)
    english_letter_count = count_pattern_matches(LETTER_RE, text)
    uppercase_count = count_pattern_matches(UPPER_RE, text)
    english_words = EN_WORD_RE.findall(text)
    english_word_count = len(english_words)
    avg_english_word_len = safe_div(sum(len(w) for w in english_words), english_word_count)

    tokens = simple_tokens(text)
    unique_token_ratio = safe_div(len(set(tokens)), len(tokens))

    laughter_marker_count = count_from_patterns(LAUGHTER_PATTERNS, text)
    negation_count = count_from_patterns(NEGATION_PATTERNS, text)
    intensifier_count = count_from_patterns(INTENSIFIER_PATTERNS, text)
    interjection_count = count_from_patterns(INTERJECTION_PATTERNS, text)

    starts_with_question = int(text.lstrip().startswith(("?", "？")))
    ends_with_exclamation = int(text.rstrip().endswith(("!", "！")))

    max_repeat = max_char_repeat(text)
    elongation_count = count_pattern_matches(ELONGATION_RE, text)

    mixed_language_flag = int(chinese_char_count > 0 and english_letter_count > 0)

    return {
        "char_count": char_count,
        "char_count_no_space": char_count_no_space,
        "space_count": space_count,
        "newline_count": newline_count,
        "blank_line_count": blank_line_count,
        "multiple_space_count": multiple_space_count,
        "digit_count": digit_count,
        "digit_ratio": round(safe_div(digit_count, char_count), 6),
        "punctuation_count": punctuation_count,
        "punctuation_ratio": round(safe_div(punctuation_count, char_count), 6),
        "exclamation_count": exclamation_count,
        "question_count": question_count,
        "ellipsis_count": ellipsis_count,
        "comma_count": comma_count,
        "period_count": period_count,
        "hashtag_count": hashtag_count,
        "mention_count": mention_count,
        "url_count": url_count,
        "emoji_count": emoji_count,
        "repeated_punct_count": repeated_punct_count,
        "max_char_repeat": max_repeat,
        "elongation_count": elongation_count,
        "chinese_char_count": chinese_char_count,
        "english_letter_count": english_letter_count,
        "chinese_ratio": round(safe_div(chinese_char_count, char_count), 6),
        "english_ratio": round(safe_div(english_letter_count, char_count), 6),
        "mixed_language_flag": mixed_language_flag,
        "uppercase_count": uppercase_count,
        "uppercase_ratio": round(safe_div(uppercase_count, char_count), 6),
        "english_word_count": english_word_count,
        "avg_english_word_len": round(avg_english_word_len, 6),
        "unique_token_ratio": round(unique_token_ratio, 6),
        "laughter_marker_count": laughter_marker_count,
        "negation_count": negation_count,
        "intensifier_count": intensifier_count,
        "interjection_count": interjection_count,
        "starts_with_question": starts_with_question,
        "ends_with_exclamation": ends_with_exclamation,
    }


def build_lf_dataframe(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    records = [extract_basic_lf(text) for text in df[text_col].fillna("").astype(str)]
    return pd.DataFrame(records)


def generate_one_lf(dataset_name: str, key: str, options: dict, split: str = "train", text_col: str = None) -> None:
    language = options.get("language")
    if language == "en":
        language_dir = "English"
    elif language == "zh":
        language_dir = "Chinese"
    else:
        raise ValueError(f"Unsupported language in config: {language}")

    input_csv = os.path.join(config.directories["datasets"], language_dir, f"{split}.csv")
    output_dir = os.path.join(config.directories["assets"], dataset_name, key)
    output_csv = os.path.join(output_dir, "lf.csv")

    if not os.path.isfile(input_csv):
        raise FileNotFoundError(f"找不到输入文件：{input_csv}")

    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_csv)
    text_col = detect_text_col(df, text_col=text_col)

    lf_df = build_lf_dataframe(df, text_col=text_col)

    lf_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print("=" * 60)
    print(f"dataset      : {dataset_name}")
    print(f"key          : {key}")
    print(f"language_dir : {language_dir}")
    print(f"input_csv    : {input_csv}")
    print(f"output_csv   : {output_csv}")
    print(f"text_col     : {text_col}")
    print(f"rows         : {len(lf_df)}")
    print(f"lf_cols      : {len(lf_df.columns)}")
    print(f"lf_head      : {list(lf_df.columns[:8])}")


def main():
    for dataset_name, key_map in config.datasets.items():
        for key, options in key_map.items():
            generate_one_lf(dataset_name=dataset_name, key=key, options=options, split="train", text_col=None)


if __name__ == "__main__":
    main()