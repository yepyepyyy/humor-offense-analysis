import pandas as pd
import regex
import string
from typing import Callable, Dict, Optional, Tuple

from bs4.element import whitespace_re

# -----------------------
# Patterns (Unicode-friendly)
# -----------------------
quotation_pattern = r'["”“、《》]'
elongation_pattern = r'(.)\1{2,}'
orphan_dots_pattern = r'[\.]{2,}'
orphan_commas_pattern = r'[\,]{2,}'
dashes_pattern = r' [\-\—] '
orphan_exclamatory_or_interrogative_pattern = r'\s+([\?\!]) '

URL_RE = regex.compile(r"https?://\S+")
BEGIN_MENTION_RE = regex.compile(r"^(@[A-Za-z0-9_]+\s?)+")
MID_MENTION_RE = regex.compile(r"@([A-Za-z0-9_]+)")
DIGIT_RE = regex.compile(r"\b\d+(?:[\.,]\d+)?\b")

# Emoji ranges (won't delete Chinese)
EMOJI_RE = regex.compile(
    "["
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA70-\U0001FAFF"
    "\u2600-\u26FF"
    "\u2700-\u27BF"
    "]+"
)


# -----------------------
# Basic helpers
# -----------------------
def normalize_spaces(text: str) -> str:
    return regex.sub(r"\s+", " ", text).strip()

def preserve_multiple_spaces(text: str) -> str:
    # do nothing: keep multiple spaces
    return text

def preserve_blank_lines(text: str) -> str:
    # do nothing: keep newlines
    return text

# -----------------------
# Step functions (text -> text)
# -----------------------
def step_lettercase(text: str) -> str:
    return text.lower()

def step_hyperlinks(text: str, placeholder: str = "<URL>") -> str:
    return regex.sub(URL_RE, placeholder, text)

def step_mentions(text: str, placeholder: str = "<USER>") -> str:
    text = regex.sub(BEGIN_MENTION_RE, "", text)
    text = regex.sub(MID_MENTION_RE, placeholder, text)
    text = regex.sub(r"(<USER>\s+){2,}", f"{placeholder} ", text)
    return text

def step_emojis(text: str, placeholder: str = "<EMOJI>") -> str:
    return regex.sub(EMOJI_RE, placeholder, text)

def step_punctuation(text: str) -> str:
    # remove ASCII punctuation only (safe for Chinese punctuation)
    return text.translate(str.maketrans("", "", string.punctuation))

def step_digits(text: str, placeholder: str = "<NUM>") -> str:
    return regex.sub(DIGIT_RE, placeholder, text)

def step_elongation(text: str, keep: int = 2) -> str:
    # sooooo -> soo, 哈哈哈哈 -> 哈哈
    return regex.sub(elongation_pattern , lambda m: m.group(1) * keep, text)

# -----------------------
# Steps that need extra resources
# -----------------------
def step_misspellings(text: str, misspell_dict: Dict[str, str]) -> str:
    # English-only in practice (word-boundary replacement)
    if not misspell_dict:
        return text
    pattern = regex.compile(r"\b(" + "|".join(map(regex.escape, misspell_dict.keys())) + r")\b")
    return pattern.sub(lambda m: misspell_dict[m.group(1)], text)

def step_msg_language(text: str, lang_model=None, threshold: float = 0.75) -> Tuple[str, Optional[str]]:
    # Usually does not modify text, only returns detected language
    if lang_model is None:
        return text, None
    labels, probs = lang_model.predict(text, k=1)
    lang = labels[0].replace("__label__", "")
    prob = float(probs[0])
    return text, (lang if prob >= threshold else None)

# -----------------------
# Mapping: step name -> callable
# -----------------------
STEP_MAP: Dict[str, Callable[[str], str]] = {
    "lettercase": step_lettercase,
    "hyperlinks": step_hyperlinks,
    "mentions": step_mentions,
    "emojis": step_emojis,
    "punctuation": step_punctuation,
    "digits": step_digits,
    "elongation": step_elongation,
    "preserve_multiple_spaces": preserve_multiple_spaces,
    "preserve_blank_lines": preserve_blank_lines,
}

def run_pipeline(
    text: str,
    steps,
    *,
    misspell_dict: Optional[Dict[str, str]] = None,
    lang_model=None,
    normalize: bool = True,
) -> Tuple[str, Dict[str, Optional[str]]]:
    """
    Returns:
      processed_text, meta
    meta may contain detected language, etc.
    """
    if text is None:
        text = ""
    text = str(text)
    meta = {"language": None}

    # Decide whether to normalize spaces at the end:
    # if user asked to preserve multiple spaces or blank lines, we should not normalize hard.
    preserve_layout = ("preserve_multiple_spaces" in steps) or ("preserve_blank_lines" in steps)

    for s in steps:
        if s == "misspellings":
            text = step_misspellings(text, misspell_dict or {})
        elif s == "msg_language":
            text, lang = step_msg_language(text, lang_model=lang_model)
            meta["language"] = lang
        else:
            func = STEP_MAP.get(s)
            if func is None:
                raise ValueError(f"Unknown preprocessing step: {s}")
            # Some funcs accept optional params, but with defaults they still work fine
            text = func(text)

    if normalize and (not preserve_layout):
        text = normalize_spaces(text)

    return text, meta


class PreProcessText():
    EN_MSG_LANGUAGE_MAP = {
        "u": "you",
        "ur": "your",
        "r": "are",
        "pls": "please",
        "plz": "please",
        "thx": "thanks",
        "ty": "thank you",
        "btw": "by the way",
        "idk": "i do not know",
        "imo": "in my opinion",
        "imho": "in my humble opinion",
        "tbh": "to be honest",
        "smh": "shaking my head",
        "omg": "oh my god",
        "lol": "laughing",
        "lmao": "laughing",
        "brb": "be right back",
        "afaik": "as far as i know",
        "fyi": "for your information",
        "asap": "as soon as possible",
        "rn": "right now",
        "nvm": "never mind",
        "w/": "with",
        "w/o": "without",
    }
    """    ZH_MSG_LANGUAGE_MAP = {
        "哈哈": "笑",
        "哈哈哈": "笑",
        "哈哈哈哈": "笑",
        "hhhh": "笑",
        "233": "笑",
        "2333": "笑",
        "23333": "笑",
        "hhh": "笑",
        "lol": "英雄联盟",
        "xd": "兄弟",

        "呜呜": "哭",
        "呜呜呜": "哭",
        "555": "哭",
        "5555": "哭",

        "yyds": "永远的神",
        "xswl": "笑死我了",
        "awsl": "啊我死了",
        "dbq": "对不起",
        "nb": "牛",
    }"""

    EN_ACRONYMS = {
        "won\'t": "will not",
        "can\'t": "can not",
        "n\'t": " not",
        "\'re": " are",
        "\'s": " is",
        "\'d": " would",
        "\'ll": " will",
        "\'t": " not",
        "\'ve": " have",
        "\'m": " am",

        "US": "United States",
        "U.S.": "United States",
        "USA": "United States",
        "UK": "United Kingdom",
        "EU": "European Union",
        "UAE": "United Arab Emirates",

        "UN": "United Nations",
        "WHO": "World Health Organization",
        "IMF": "International Monetary Fund",
        "OECD": "Organisation for Economic Co-operation and Development",
        "WTO": "World Trade Organization",
        "NATO": "North Atlantic Treaty Organization",

        "GDP": "gross domestic product",
        "CPI": "consumer price index",
        "IPO": "initial public offering",
        "AI": "artificial intelligence",
        "ML": "machine learning",

        "FB": "Facebook",
        "IG": "Instagram",
        "YT": "YouTube",
        "X": "Twitter",

        "U.S": "United States",
        "U.K.": "United Kingdom",
    }

    ZH_ALIASES = {
        "美利坚": "美国",
        "漂亮国": "美国",
        "米国": "美国",
        "鹰酱": "美国",

        "英伦": "英国",

        "北约": "北约组织",
        "世卫": "世界卫生组织",
        "世贸": "世界贸易组织",

        "阿美莉卡": "美国",
        "霓虹": "日本",

        "央行": "中央银行",
        "国债": "国家债券",

        "AI": "人工智能",
        "GDP": "gross domestic product",
        "哈哈": "笑",
        "哈哈哈": "笑",
        "哈哈哈哈": "笑",
        "hhhh": "笑",
        "233": "笑",
        "2333": "笑",
        "23333": "笑",
        "hhh": "笑",
        "lol": "英雄联盟",
        "xd": "兄弟",

        "呜呜": "哭",
        "呜呜呜": "哭",
        "555": "哭",
        "5555": "哭",

        "yyds": "永远的神",
        "xswl": "笑死我了",
        "awsl": "啊我死了",
        "dbq": "对不起",
        "nb": "牛",
    }

    # -----------------------
    # Patterns (Unicode-friendly)
    # -----------------------
    WHITESPACE_RE = regex.compile(r"\s+")

    # ✅这些必须是“编译后的正则对象”，否则下面 .sub 会报错
    quotation_pattern = regex.compile(r'["”“、《》]')
    elongation_pattern = regex.compile(r'(.)\1{2,}')

    orphan_dots_pattern = regex.compile(r'[\.]{2,}')
    orphan_commas_pattern = regex.compile(r'[\,]{2,}')
    dashes_pattern = regex.compile(r' [\-\—] ')
    orphan_exclamatory_or_interrogative_pattern = regex.compile(r'\s+([\?\!]) ')

    URL_RE = regex.compile(r"https?://\S+")
    hashtag_RE = regex.compile(r'#([\p{L}0-9\_]+)')
    BEGIN_MENTION_RE = regex.compile(r"^(@[A-Za-z0-9_]+\s?)+")
    MID_MENTION_RE = regex.compile(r"@([A-Za-z0-9_]+)")
    DIGIT_RE = regex.compile(r"\b\d+(?:[\.,]\d+)?\b")

    # Emoji ranges (won't delete Chinese)
    EMOJI_RE = regex.compile(
        "["
        "\U0001F300-\U0001F5FF"
        "\U0001F600-\U0001F64F"
        "\U0001F680-\U0001F6FF"
        "\U0001F900-\U0001F9FF"
        "\U0001FA70-\U0001FAFF"
        "\u2600-\u26FF"
        "\u2700-\u27BF"
        "]+"
    )


    # ------------------------
    # Helper
    # ------------------------


    @staticmethod
    def _ensure_series(sentences: pd.Series) -> pd.Series:
        return sentences.fillna("").astype(str)

    @staticmethod
    def camel_case_split(identifier: str) -> str:
        matches = regex.finditer(
            r".+?(?:(?<=\p{Ll})(?=\p{Lu})|(?<=\p{Lu})(?=\p{Lu}\p{Ll})|$)",
            identifier,
        )
        return " ".join(m.group(0) for m in matches)

    def remove_urls(self, sentences: pd.Series) -> pd.Series:
        sentences = self._ensure_series(sentences)
        return sentences.apply(lambda x: self.URL_RE.sub("", x))

    def remove_mentions(self, sentences: pd.Series, placeholder="@USER") -> pd.Series:
        sentences = self._ensure_series(sentences)
        sentences = sentences.apply(lambda x: self.BEGIN_MENTION_RE.sub("", x))
        sentences = sentences.apply(lambda x: self.MID_MENTION_RE.sub(placeholder, x))
        sentences = sentences.apply(lambda x: regex.sub(rf"({regex.escape(placeholder)}\s+){{2,}}", placeholder + " ", x))
        return sentences

    def expand_hashtags(self, sentences: pd.Series) -> pd.Series:
        sentences = self._ensure_series(sentences)
        return sentences.apply(
            lambda x: self.hashtag_RE.sub(lambda m: " " + self.camel_case_split(m.group(1)), x)
        )

    def remove_digits(self, sentences: pd.Series) -> pd.Series:
        sentences = self._ensure_series(sentences)
        return sentences.apply(lambda x: self.DIGIT_RE.sub("", x))

    def remove_whitespaces(self, sentences: pd.Series) -> pd.Series:
        sentences = self._ensure_series(sentences)

        # 合并空白
        sentences = sentences.apply(lambda x: self.WHITESPACE_RE.sub(" ", x))

        # ..  或者 ".   " 规整为 ". "
        sentences = sentences.apply(lambda x: self.orphan_dots_pattern.sub(". ", x))

        # dash 规整成空格（你原来是替换成 ". "，我建议别强行加句号）
        sentences = sentences.apply(lambda x: self.dashes_pattern.sub(" ", x))

        # "  ? " -> "? "
        sentences = sentences.apply(lambda x: self.orphan_exclamatory_or_interrogative_pattern.sub(r"\1 ", x))

        return sentences.str.strip()

    def remove_emojis(self, sentences: pd.Series) -> pd.Series:
        sentences = self._ensure_series(sentences)
        return sentences.apply(lambda x: self.EMOJI_RE.sub("", x))

    def remove_quotations(self, sentences: pd.Series) -> pd.Series:
        sentences = self._ensure_series(sentences)
        return sentences.apply(lambda x: self.quotation_pattern.sub("", x))

    def remove_elongations(self, sentences: pd.Series) -> pd.Series:
        sentences = self._ensure_series(sentences)

        # !!! / ??? 压缩
        for ch in ["!", "¡", "?", "¿"]:
            pat = regex.compile(rf"\{ch}{{2,}}")
            sentences = sentences.apply(lambda x: pat.sub(ch, x))

        # 重复字符压缩
        sentences = sentences.apply(lambda x: self.elongation_pattern.sub(r"\1", x))



        return sentences

    def to_lower(self, sentences: pd.Series) -> pd.Series:
        sentences = self._ensure_series(sentences)
        return sentences.str.lower()

    def remove_punctuation(self, sentences: pd.Series) -> pd.Series:
        sentences = self._ensure_series(sentences)
        return sentences.apply(lambda x: x.translate(str.maketrans("", "", string.punctuation)))


    def expand_acronyms(self, sentences: pd.Series, acronyms: dict) -> pd.Series:
        sentences = self._ensure_series(sentences)

        for key, value in acronyms.items():
            pat = regex.compile(r"(?i)\b" + key + r"\b")
            sentences = sentences.apply(lambda x, p=pat, v=value: p.sub(v, x))

        return sentences
