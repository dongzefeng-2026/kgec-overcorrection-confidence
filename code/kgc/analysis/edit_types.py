"""编辑的韩语形态句法分类（返修 R1.10：语言学层面的错误分析）。

对一条编辑（源 span 的 eojeol 序列 -> 修改后的 eojeol 序列）用 Kiwi 做形态分析，
按以下顺序归入一个主类（规则透明、可复现，不依赖模型）：

  SPACING      分写：去掉空格后两侧字符串相同（한국어를공부합니다 -> 한국어를 공부합니다）
  PUNCT        只差标点
  WORD_ORDER   eojeol 多重集合相同、顺序不同
  PARTICLE     实词语素相同，只有助词（J*）不同（학교를 -> 학교에）
  ENDING       实词语素相同，只有语尾（E*：先语末/终结/连接/转成语尾）不同（만나습니다 -> 만났습니다）
  PARTICLE+ENDING  实词相同，助词与语尾都不同
  SPELLING     实词不同，但字母（jamo）层面高度相似（마시있어요 -> 맛있어요）
  LEXICAL      实词不同且字形不相似：换词（같이 -> 함께）
  INSERTION / DELETION  一侧为空；再按插入/删除内容是助词、语尾还是实词细分
  REWRITE      多 eojeol 改写，以上都不适用

另附两个与语体相关的标记（可与主类同时成立）：
  register_shift  句末终结语尾的语体层级改变（합쇼체 -습니다 / 해요체 -어요 / 해체·해라체 -다·-어）
  honorific_shift 主体尊敬先语末语尾 -시- 的数量改变
"""
from __future__ import annotations

import difflib
import unicodedata
from functools import lru_cache
from typing import Dict, List, Optional, Sequence

from kiwipiepy import Kiwi

_kiwi = Kiwi()

PUNCT_CHARS = set(""".,?!;:…()[]{}<>"'“”‘’~-·""")
SPELL_SIM = 0.75          # jamo 层相似度阈值（>= 视为拼写/形态书写问题）

# 终结语尾 -> 语体层级。按语尾结尾音节判断：Kiwi 用组合用字母（如 'ᆸ니다' 中的 U+11B8），
# 直接列举 'ㅂ니다' 会漏判
_HASIPSIO = ("니다", "니까", "시오", "시다", "디다")
_HAEYO = ("요",)


@lru_cache(maxsize=200000)
def morphs(text: str):
    return tuple((t.form, t.tag) for t in _kiwi.tokenize(text))


def _is_punct_tok(t: str) -> bool:
    return all(ch in PUNCT_CHARS for ch in t)


def _jamo(s: str) -> str:
    return unicodedata.normalize("NFD", s)


def _lexical(ms):
    return [f for f, t in ms if not (t.startswith("J") or t.startswith("E") or t.startswith("S"))]


def _particles(ms):
    return [f for f, t in ms if t.startswith("J")]


def _endings(ms):
    return [f for f, t in ms if t.startswith("E")]


def speech_level(text: str) -> Optional[str]:
    """句末终结语尾的语体层级；无终结语尾时返回 None。"""
    efs = [f for f, t in morphs(text) if t == "EF"]
    if not efs:
        return None
    ef = efs[-1]
    if ef.endswith(_HASIPSIO):
        return "hasipsio"
    if ef.endswith(_HAEYO):
        return "haeyo"
    return "hae/haera"


def honorific_count(text: str) -> int:
    return sum(1 for f, t in morphs(text) if t == "EP" and f in ("시", "으시", "셨", "셔"))


def classify(src_toks: Sequence[str], cor_toks: Sequence[str],
             src_sent: Optional[str] = None, cor_sent: Optional[str] = None) -> Dict[str, object]:
    """返回 {"type": 主类, "register_shift": bool, "honorific_shift": bool}。

    src_sent/cor_sent 为整句（用于语体与敬语标记）；缺省时只看编辑本身。
    """
    S = [t for t in src_toks if t]
    C = [t for t in cor_toks if t and t != "-NONE-"]
    s_join, c_join = "".join(S), "".join(C)
    flags = {"register_shift": False, "honorific_shift": False}
    if src_sent is not None and cor_sent is not None:
        a, b = speech_level(src_sent), speech_level(cor_sent)
        flags["register_shift"] = a is not None and b is not None and a != b
        flags["honorific_shift"] = honorific_count(src_sent) != honorific_count(cor_sent)

    def out(t):
        return {"type": t, **flags}

    if s_join == c_join:
        return out("SPACING")
    if [t for t in S if not _is_punct_tok(t)] == [t for t in C if not _is_punct_tok(t)]:
        return out("PUNCT")
    if S and C and sorted(S) == sorted(C):
        return out("WORD_ORDER")
    if not S or not C:
        side = C or S
        ms = morphs(" ".join(side))
        if not _lexical(ms):
            kind = "PARTICLE" if _particles(ms) and not _endings(ms) else (
                "ENDING" if _endings(ms) and not _particles(ms) else "FUNCTION")
            return out(("INSERTION" if not S else "DELETION") + ":" + kind)
        return out(("INSERTION" if not S else "DELETION") + ":CONTENT")

    ms_s, ms_c = morphs(" ".join(S)), morphs(" ".join(C))
    lex_s, lex_c = _lexical(ms_s), _lexical(ms_c)
    if lex_s == lex_c:
        dp = _particles(ms_s) != _particles(ms_c)
        de = _endings(ms_s) != _endings(ms_c)
        if dp and de:
            return out("PARTICLE+ENDING")
        if dp:
            return out("PARTICLE")
        if de:
            return out("ENDING")
        return out("SPELLING")      # 语素序列相同但书写不同（如缩约、标点外的字形差异）
    sim = difflib.SequenceMatcher(None, _jamo(s_join), _jamo(c_join), autojunk=False).ratio()
    if sim >= SPELL_SIM:
        return out("SPELLING")
    if len(S) > 2 or len(C) > 2:
        return out("REWRITE")
    return out("LEXICAL")


COARSE = {
    "SPACING": "Spacing", "PUNCT": "Punctuation", "WORD_ORDER": "Word order",
    "PARTICLE": "Particle", "ENDING": "Ending", "PARTICLE+ENDING": "Particle+ending",
    "SPELLING": "Spelling", "LEXICAL": "Lexical choice", "REWRITE": "Multi-word rewrite",
}


def coarse(t: str) -> str:
    if t.startswith(("INSERTION", "DELETION")):
        kind = t.split(":")[1]
        if kind == "PARTICLE":
            return "Particle"
        if kind == "ENDING":
            return "Ending"
        return "Insertion/deletion"
    return COARSE.get(t, t)


def edit_texts(source: str, start: int, end: int, corr: str) -> List[List[str]]:
    toks = source.split()
    return [toks[start:end], [] if corr in ("", "-NONE-") else corr.split()]
