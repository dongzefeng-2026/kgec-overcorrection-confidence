"""韩语 GEC 提示模板。

设计约束（GEC 特有）：
- 最小编辑：只改语法错误，不做风格改写（否则 M2 span 匹配大量失配，且教学上有害）；
- 保义：不得增删信息内容；
- 纯输出：只返回修正句，不带解释/引号/前后缀（便于自动解析）。

few-shot 示例取自 KAGAS 论文所述的韩语错误类型（助词、词尾、拼写、分写），
不使用 KoLLA 评测集内的任何句子，避免测试集泄漏。
"""
from __future__ import annotations

import re
from typing import List, Tuple

SYSTEM = (
    "당신은 한국어 문법 교정 전문가입니다. "
    "학습자가 쓴 문장에서 문법 오류만 최소한으로 고치세요.\n"
    "규칙:\n"
    "1. 조사, 어미, 맞춤법, 띄어쓰기 오류만 고치세요.\n"
    "2. 문법적으로 맞는 단어를 동의어로 바꾸지 마세요. "
    "예: '같이'를 '함께'로, '출발했습니다'를 '떠났습니다'로 바꾸면 안 됩니다.\n"
    "3. 내용을 추가하거나 삭제하지 마세요.\n"
    "4. 오류가 없으면 원문을 그대로 출력하세요.\n"
    "5. 설명 없이 교정된 문장만 출력하세요."
)

# (오류 문장, 교정 문장) —— 조사/어미/맞춤법/띄어쓰기 각 1 예시
FEWSHOT: List[tuple] = [
    ("저는 학교를 갑니다.", "저는 학교에 갑니다."),          # 조사(助词)
    ("어제 친구를 만나습니다.", "어제 친구를 만났습니다."),   # 어미(词尾)
    ("음식이 마시있어요.", "음식이 맛있어요."),              # 맞춤법(拼写)
    ("저는 한국어를공부합니다.", "저는 한국어를 공부합니다."),  # 띄어쓰기(分写)
]

USER_TMPL = "다음 문장의 문법 오류를 교정하세요.\n\n문장: {sent}\n교정:"


def build_messages(sentence: str, n_shot: int = 0) -> List[dict]:
    """构造 chat messages。n_shot=0 为零样本；>0 时取前 n_shot 个示例。"""
    msgs = [{"role": "system", "content": SYSTEM}]
    for src, tgt in FEWSHOT[:n_shot]:
        msgs.append({"role": "user", "content": USER_TMPL.format(sent=src)})
        msgs.append({"role": "assistant", "content": tgt})
    msgs.append({"role": "user", "content": USER_TMPL.format(sent=sentence)})
    return msgs


_TRAIL_PAREN = re.compile(r"\s*[\(（\[【][^)）\]】]*[\)）\]】]\s*$")


def strip_commentary(text: str, source: str) -> str:
    """剥离模型附加在句尾的解说，如 "... ( 이미 올바른 문장입니다 . )"。

    仅当源句本身不含括号时才剥离，避免误删合法内容；反复剥离直至稳定
    （部分模型会叠加两段注释）。对全部系统统一施加，剥离率在论文中披露。
    """
    if any(ch in source for ch in "(（[【"):
        return text
    prev = None
    while prev != text:
        prev = text
        text = _TRAIL_PAREN.sub("", text).strip()
    return text or prev


def clean_output(raw: str, source: str) -> Tuple[str, str]:
    """解析模型输出为单个修正句。返回 (修正句, 状态)。

    状态区分三种情况，含义完全不同，必须分开统计：
      ok          —— 正常解析（含模型主动判定无需修改）
      fallback    —— 输出跑飞（空/超长改写），安全回退原句；这是**故障**
      no_change   —— 解析正常且与原句一致；KoLLA 中约 17.6% 的句子本就无需修改，
                     这是**正常行为**，不可与 fallback 混为一谈
    回退是 GEC 的安全默认：宁可不给反馈，也不给错误反馈。
    """
    if raw is None:
        return source, "fallback"
    text = raw.strip()
    for pref in ("교정:", "교정된 문장:", "정답:", "출력:", "Corrected:", "수정:"):
        if text.startswith(pref):
            text = text[len(pref):].strip()
    text = text.strip().strip('"').strip("'").strip("「」『』").strip()
    # 多行时取第一行非空（模型有时附加解释）
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return source, "fallback"
    text = strip_commentary(lines[0], source)
    if not text:
        return source, "fallback"
    # 异常长度视为跑飞（改写/解释），回退原句
    if len(text) > max(200, len(source) * 3):
        return source, "fallback"
    return text, ("no_change" if text.strip() == source.strip() else "ok")
