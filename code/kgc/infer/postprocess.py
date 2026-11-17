"""模型输出的格式规范化（返修新增，作用于 clean_output + retok 之后的 token 串）。

人工核查未匹配编辑（R2.2）发现，原解析规则遗漏了三类输出格式问题，它们会被计为"编辑"：
  1. 回显提示模板中的标签"문장:"（用户模板为 "문장: {sent}\\n교정:"）——HyperCLOVAX 0-shot
     有 90% 的输出以此开头；
  2. 以 "原句 → 改句" 的形式作答；
  3. markdown 强调符号（**고**、*Inception*）；
  4. 输出与展示给模型的句子相同时映射回 .m2 源句（5 个源句的分词与还原规则不一致）。
规则：仅当源句不含相应字符时才处理；对所有系统的贪心输出与采样输出同样施加；
各规则的触发次数逐配置记录并在论文中报告。
"""
from __future__ import annotations

import re
from typing import Dict, Tuple

from kgc.data.tokenize import detok, retok

_LABEL = re.compile(r"^\s*문장\s*:\s*")


def normalize_output(hyp: str, source: str) -> Tuple[str, Dict[str, bool]]:
    flags = {"label_echo": False, "arrow": False, "markdown": False, "identity_mapped": False}
    text = hyp
    if "→" in text and "→" not in source:
        text = text.rsplit("→", 1)[1]
        flags["arrow"] = True
    if _LABEL.match(text) and not _LABEL.match(source):
        text = _LABEL.sub("", text, count=1)
        flags["label_echo"] = True
    if "*" in text and "*" not in source:
        text = text.replace("*", "")
        flags["markdown"] = True
    if flags["arrow"] or flags["label_echo"] or flags["markdown"]:
        text = retok(text)          # 只在规则改动了文本时重新分词（retok 对少数 .m2 源句不幂等）
    if not text.strip():
        return source, flags
    # 输出与展示给模型的句子完全相同 => 视为未修改，映射回 .m2 源句。
    # 少数源句的 .m2 分词与 retok 规则不一致（如"철수 씀."），否则原样返回也会产生伪标点编辑。
    if text == retok(detok(source)) and text != source:
        flags["identity_mapped"] = True
        return source, flags
    return text, flags
