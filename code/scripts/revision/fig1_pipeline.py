"""Fig 1：从模型输出到编辑与置信度的流程实例（Midm-2.0 零样本，KoLLA 第 6 句的真实数据）。

输出 600 dpi LZW 压缩 TIFF（PLOS 要求）与 PNG 预览。字体：拉丁字母用 Times New Roman，韩文用 Noto Serif CJK（衬线体）。
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
for f in ("/home/zhao/.fonts/times/Times_New_Roman.ttf", "/home/zhao/.fonts/times/Times_New_Roman_Bold.ttf",
          "/home/zhao/.fonts/times/Times_New_Roman_Italic.ttf", "/home/zhao/.fonts/times/Times_New_Roman_Bold_Italic.ttf",
          "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc", "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"):
    if Path(f).exists():
        font_manager.fontManager.addfont(f)
plt.rcParams["font.family"] = ["Times New Roman", "Noto Serif CJK JP"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.size"] = 8.5

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
BOX, EDGE, ACCENT = "#f6f6f4", "#c3c2b7", "#2a78d6"

steps = [
    ("1  Input shown to the model",
     [("Source in the .m2 file (tokenised):  ", INK2), ("저녁 우리 비행기를 출발했습니다 .", INK)],
     [("Shown to the model (natural punctuation):  ", INK2), ("저녁 우리 비행기를 출발했습니다.", INK)],
     "'In the evening our plane departed.'  The learner omitted the locative particle on 저녁 and marked the subject 비행기 with the object particle 를."),
    ("2  Clean the response",
     [("Model response:  ", INK2), ("저녁에 우리 비행기를 출발했습니다.", INK)],
     [("Rules: remove labels (교정:, 문장:), trailing comments, arrows and markdown; keep the first line.", INK2)],
     None),
    ("3  Return to the reference token grid",
     [("Re-tokenised output:  ", INK2), ("저녁에 우리 비행기를 출발했습니다 .", INK)],
     [("Punctuation is split off exactly as in the .m2 file, so it cannot create spurious edits.", INK2)],
     None),
    ("4  Extract edits",
     [("Source  ", INK2), ("저녁", ACCENT), ("  우리  비행기를  출발했습니다  .", INK)],
     [("Output  ", INK2), ("저녁에", ACCENT), ("  우리  비행기를  출발했습니다  .", INK)],
     "One edit:  저녁 → 저녁에  ('evening' → 'in the evening', particle insertion)"),
    ("5  Compare with each annotator",
     [("Reference 1:  저녁 ", INK2), ("열 시에", INK), (" 우리 ", INK2), ("비행기가", INK), (" 출발했습니다 .   (inserts 'at ten o'clock'; 를 → 가)", INK2)],
     [("Reference 2:  ", INK2), ("저녁에", ACCENT), (" 우리 ", INK2), ("비행기가", INK), (" 출발했습니다 .   (adds 에; 를 → 가)", INK2)],
     "저녁 → 저녁에 matches Reference 2: the edit is supported.  The missed 를 → 가 lowers recall."),
    ("6  Confidence from ten sampled corrections",
     [("5 × 저녁에 우리 비행기를 …    2 × 저녁에 우리 비행기가 …    3 × 저녁 우리 비행기를 … (unchanged)", INK2)],
     [("저녁 → 저녁에 appears in 7 of 10 samples:  ", INK2), ("confidence = 0.7", ACCENT)],
     "The edit is shown to the learner only if 0.7 is at or above the threshold chosen for the desired coverage."),
]

fig = plt.figure(figsize=(7.5, 7.0), dpi=150)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

top, gap, h = 98.8, 1.5, 15.0
for k, (title, l1, l2, note) in enumerate(steps):
    y1 = top - k * (h + gap)
    y0 = y1 - h
    ax.add_patch(FancyBboxPatch((1.2, y0), 97.6, h, boxstyle="round,pad=0,rounding_size=1.2",
                                fc=BOX, ec=EDGE, lw=0.6))
    ax.text(2.6, y1 - 2.6, title, fontsize=8.5, fontweight="bold", color=INK, va="top")
    for row, segs in enumerate((l1, l2)):
        x = 4.0
        yy = y1 - 5.6 - row * 3.4
        for txt, col in segs:
            t = ax.text(x, yy, txt, fontsize=8, color=col, va="top",
                        fontweight="bold" if col == ACCENT else "normal")
            fig.canvas.draw()
            bb = t.get_window_extent().transformed(ax.transData.inverted())
            x = bb.x1
    if note:
        ax.text(4.0, y0 + 1.0, note, fontsize=7.4, color=MUTED, va="bottom",
                style="italic")
    if k < len(steps) - 1:
        ax.annotate("", xy=(50, y0 - gap + 0.15), xytext=(50, y0 - 0.15),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.8, mutation_scale=7))

OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "Fig1.tif", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
fig.savefig(OUT / "Fig1_preview.png", dpi=200)
print("saved", OUT / "Fig1.tif")
