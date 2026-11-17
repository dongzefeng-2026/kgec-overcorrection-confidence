"""返修稿图 2–7 与补充图（PLOS：TIFF，600 dpi，LZW；宽 ≤ 7.5 in）。

配色（按数据可视化规范，已用 validate_palette 验证）：
  韩语专精 = #2a78d6（slot 1），多语言 = #eb6834（slot 2），第三类 = #1baf7a（slot 3）；
  基线/参考 = 灰色墨色。0-shot 空心、4-shot 实心，作为颜色之外的第二编码；
  ≥2 个系列都有图例；单轴；网格与坐标轴为浅色细线。
用法: python make_figures.py <out_dir>
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kgc.data.kolla import load_m2  # noqa: E402
from kgc.data.kolla_audit import kept_indices  # noqa: E402
from kgc.eval.calibration import risk_coverage, score_edits  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "results/revision"
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else RES / "figures")
OUT.mkdir(parents=True, exist_ok=True)

for f in ("/home/zhao/.fonts/times/Times_New_Roman.ttf", "/home/zhao/.fonts/times/Times_New_Roman_Bold.ttf",
          "/home/zhao/.fonts/times/Times_New_Roman_Italic.ttf", "/home/zhao/.fonts/times/Times_New_Roman_Bold_Italic.ttf",
          "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc", "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"):
    if Path(f).exists():
        font_manager.fontManager.addfont(f)
plt.rcParams.update({
    "font.family": ["Times New Roman", "Noto Serif CJK JP"], "font.size": 8.5, "mathtext.fontset": "stix",
    "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.6, "axes.labelcolor": "#0b0b0b",
    "xtick.color": "#52514e", "ytick.color": "#52514e", "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.5, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})
KOR, MUL, THIRD, YEL = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
KOREAN = {"Midm-2.0", "Kanana-1.5-8B", "A.X-4.0-Light", "EXAONE-3.5-7.8B", "HyperCLOVAX-1.5B"}


def model_of(name):
    return name.split(" (")[0]


def color_of(name):
    return KOR if model_of(name) in KOREAN else MUL


def filled(name):
    return "4-shot" in name


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.tif", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(OUT / f"{stem}_preview.png", dpi=200)
    plt.close(fig)
    print("saved", stem)


def legend_handles(include_noop=False):
    h = [Line2D([], [], marker="o", ls="", mfc=KOR, mec=KOR, ms=5, label="Korean-specialised"),
         Line2D([], [], marker="o", ls="", mfc=MUL, mec=MUL, ms=5, label="Multilingual (Qwen3)"),
         Line2D([], [], marker="o", ls="", mfc="white", mec=INK2, ms=5, label="Zero-shot"),
         Line2D([], [], marker="o", ls="", mfc=INK2, mec=INK2, ms=5, label="Four-shot")]
    if include_noop:
        h.append(Line2D([], [], marker="s", ls="", mfc=MUTED, mec=MUTED, ms=4.5, label="No-op baseline"))
    return h


def point(ax, x, y, name, lo=None, hi=None, size=5):
    c = color_of(name)
    if lo is not None:
        ax.plot([lo, hi], [y, y], color=c, lw=1.0, alpha=0.8, solid_capstyle="round")
    ax.plot([x], [y], marker="o", ms=size, mfc=c if filled(name) else "white", mec=c, mew=1.2, ls="")


# ---------------------------------------------------------------- Fig 2
S = json.load(open(RES / "stats_main.json"))["systems"]
names = sorted([n for n in S if n != "No-op"], key=lambda n: S[n]["F"])
fig, axes = plt.subplots(1, 3, figsize=(7.5, 4.4), sharey=True, gridspec_kw={"wspace": 0.16})
ys = np.arange(len(names))
for ax, key, lo, hi, title in ((axes[0], "F", "F_lo", "F_hi", "A  M² F$_{0.5}$ (two references)"),
                               (axes[1], "GLEU", "GLEU_lo", "GLEU_hi", "B  GLEU"),
                               (axes[2], "false_alarm", "FA_lo", "FA_hi", "C  Changed sentences that\n    needed no correction")):
    for y, n in zip(ys, names):
        point(ax, S[n][key], y, n, S[n][lo], S[n][hi])
    ax.set_title(title, loc="left", fontsize=8, fontweight="bold", color=INK)
    ax.grid(axis="y", visible=False)
    if key == "GLEU":
        nv = S["No-op"]["GLEU"]
        ax.axvline(nv, color=MUTED, lw=0.8)
        ax.text(nv, -0.75, " no-op baseline", color=INK2, fontsize=6.5, va="center", ha="left")
axes[0].set_yticks(ys)
axes[0].set_yticklabels(names)
axes[0].set_xlim(0.35, 0.7)
axes[1].set_xlim(0.34, 0.6)
axes[1].set_xticks([0.4, 0.5, 0.6])
axes[0].set_ylim(-1.2, len(names) - 0.4)
axes[2].set_xlim(0, 1)
axes[2].set_xticks([0, 0.25, 0.5, 0.75, 1.0])
axes[2].xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
fig.legend(handles=legend_handles(), loc="lower center", ncol=4, bbox_to_anchor=(0.55, -0.01), fontsize=7.5)
fig.subplots_adjust(left=0.215, right=0.975, top=0.9, bottom=0.14)
save(fig, "Fig2")

# ---------------------------------------------------------------- Fig 3
T = json.load(open(RES / "edit_types.json"))
ORDER = ["Particle", "Ending", "Spelling", "Spacing", "Lexical choice", "Insertion/deletion",
         "Multi-word rewrite", "Punctuation", "Particle+ending", "Word order"]
gold = {k: (T["gold_counts"]["ann0"].get(k, 0) + T["gold_counts"]["ann1"].get(k, 0)) for k in ORDER}
gtot = sum(gold.values())
pooled = T["pooled"]
stot = sum(v["n"] for v in pooled.values())
fa = T["pooled_false_alarm_pct"]
cats = [c for c in ORDER if c in pooled and pooled[c]["n"] >= 30]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.5, 4.3), sharey=True, gridspec_kw={"wspace": 0.08})
y = np.arange(len(cats))[::-1]
bh = 0.26
a1.barh(y + bh, [100 * gold[c] / gtot for c in cats], bh * 0.9, color="#b9b8b1", label="Reference edits")
a1.barh(y, [100 * pooled[c]["n"] / stot for c in cats], bh * 0.9, color=KOR, label="All system edits")
a1.barh(y - bh, [fa.get(c, 0) for c in cats], bh * 0.9, color=MUL,
        label="System edits in sentences needing no correction")
a1.set_yticks(y)
a1.set_yticklabels(cats)
a1.set_xlabel("Share of edits (%)")
a1.set_title("A  What is edited", loc="left", fontsize=8, fontweight="bold")
a1.grid(axis="y", visible=False)
a1.legend(loc="upper left", bbox_to_anchor=(0.0, -0.15), ncol=1, fontsize=7.5, handlelength=1.2)
for yy, c in zip(y, cats):
    sr, mc = pooled[c]["supported_rate"], pooled[c]["mean_conf"]
    a2.plot([sr, mc], [yy, yy], color="#c3c2b7", lw=1.4, solid_capstyle="round")
    a2.plot([sr], [yy], marker="o", ms=6, mfc=KOR, mec=KOR, ls="")
    a2.plot([mc], [yy], marker="o", ms=6, mfc="white", mec=KOR, mew=1.2, ls="")
a2.set_xlim(0, 1)
a2.set_xlabel("Proportion")
a2.set_title("B  Support versus confidence", loc="left", fontsize=8, fontweight="bold")
a2.grid(axis="y", visible=False)
a2.legend(handles=[Line2D([], [], marker="o", ls="", mfc=KOR, mec=KOR, ms=6, label="Supported by a reference"),
                   Line2D([], [], marker="o", ls="", mfc="white", mec=KOR, ms=6, label="Mean confidence (ESC)")],
          loc="upper left", bbox_to_anchor=(0.0, -0.15), ncol=1, fontsize=7.5)
fig.subplots_adjust(left=0.16, right=0.98, top=0.92, bottom=0.3)
save(fig, "Fig3")

# ---------------------------------------------------------------- Fig 4
models = ["Midm-2.0", "Kanana-1.5-8B", "A.X-4.0-Light", "EXAONE-3.5-7.8B", "HyperCLOVAX-1.5B",
          "Qwen3-8B", "Qwen3-14B", "Qwen3-32B"]
fig, ax = plt.subplots(figsize=(5.2, 3.0))
y = np.arange(len(models))[::-1]
for yy, m in zip(y, models):
    z = T["systems"][f"{m} (0-shot)"]["register_shift_rate"] * 100
    f = T["systems"][f"{m} (4-shot)"]["register_shift_rate"] * 100
    c = KOR if m in KOREAN else MUL
    ax.plot([z, f], [yy, yy], color=c, lw=1.2, alpha=0.7)
    ax.plot([z], [yy], marker="o", ms=5.5, mfc="white", mec=c, mew=1.2, ls="")
    ax.plot([f], [yy], marker="o", ms=5.5, mfc=c, mec=c, ls="")
r_lo, r_hi = sorted(T["ref_register_shift_rate"][f"ann{k}"] * 100 for k in (0, 1))
ax.axvspan(r_lo, r_hi, color="#e1e0d9", lw=0, zorder=0)
ax.text(r_hi + 0.3, len(models) - 0.45, f"Human annotators ({r_lo:.1f}–{r_hi:.1f}%)", fontsize=7, color=INK2,
        va="center", ha="left")
ax.set_yticks(y)
ax.set_yticklabels(models)
ax.set_xlabel("Sentences whose speech level was changed (%)")
ax.grid(axis="y", visible=False)
ax.set_ylim(-0.6, len(models) - 0.1)
ax.legend(handles=legend_handles(), loc="lower right", fontsize=7)
fig.subplots_adjust(left=0.24, right=0.98, top=0.95, bottom=0.16)
save(fig, "Fig4")

# ---------------------------------------------------------------- Fig 5
fig, ax = plt.subplots(figsize=(5.2, 4.0))
y = np.arange(len(names))
for yy, n in zip(y, names):
    v = S[n]
    ax.plot([v["F_ann0"], v["F"]], [yy, yy], color="#e1e0d9", lw=1.2)
    ax.plot([v["F_ann0"]], [yy], marker="o", ms=5, mfc=KOR, mec=KOR, ls="")
    ax.plot([v["F_ann1"]], [yy], marker="o", ms=5, mfc=MUL, mec=MUL, ls="")
    ax.plot([v["F"]], [yy], marker="D", ms=4.5, mfc=THIRD, mec=THIRD, ls="")
ax.set_yticks(y)
ax.set_yticklabels(names)
ax.set_xlabel("M² F$_{0.5}$")
ax.grid(axis="y", visible=False)
ax.legend(handles=[Line2D([], [], marker="o", ls="", mfc=KOR, mec=KOR, ms=5, label="Reference 1 only"),
                   Line2D([], [], marker="o", ls="", mfc=MUL, mec=MUL, ms=5, label="Reference 2 only"),
                   Line2D([], [], marker="D", ls="", mfc=THIRD, mec=THIRD, ms=4.5, label="Both references")],
          loc="lower right", fontsize=7)
fig.subplots_adjust(left=0.3, right=0.98, top=0.97, bottom=0.12)
save(fig, "Fig5")

# ---------------------------------------------------------------- Fig 6
SEL = json.load(open(RES / "selective.json"))["configs"]
OP = json.load(open(RES / "operating_point.json"))
m2 = load_m2(ROOT / "data/kolla_v2/KoLLA_multi-refs.m2")
kept = kept_indices(m2)
show = [("Midm-2.0 (0-shot)", KOR), ("Kanana-1.5-8B (4-shot)", THIRD), ("Qwen3-32B (0-shot)", MUL),
        ("A.X-4.0-Light (4-shot)", YEL)]
fig = plt.figure(figsize=(7.5, 5.6))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.42, wspace=0.62, width_ratios=[1, 1.05])
a1, a2, a3 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
from stats_main import sysname  # noqa: E402
import glob  # noqa: E402
files = {sysname(f): f for f in glob.glob(str(ROOT / "results/runs_v2/*.jsonl"))}
for n, c in show:
    R = {}
    for l in open(files[n], encoding="utf-8"):
        r = json.loads(l)
        R[r["sent_id"]] = r
    E = [e for i in kept for e in score_edits(m2[i], R[i]["greedy"], R[i]["samples"], i)]
    rc = [(cov, risk) for _t, cov, risk in risk_coverage(E) if cov > 0]
    cov = [p[0] for p in rc]
    risk = [p[1] for p in rc]
    a1.plot(cov, risk, color=c, lw=1.6, drawstyle="steps-post", label=n)
    a1.plot([1.0], [risk[0]], marker="o", ms=4, color=c)
a1.set_xlim(0, 1.0)
a1.set_ylim(0, 1)
a1.invert_xaxis()
a1.set_xlabel("Coverage (share of edits shown)")
a1.set_ylabel("Unsupported rate among shown edits")
a1.set_title("A  Risk–coverage", loc="left", fontsize=8, fontweight="bold")
a1.legend(loc="upper right", fontsize=6.8, handlelength=1.5)
cfg = sorted(SEL, key=lambda n: SEL[n]["esc"]["delta50"])
for yy, n in enumerate(cfg):
    e = SEL[n]["esc"]
    point(a2, e["delta50"], yy, n, *e["delta50_ci"], size=4.5)
    h = OP[n]["heldout"]["reduction_pct"]
    a2.plot([h["median"]], [yy], marker="|", ms=8, mew=1.4, color=INK2, ls="")
a2.set_yticks(range(len(cfg)))
a2.set_yticklabels(cfg, fontsize=6.3)
a2.set_xlabel("Reduction in unsupported rate at 50% coverage (%)")
a2.set_title("B  Withholding the less confident half", loc="left", fontsize=8, fontweight="bold")
a2.grid(axis="y", visible=False)
a2.legend(handles=[Line2D([], [], marker="o", ls="-", color=INK2, mfc="white", ms=4, label="Whole corpus (95% CI)"),
                   Line2D([], [], marker="|", ls="", color=INK2, ms=8, mew=1.4, label="Held-out essays (median)")],
          loc="lower right", fontsize=6.5)
EST = [("esc", "Sampling agreement\n(ESC)"), ("mintok", "Minimum token\nprobability"),
       ("seqlp", "Sequence\nprobability"), ("verbalized", "Verbal\nself-rating"), ("random", "Random")]
rng = np.random.default_rng(0)
for k, (src, lab) in enumerate(EST):
    vals = [SEL[n][src]["within_acc"] for n in SEL]
    xs = k + rng.uniform(-0.18, 0.18, len(vals))
    a3.plot(xs, vals, "o", ms=3.8, mfc="white", mec=KOR, mew=0.9, ls="")
    a3.plot([k - 0.28, k + 0.28], [np.median(vals)] * 2, color=INK, lw=1.6)
a3.axhline(0.5, color=MUTED, lw=0.8)
a3.text(-0.45, 0.505, "chance", fontsize=7, color=INK2, va="bottom", ha="left")
a3.set_xticks(range(len(EST)))
a3.set_xticklabels([lab for _s, lab in EST])
a3.set_ylabel("Within-sentence ranking accuracy")
a3.set_ylim(0.35, 0.85)
a3.set_title("C  Ranking a supported above an unsupported edit in the same sentence (one point per configuration)",
             loc="left", fontsize=8, fontweight="bold")
a3.grid(axis="x", visible=False)
fig.subplots_adjust(left=0.08, right=0.9, top=0.95, bottom=0.11)
save(fig, "Fig6")

# ---------------------------------------------------------------- Fig 7
ST = json.load(open(RES / "stratified.json"))
pc = ST["per_config"]
cfg = sorted(pc, key=lambda n: pc[n]["fixed"]["ECE"]["diff"])
fig, axes = plt.subplots(1, 2, figsize=(7.5, 4.0), sharey=True, gridspec_kw={"wspace": 0.08})
for ax, met, title in ((axes[0], "ECE", "A  Expected calibration error"),
                       (axes[1], "E_AURC", "B  Excess AURC (ranking)")):
    for yy, n in enumerate(cfg):
        d = pc[n]["fixed"][met]
        c = color_of(n)
        ax.plot([d["high"], d["low"]], [yy, yy], color="#c3c2b7", lw=1.2)
        ax.plot([d["high"]], [yy], marker="o", ms=5, mfc="white", mec=c, mew=1.2, ls="")
        ax.plot([d["low"]], [yy], marker="o", ms=5, mfc=c, mec=c, ls="")
    ax.set_title(title, loc="left", fontsize=8, fontweight="bold")
    ax.grid(axis="y", visible=False)
axes[0].set_yticks(range(len(cfg)))
axes[0].set_yticklabels(cfg)
fig.legend(handles=[Line2D([], [], marker="o", ls="", mfc="white", mec=INK2, ms=5, label="Higher-rated essays (grammar > 3)"),
                    Line2D([], [], marker="o", ls="", mfc=INK2, mec=INK2, ms=5, label="Lower-rated essays (grammar < 2)"),
                    Line2D([], [], marker="o", ls="", mfc=KOR, mec=KOR, ms=5, label="Korean-specialised"),
                    Line2D([], [], marker="o", ls="", mfc=MUL, mec=MUL, ms=5, label="Multilingual")],
           loc="lower center", ncol=4, fontsize=7, bbox_to_anchor=(0.58, -0.01))
fig.subplots_adjust(left=0.215, right=0.98, top=0.92, bottom=0.15)
save(fig, "Fig7")
