"""补充材料：S1–S5 Table（xlsx）与 S1–S3 Fig（TIFF）。

用法: python make_supporting.py <out_dir>
"""
import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kgc.data.kolla import load_m2  # noqa: E402
from kgc.data.kolla_audit import kept_indices  # noqa: E402
from kgc.eval.calibration import risk_coverage, score_edits  # noqa: E402
from kgc.eval.m2_official import corpus_multi, corpus_stats_for_runs  # noqa: E402
from stats_main import sysname  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "results/revision"
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else RES / "supporting")
OUT.mkdir(parents=True, exist_ok=True)
MODELS = ["Midm-2.0", "Kanana-1.5-8B", "A.X-4.0-Light", "EXAONE-3.5-7.8B", "HyperCLOVAX-1.5B",
          "Qwen3-8B", "Qwen3-14B", "Qwen3-32B"]
NAMES = [f"{m} ({s}-shot)" for m in MODELS for s in ("0", "4")]


def J(n):
    return json.load(open(RES / n, encoding="utf-8"))


def write_xlsx(path, sheets, notes):
    with pd.ExcelWriter(path) as xw:
        pd.DataFrame({"Notes": notes}).to_excel(xw, sheet_name="Notes", index=False)
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name[:31], index=False)
    print("saved", path)


# ---------------- S1 Table
SM, FULL = J("stats_main.json")["systems"], J("stats_main_full1418.json")["systems"]
PC = json.load(open(ROOT / "results/runs_v2/_parse_fix_counts.json"))
m2 = load_m2(ROOT / "data/kolla_v2/KoLLA_multi-refs.m2")
kept = kept_indices(m2)
files = {sysname(f): f for f in glob.glob(str(ROOT / "results/runs_v2/*.jsonl"))}
rows = []
for n in NAMES + ["No-op"]:
    R = {}
    for l in open(files[n], encoding="utf-8"):
        r = json.loads(l)
        R[r["sent_id"]] = r
    st_v1 = corpus_stats_for_runs(m2, [R[i].get("greedy_v1", R[i]["greedy"]) for i in range(len(m2))])
    pc = PC[Path(files[n]).name]
    rows.append({"Configuration": n,
                 "M2 F0.5 (1,393; revised parsing)": round(SM[n]["F"], 4),
                 "M2 F0.5 (all 1,418 entries)": round(FULL[n]["F"], 4),
                 "M2 F0.5 (1,393; original parsing)": round(corpus_multi(st_v1, order=kept)[2], 4),
                 "Sentence-averaged F0.5 (original metric)": round(SM[n]["sentF"], 4),
                 "GLEU (1,393)": round(SM[n]["GLEU"], 4), "GLEU (1,418)": round(FULL[n]["GLEU"], 4),
                 "Changed unneeded (%), 1,393": round(100 * SM[n]["false_alarm"], 1),
                 "Outputs with echoed label": pc["greedy_label_echo"], "Outputs with arrow": pc["greedy_arrow"],
                 "Outputs with markdown": pc["greedy_markdown"], "Unchanged outputs mapped to source": pc["greedy_identity_mapped"],
                 "Sampled outputs changed by the rules": pc["samples_changed"]})
write_xlsx(OUT / "S1_Table.xlsx", {"S1 Table": pd.DataFrame(rows)},
           ["S1 Table. Correction quality under alternative protocols and the activity of the output-parsing rules.",
            "1,393: after excluding the 25 audited entries; 1,418: all entries. Original parsing: without the rules for echoed labels, arrows, markdown and unchanged outputs added in the revision.",
            "Sentence-averaged F0.5 is the metric of the original submission; it gives full credit to an unchanged sentence whenever either annotator left it unchanged."])

# ---------------- S2 Table
LS = J("length_stats.json")["rows"]
rows = []
for k, v in LS.items():
    rows.append({"Text": k, "Eojeol": v["eojeol"]["total"], "Eojeol change vs source (%)": round(v["eojeol"]["vs_source_pct"], 1),
                 "Eojeol change vs mean reference (%)": round(v["eojeol"]["vs_ref_mean_pct"], 1),
                 "Mean |difference| to nearest reference (eojeol per sentence)": round(v["eojeol"]["abs_diff_nearest_ref"], 2),
                 "Sentences with changed eojeol count (%)": round(100 * v["eojeol"]["changed_len_rate"], 1),
                 "Morphemes": v["morpheme"]["total"], "Morpheme change vs source (%)": round(v["morpheme"]["vs_source_pct"], 1),
                 "Morpheme change vs mean reference (%)": round(v["morpheme"]["vs_ref_mean_pct"], 1),
                 "Mean |difference| to nearest reference (morphemes per sentence)": round(v["morpheme"]["abs_diff_nearest_ref"], 2)})
write_xlsx(OUT / "S2_Table.xlsx", {"S2 Table": pd.DataFrame(rows)},
           ["S2 Table. Word counts of sources, references and system outputs (1,393 sentences).",
            "Eojeol exclude tokens consisting only of punctuation; morphemes are Kiwi morphemes excluding symbols."])

# ---------------- S3 Table
T = J("stats_main.json")["tests"]
rows = [{"Family": t["family"], "System A": t["a"], "System B": t["b"],
         "Difference in M2 F0.5 (A-B)": round(t["dF"], 4), "95% CI": f"[{t['dF_lo']:.4f}, {t['dF_hi']:.4f}]",
         "p": round(t["p_F"], 4), "p (Holm)": round(t["p_F_holm"], 4),
         "Difference in GLEU (A-B)": round(t["dGLEU"], 4), "95% CI ": f"[{t['dGLEU_lo']:.4f}, {t['dGLEU_hi']:.4f}]",
         "p ": round(t["p_GLEU"], 4), "p (Holm) ": round(t["p_GLEU_holm"], 4)} for t in T]
write_xlsx(OUT / "S3_Table.xlsx", {"S3 Table": pd.DataFrame(rows)},
           ["S3 Table. Paired comparisons between systems on KoLLA v2 (1,393 sentences).",
            "p: essay-level approximate randomization test (2,000 permutations); CI: paired essay-cluster bootstrap (1,000 resamples); Holm adjustment within each family."])

# ---------------- S4 Table
GR = J("groups_register.json")
rows = []
for n in NAMES:
    r = GR["configs"][n]
    d = {"Configuration": n}
    for g in ("FB", "FI", "HB", "HI"):
        d[f"{g} F0.5"] = round(r[g]["F"], 3)
        d[f"{g} 95% CI"] = f"[{r[g]['lo']:.3f}, {r[g]['hi']:.3f}]"
        d[f"{g} changed unneeded (%)"] = None if r[g]["false_alarm"] is None else round(100 * r[g]["false_alarm"], 1)
    d["Lowest-scoring group"] = r["worst_group"]
    rows.append(d)
gi = pd.DataFrame([{"Group": g, **{k: (round(v, 2) if isinstance(v, float) else v) for k, v in GR["group_info"][g].items()}}
                   for g in ("FB", "FI", "HB", "HI")])
write_xlsx(OUT / "S4_Table.xlsx", {"S4 Table": pd.DataFrame(rows), "Group information": gi},
           ["S4 Table. Correction quality by learner group. FB/FI: foreign beginner/intermediate; HB/HI: heritage beginner/intermediate.",
            "F0.5: corpus-level M2 against both references; CI: essay-cluster bootstrap. Changed unneeded: share of sentences unchanged by both annotators that the system altered."])

# ---------------- S5 Table
ST, SJ = J("stratified.json"), J("stratified_joint.json")
rows = []
for n in NAMES:
    for scheme in ("fixed", "tertile"):
        for met in ("ECE", "E_AURC"):
            d = ST["per_config"][n][scheme][met]
            rows.append({"Configuration": n, "Strata": scheme, "Measure": met, "Low": round(d["low"], 4),
                         "High": round(d["high"], 4), "Low - high": round(d["diff"], 4),
                         "95% CI": f"[{d['ci'][0]:.4f}, {d['ci'][1]:.4f}]", "Permutation p": round(d["p_perm"], 4),
                         "Edits (low)": d["n_low"], "Edits (high)": d["n_high"]})
am = pd.DataFrame([{"Test": k, **{kk: vv for kk, vv in v.items() if kk != "per_model"}} for k, v in ST["across_models"].items()])
gee = pd.DataFrame([{"Model": lab, "Term": term, **vals} for lab, d in ST["gee"].items() for term, vals in d.items()
                    if isinstance(vals, dict)])
joint = pd.DataFrame([{"Measure": k, **v} for k, v in SJ["tests"].items()])
write_xlsx(OUT / "S5_Table.xlsx", {"Per configuration": pd.DataFrame(rows), "Across models (Wilcoxon)": am,
                                   "Joint permutation": joint, "GEE": gee},
           ["S5 Table. Calibration and ranking by proficiency stratum.",
            "Fixed strata: grammar rating < 2 (low) and > 3 (high); tertile strata: cut points 2.0 and 3.0, ties to the lower stratum.",
            "Permutation tests reassign stratum labels among essays; the joint test applies one permutation to all configurations.",
            "GEE: logistic model of edit support with confidence, stratum and their interaction, configuration fixed effects, essays as clusters."])

# ---------------- 补充图
for f in ("/home/zhao/.fonts/times/Times_New_Roman.ttf", "/home/zhao/.fonts/times/Times_New_Roman_Bold.ttf"):
    if Path(f).exists():
        font_manager.fontManager.addfont(f)
plt.rcParams.update({"font.family": ["Times New Roman"], "font.size": 8, "mathtext.fontset": "stix",
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#c3c2b7",
                     "grid.color": "#e1e0d9", "axes.grid": True, "grid.linewidth": 0.5})
KOR, MUL, INK2 = "#2a78d6", "#eb6834", "#52514e"
KOREAN = set(MODELS[:5])


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.tif", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(OUT / f"{stem}_preview.png", dpi=150)
    plt.close(fig)
    print("saved", stem)


fig, axes = plt.subplots(4, 4, figsize=(7.5, 7.5), sharex=True, sharey=True)
for ax, n in zip(axes.flat, NAMES):
    R = {}
    for l in open(files[n], encoding="utf-8"):
        r = json.loads(l)
        R[r["sent_id"]] = r
    E = [e for i in kept for e in score_edits(m2[i], R[i]["greedy"], R[i]["samples"], i)]
    rc = [(c, k) for _t, c, k in risk_coverage(E) if c > 0]
    ax.plot([p[0] for p in rc], [p[1] for p in rc], drawstyle="steps-post",
            color=KOR if n.split(" (")[0] in KOREAN else MUL, lw=1.4)
    ax.set_title(n, fontsize=7.5)
    ax.set_xlim(1.0, 0)
    ax.set_ylim(0, 1)
fig.supxlabel("Coverage (share of edits shown)")
fig.supylabel("Unsupported rate among shown edits")
fig.tight_layout()
save(fig, "S1_Fig")

MV = J("majority_vote.json")
fig, axes = plt.subplots(4, 4, figsize=(7.5, 7.5), sharex=True)
for ax, n in zip(axes.flat, NAMES):
    cur = MV[n]["esc_curve"]
    taus = sorted(cur, key=float)
    c = KOR if n.split(" (")[0] in KOREAN else MUL
    ax.plot([float(t) for t in taus], [cur[t]["F"] for t in taus], "o-", ms=2.5, color=c, lw=1.2)
    ax.plot([0.6], [MV[n]["mv_all"]["F"]], marker="D", ms=4.5, mfc="white", mec=INK2, ls="")
    ax.set_title(n, fontsize=7.5)
fig.supxlabel("Confidence threshold τ (edits with ESC ≥ τ are applied)")
fig.supylabel("M² F$_{0.5}$")
fig.legend(handles=[plt.Line2D([], [], color=INK2, marker="o", ms=3, label="Greedy edits with ESC ≥ τ"),
                    plt.Line2D([], [], color=INK2, marker="D", ms=4.5, mfc="white", ls="", label="Majority voting over all sampled edits")],
           loc="lower center", ncol=2, frameon=False)
fig.tight_layout(rect=(0, 0.04, 1, 1))
save(fig, "S2_Fig")

OP = J("operating_point.json")
fig, ax = plt.subplots(figsize=(5.2, 3.2))
Ns = [2, 4, 6, 8, 10]
allv = []
for n in NAMES:
    v = [OP[n]["n_sensitivity"][str(k)]["E_AURC"] for k in Ns]
    allv.append(v)
    ax.plot(Ns, v, color=KOR if n.split(" (")[0] in KOREAN else MUL, lw=0.8, alpha=0.6)
ax.plot(Ns, np.median(allv, axis=0), color="#0b0b0b", lw=2, label="Median over configurations")
ax.set_xlabel("Number of sampled corrections N")
ax.set_ylabel("Excess AURC (lower is better)")
ax.set_xticks(Ns)
ax.legend(frameon=False)
fig.tight_layout()
save(fig, "S3_Fig")
