# Over-correction and confidence in prompted LLMs for Korean GEC: code and data

This repository accompanies the article *Over-correction and confidence in prompted language models for Korean grammatical error correction: An exploratory evaluation on second-language and heritage learner writing* (PLOS ONE, under review, PONE-D-26-44963).

## Contents

| Path | Description |
|---|---|
| `code/kgc/` | Library: corpus loading, essay linking and audit (`data/`), prompting and inference with vLLM (`infer/`), output parsing (`infer/prompts.py`, `infer/postprocess.py`), edit extraction and matching (`eval/calibration.py`), wrappers around the official M² scorer and GLEU (`eval/m2_official.py`, `eval/gleu_official.py`), resampling statistics (`eval/stats.py`), and Korean edit categories (`analysis/edit_types.py`). |
| `code/scripts/revision/` | Scripts that produce every number, table and figure of the revised article (see below). The shell scripts record the inference runs as executed on the GPU server and contain that server's paths. |
| `results/runs/` | KoLLA v2 outputs of the fourteen configurations of the original study and the no-op baseline as generated: greedy and ten sampled corrections, token log-probabilities and self-ratings. |
| `results/runs_rev/` | The same for Qwen3-32B (added in revision), with the model revisions used. |
| `results/runs_v2/` | All sixteen configurations and the baseline after the output-parsing rules; the outputs before the rules are kept in the fields `greedy_v1` and `samples_v1`. `_parse_fix_counts.json` counts how often each rule applied. All analyses read these files. |
| `results/revision/` | All analysis results as JSON; the 304 reviewed edits (`unsupported_sample.jsonl`, with the hidden key in `unsupported_sample_key.jsonl`) and both judges' labels and rationales (`judge_A_claude.jsonl`, `judge_B_qwen32b.jsonl`). |
| `korlang8/` | Construction of the Kor-Lang8 evaluation sample (filters, seed, and the line numbers of the pairs used in the source file) and its aggregate results. The sentence pairs and the model outputs on them are third-party material and are not redistributed; see *Data availability* below for how to obtain them. |
| `docs/` | S1–S4 Appendices of the article: formal definitions; prompts, generation settings, model revisions and data sources; rules for edit categories; the guideline for the review of unsupported edits. |

## Reproducing the analyses

1. Python ≥ 3.10 with `numpy`, `scipy`, `pandas`, `statsmodels`, `scikit-learn`, `matplotlib`, `openpyxl` and `kiwipiepy==0.24.1`; inference additionally needs `vllm==0.28.0`.
2. Obtain the reference scorers: `bash code/scripts/revision/get_vendor.sh` clones `soyoung97/Standard_Korean_GEC` at a fixed commit (M² scorer and GLEU, under their own licences) into `code/vendor/`.
3. Download KoLLA v2 (Zenodo record 15287129) into `data/kolla_v2/`.
4. From `code/`, run:
   - `python scripts/revision/clean_runs.py ../results/runs ../results/runs_v2` and `python scripts/revision/clean_runs.py ../results/runs_rev ../results/runs_v2` (parsing rules; this regenerates `results/runs_v2/`)
   - `python scripts/revision/misc_values.py` (essay linking, unchanged sentences, supported share before and after the matching correction)
   - `python scripts/revision/stats_main.py` and `python scripts/revision/stats_main.py --full --out ../results/revision/stats_main_full1418.json` (Table 2, S1 and S3 Tables)
   - `python scripts/revision/edit_type_analysis.py`, `groups_register.py`, `length_stats.py` (Figs 3–4, S2 and S4 Tables)
   - `python scripts/revision/selective.py`, `operating_point.py`, `majority_vote.py` (Tables 5–6, Fig 6, S1–S3 Figs)
   - `python scripts/revision/stratified.py`, `stratified_joint.py` (Fig 7, S5 Table)
   - `python scripts/revision/sample_unsupported.py` (the review sample; judge B: `judge_qwen.py`)
   - `python scripts/revision/make_figures.py <dir>`, `fig1_pipeline.py <dir>` and `make_supporting.py <dir>`
5. Kor-Lang8: place `kor_lang8.txt` (see *Data availability*) in `data/kor_lang8/raw/` and run `python scripts/revision/build_korlang8.py` from `code/`; it rebuilds `data/kor_lang8/korlang8_eval.m2` and a manifest that should equal `korlang8/korlang8_eval.manifest.json`. `korlang8_worker.sh` runs inference, and `korlang8_analysis.py` produces Table 7.
6. Inference on KoLLA: `code/kgc/infer/run_vllm.py` with the settings in S2 Appendix (`run_qwen32b.sh` shows the command line used).

## Data availability

- **Public without restriction (this repository):** all code, all model outputs on KoLLA v2, the audit rules and exclusions, all evaluation results, the 304 reviewed edits with both judges' labels and rationales, and the Kor-Lang8 construction rules, line numbers and aggregate results.
- **KoLLA v2** (primary corpus): Zenodo, release of 26 April 2025, <https://doi.org/10.5281/zenodo.15287129> (GPL-3.0).
- **Kor-Lang8** (third-party, supplementary external check only): derived from the Lang-8 Learner Corpora, which NAIST makes available only for research and educational purposes through a request form. It is not redistributed here. It can be obtained either
  1. as the file used in the article, `data/kor_lang8.txt` in <https://github.com/WOOSHIK-M/KoreanGEC> (13,080 pairs, MD5 `22cf8396a331bc10e648fa133d57b287`), or
  2. by requesting the Lang-8 Learner Corpora from NAIST (<https://sites.google.com/site/naistlang8corpora/>) and running the KAGAS scripts in <https://github.com/soyoung97/Standard_Korean_GEC>.

  The line numbers in `korlang8/korlang8_eval.manifest.json` refer to the file with the checksum above.

## Licences

Code: MIT (see `LICENSE`). Model outputs, labels and results: CC BY 4.0. KoLLA v2 text contained in the outputs is distributed under the corpus's GPL-3.0 licence. The vendored scorers are not included and keep their own licences.
