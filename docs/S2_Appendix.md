# S2 Appendix. Prompts, generation settings and data sources

## Prompts

**System prompt (Korean, as used).**

> 당신은 한국어 문법 교정 전문가입니다. 학습자가 쓴 문장에서 문법 오류만 최소한으로 고치세요.
> 규칙:
> 1. 조사, 어미, 맞춤법, 띄어쓰기 오류만 고치세요.
> 2. 문법적으로 맞는 단어를 동의어로 바꾸지 마세요. 예: '같이'를 '함께'로, '출발했습니다'를 '떠났습니다'로 바꾸면 안 됩니다.
> 3. 내용을 추가하거나 삭제하지 마세요.
> 4. 오류가 없으면 원문을 그대로 출력하세요.
> 5. 설명 없이 교정된 문장만 출력하세요.

*English translation.* You are an expert in Korean grammar correction. Correct only the grammatical errors in the sentence written by the learner, minimally. Rules: (1) Correct only errors in particles, endings, spelling and spacing. (2) Do not replace grammatically correct words with synonyms; for example, do not change '같이' to '함께' or '출발했습니다' to '떠났습니다'. (3) Do not add or delete content. (4) If there is no error, output the original sentence unchanged. (5) Output only the corrected sentence, without explanation.

**User turn.**

> 다음 문장의 문법 오류를 교정하세요.
>
> 문장: {sentence}
> 교정:

*English translation.* Correct the grammatical errors in the following sentence. Sentence: {sentence} Correction:

**Four-shot exemplars** (user turn as above, followed by an assistant turn containing the correction):

| Error type | Learner sentence | Correction |
|---|---|---|
| Particle | 저는 학교를 갑니다. | 저는 학교에 갑니다. |
| Ending | 어제 친구를 만나습니다. | 어제 친구를 만났습니다. |
| Spelling | 음식이 마시있어요. | 음식이 맛있어요. |
| Spacing | 저는 한국어를공부합니다. | 저는 한국어를 공부합니다. |

Three of the four corrections end in the formal 합쇼체 (-ㅂ니다); one ends in the polite 해요체 (-어요). The exemplars were written for this study and do not occur in either evaluation corpus.

**Verbal self-rating.** After the greedy correction, the conversation was extended with the model's correction as an assistant turn and the user turn "방금 교정에 대한 확신도를 0에서 100 사이의 숫자로만 답하세요." ("Answer only with a number from 0 to 100 for your confidence in the correction you just made."). In this follow-up conversation the learner sentence appeared in its tokenised form from the .m2 file (punctuation separated by spaces) rather than in the form shown for correction. The first number in the reply was divided by 100; replies without a number were assigned 0.5.

## Generation settings

| Setting | Value |
|---|---|
| Inference engine | vLLM 0.28.0, bfloat16, tensor parallel size 1 |
| Context length / new tokens | 2,048 / 256 |
| Greedy correction | temperature 0, token log-probabilities returned |
| Sampled corrections | 10 per sentence, temperature 0.7, nucleus p = 0.95, seed 1234 |
| Self-rating | greedy, at most 8 new tokens |
| Reasoning mode | disabled (enable_thinking = false) for Qwen3 |
| Hardware (original 14 configurations) | 2 × NVIDIA RTX 5090 (32 GB) |
| Hardware (Qwen3-32B, Kor-Lang8) | NVIDIA RTX 6000D (85 GB); sampling with vLLM's native top-k/top-p implementation (VLLM_USE_FLASHINFER_SAMPLER = 0) |

## Models

| Model | Repository | Revision (commit) |
|---|---|---|
| Midm-2.0 | K-intelligence/Midm-2.0-Base-Instruct | 35479c5fc9a18a5db7cc6dbadcf1db68db7beab0 |
| Kanana-1.5-8B | kakaocorp/kanana-1.5-8b-instruct-2505 | c963a5f4f6496c749f94064a20b33028b0db9f19 |
| A.X-4.0-Light | skt/A.X-4.0-Light | ba21c20ea1b31ded1ec3e2fb432335077dc4be98 |
| EXAONE-3.5-7.8B | LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct | 553ea250b9a5317231459279d5847d6cf955b9aa |
| HyperCLOVAX-1.5B | naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B | 0728a47d632019a8da5f53b663db1c175dc04115 |
| Qwen3-8B | Qwen/Qwen3-8B | b968826d9c46dd6066d109eabc6255188de91218 |
| Qwen3-14B | Qwen/Qwen3-14B | 40c069824f4251a91eefaf281ebe4c544efd3e18 |
| Qwen3-32B | Qwen/Qwen3-32B | 9216db5781bf21249d130ec9da846c4624c16137 |

Weights were obtained from the ModelScope mirrors of these repositories, and for Qwen3-32B from Hugging Face; the SHA-256 checksums of all weight files were verified to be identical between the two hosts.

## Data sources

**KoLLA v2.** Zenodo record 15287129 (release of 26 April 2025), files KoLLA_multi-refs.m2 and FB/FI/HB/HI.csv. The later release (record 15287128, 20 August 2025) contains the same .m2 file but omits the essay texts from the CSV files; the essay texts are needed to link sentences to essays and rubric ratings.

**Kor-Lang8.** The file of 13,080 sentence pairs produced by the KAGAS filtering scripts was taken from a public GitHub repository (WOOSHIK-M/KoreanGEC, file data/kor_lang8.txt, MD5 22cf8396a331bc10e648fa133d57b287); its size matches the count reported in the KAGAS documentation for the output of those scripts. The Lang-8 Learner Corpora themselves are available from NAIST for research and educational purposes through a request form (https://sites.google.com/site/naistlang8corpora/), and the KAGAS scripts (https://github.com/soyoung97/Standard_Korean_GEC) produce the file of sentence pairs from them; the checksum identifies the file to which the released line numbers refer. Rebuilding the evaluation sample from the public copy with the released script reproduces the released manifest and .m2 file exactly. The construction of the evaluation sample (filters, sampling seed 2026, line numbers of the pairs used) is given in the released code and manifest.

**Exclusions.** The 25 KoLLA entries excluded by the audit are listed in the released code (kgc/data/kolla_audit.py) with the reason for each.
