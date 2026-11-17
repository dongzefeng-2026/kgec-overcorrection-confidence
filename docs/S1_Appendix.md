# S1 Appendix. Formal definitions

## Edits

Let *x* = (x~1~, …, x~n~) be a source sentence tokenised as in the .m2 file and *y* a system output after cleaning and re-tokenisation. A longest-common-subsequence alignment of *x* and *y* yields blocks of equal and unequal tokens. Each maximal unequal block becomes an edit *e* = (*i*, *j*, *c*), where tokens x~i~ … x~j−1~ are replaced by the token string *c* (an insertion has *i* = *j*; a deletion has empty *c*). Adjacent edits (one ending where the next begins) are merged into one, with concatenated replacements. *E*(*x*, *y*) denotes the resulting edit set.

## Support

For each annotator *a*, let *G*~a~ be the set of that annotator's edits in the .m2 file, with deletions written as an empty replacement and whitespace normalised. Let *R*~a~ be the set of all runs of adjacent edits of *G*~a~ (including single edits), each run represented by its span and the concatenation of its replacements. An edit *e* is supported if *e* ∈ *R*~1~ ∪ *R*~2~. Runs are formed within an annotator only; the earlier implementation formed them over the union of both annotators' edits, which produced composite edits that neither annotator made and discarded some single edits.

## Edit-level self-consistency confidence

With *N* sampled outputs *y*^(1)^ … *y*^(N)^ processed in the same way,

conf(*e*) = (1/*N*) Σ~k~ **1**[*e* ∈ *E*(*x*, *y*^(k)^)],  *e* ∈ *E*(*x*, *y*).

With *N* = 10 the confidence takes the eleven values 0, 0.1, …, 1.

The alternative signals are: the sequence probability exp(Σ log p / *T*) of the greedy output with *T* tokens, assigned to every edit of the sentence; the minimum token probability over the tokens of the greedy output that overlap the replacement string of the edit (the whole-output minimum if the replacement is empty or cannot be located; the replacement is located by string search, and because token offsets refer to the raw output, the location is approximate where re-tokenisation inserted spaces before punctuation); the verbal self-rating divided by 100, assigned to every edit of the sentence; and a uniform random number.

## Selective correction and risk–coverage

Let 𝓔 be the pooled set of greedy edits of a configuration over the corpus. For threshold τ, coverage *c*(τ) = |{*e* ∈ 𝓔 : conf(*e*) ≥ τ}| / |𝓔| and risk *r*(τ) is the proportion of unsupported edits among those retained. AURC is the area under the curve of risk against coverage when edits are added in decreasing order of confidence (trapezoidal integration over coverage from 0 to 1). With base unsupported rate ρ = *r*(0), an ideal ordering that places all supported edits first attains

AURC\*(ρ) = ρ + (1 − ρ) ln(1 − ρ),

and the excess is E-AURC = AURC − AURC\*(ρ). Δ~50~ = (ρ − *r*~50~)/ρ, where *r*~50~ is the unsupported rate among the 50 percent of edits with the highest confidence (ties broken by order of appearance).

For deployment (held-out analysis), the threshold for target coverage *c*\* is the smallest τ ∈ {0, 0.1, …, 1} with *c*(τ) ≤ *c*\*.

## Calibration

The expected calibration error uses ten equal-width bins *B*~1~ … *B*~10~ over [0, 1]: ECE = Σ~b~ (|*B*~b~| / |𝓔|) · |mean conf in *B*~b~ − proportion supported in *B*~b~|. The Brier score is the mean of (conf(*e*) − **1**[*e* supported])^2^.

## Within-sentence ranking accuracy

Over all pairs (*e*^+^, *e*^−^) of a supported and an unsupported edit from the same sentence, the accuracy is the mean of **1**[conf(*e*^+^) > conf(*e*^−^)] + ½ **1**[conf(*e*^+^) = conf(*e*^−^)].

## Correction scores

The M² scorer of Dahlmeier and Ng, in the Python 3 implementation distributed with KAGAS, was run with its default settings (maximum of two unchanged words per edit, β = 0.5). With two annotators it chooses for each sentence the annotator that maximises the cumulative F~0.5~ given the sentences already scored. GLEU follows Napoles et al. (2016) with four-gram precision, a random choice of reference per sentence and 500 iterations. Corpus-level bootstrap and permutation statistics re-run these selections on the resampled sentences.

## Resampling

*Essay-cluster bootstrap.* Clusters (essays, and blocks of consecutive unlinked sentences) are drawn with replacement until the original number of clusters is reached; the statistic is recomputed on the concatenated sentences; the 2.5th and 97.5th percentiles of 1,000 replicates form the interval.

*Paired approximate randomization.* For systems *A* and *B*, each cluster's outputs are exchanged between *A* and *B* with probability ½; the statistic difference is recomputed; p = (#{|Δ\*| ≥ |Δ|} + 1)/(*R* + 1) with *R* = 2,000.

*Holm adjustment* within the families of prompting, multilingual-versus-specialised and scale comparisons.

*Proficiency permutation.* Low and high stratum labels are permuted among the rated essays of those strata with fixed stratum sizes; in the joint version a single permutation is applied to all configurations and the statistic is the mean difference across configurations.

*GEE.* Logistic link, exchangeable working correlation within essays, predictors conf (centred), stratum indicator (low versus high) and their product, configuration as a fixed effect; robust standard errors. The continuous version replaces the stratum indicator with the centred grammar rating and uses all rated essays.
