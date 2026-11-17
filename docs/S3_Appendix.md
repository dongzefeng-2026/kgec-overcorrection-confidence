# S3 Appendix. Rules for edit categories

Each edit, a source span *S* (a sequence of eojeol) and its replacement *C*, is analysed with the Kiwi morphological analyser (kiwipiepy 0.24.1). Morphemes tagged J\* are particles, E\* endings (pre-final, final, connective and nominalising/adnominal endings), S\* symbols, and all others lexical. The first rule that applies assigns the category:

1. **Spacing**: *S* and *C* are identical once spaces are removed.
2. **Punctuation**: *S* and *C* are identical once punctuation tokens are removed.
3. **Word order**: *S* and *C* contain the same eojeol in a different order.
4. **Insertion or deletion**: *S* or *C* is empty. If the inserted or deleted material consists only of particles it is counted as *particle*, only of endings as *ending*; otherwise as *insertion/deletion*.
5. If the lexical morphemes of *S* and *C* are identical: **particle** if only the particles differ, **ending** if only the endings differ, **particle and ending** if both differ, otherwise **spelling** (same morphemes, different written form, e.g. a contraction).
6. If the lexical morphemes differ and the two strings have a similarity of at least 0.75 at the level of jamo letters (difflib ratio over Unicode NFD decomposition): **spelling**.
7. Otherwise, **multi-word rewrite** if *S* or *C* has more than two eojeol, else **lexical choice**.

**Speech level.** The speech level of a sentence is taken from its last final ending (EF): 합쇼체 if the ending ends in 니다, 니까, 시오, 시다 or 디다; 해요체 if it ends in 요; otherwise the plain or intimate style. Sentences without a final ending (fragments) have no speech level. A sentence-level speech-level shift is recorded when source and output both have a speech level and they differ.

**Honorific.** The number of pre-final endings 시/으시 (including fused forms 셨, 셔) is counted in source and output; a difference is recorded as an honorific shift.

**Known limitations.** The analyser is trained on native Korean and can mis-segment learner forms; such forms are usually caught by rule 6 and counted as spelling. Rule 6 also counts form-similar lexical substitutions as spelling, including some that reverse meaning (있었기 → 없었기); the category label therefore describes the form of the change, not its effect. In the review sample, categories were used only to group edits; the judgements of acceptability were made on the edits themselves.
