# Annotation guideline: unsupported edits (Revision R2.2)

Each item shows a learner sentence (SOURCE), the two human corrections (REF-1, REF-2), and one edit
proposed by a system (EDIT: original span → proposed replacement), together with the full system
correction (SYSTEM) for context. The edit was not matched by either reference under exact span
and string matching. Judge the edit itself, in the context of the sentence, and assign exactly one
label:

- **VALID** — Valid alternative correction. The source contains a genuine error (grammar,
  particle, ending, spelling, spacing, word choice that is wrong in context) at or around this span,
  and the edit repairs it acceptably, but differently from both references.
- **UNNECESSARY** — Unnecessary change. The original span was already acceptable Korean in this
  context; the edit is grammatical and preserves the meaning, but it is a stylistic, lexical
  (synonym), register (speech level / honorific) or orthographic-preference change that a teacher
  would not require. Shown to a learner, it would wrongly suggest that correct language was an
  error.
- **ERROR** — Erroneous change. The edit introduces a grammatical or orthographic error, changes or
  removes the intended meaning, adds content the learner did not express, or otherwise makes the
  sentence worse or no better.
- **GRANULARITY** — Matching artefact. The edit makes the same repair as at least one reference,
  but is recorded with a different span or segmentation (for example, merged with or split from an
  adjacent change, or differing only in spacing that the reference also changes), so it failed exact
  matching even though it is substantively supported.

Decision order: first check GRANULARITY (is the same repair present in a reference?); then decide
whether the original span was erroneous (→ VALID or ERROR depending on the quality of the repair)
or acceptable (→ UNNECESSARY or ERROR depending on whether the edit damages the sentence).

Additionally flag (yes/no) whether the edit changes the speech level or honorific marking
(REGISTER), since this is a Korean-specific source of unnecessary feedback.

Output for each item: `{"id": ..., "label": "VALID|UNNECESSARY|ERROR|GRANULARITY", "register": true|false,
"rationale": "<one sentence in English>"}`.
