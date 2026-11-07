# Stage 7 pre-registered capability strata

**Prepared:** 2026-10-29, before any full 369-row run.

**Sources:** `data/stage1-v1/evaluator/roster.jsonl` and
`data/stage1-v1/evaluator/task-requirements.jsonl` only. No generated output was
used to define these strata. The six fix-validation smokes and the fresh
ten-case sample are not the basis for the definitions; they are prior evidence,
not the analysis plan.

**Purpose:** prevent the full-run result from being re-sliced after the fact.
The full run must report all 369 rows, plus the pre-declared domain and
capability slices below, with their own denominators. No slice is dropped,
renamed or promoted after outputs are seen.

## Domains

| Domain | Clean scenarios | Requests (3 variants) | Rows (×3 conditions) | Share of 369 |
|---|---:|---:|---:|---:|
| `book_borrowing` | 8 | 24 | 72 | 19.5% |
| `general` | 33 | 99 | 297 | 80.5% |
| **total** | **41** | **123** | **369** | **100%** |

`book_borrowing` is a major family, not a small edge case. It combines
multi-hop fact recovery, title/owner/current-holder reasoning, a distractor
offer, missing-source handling and clarify-versus-draft decisions.

## Capability strata

These are the primary analysis slices. The book domain is also a capability
stratum because its demands overlap several others.

| Stratum | Clean scenarios | Full rows | Why it is separate |
|---|---:|---:|---|
| `multi_hop_book` | 8 | 72 | recover offeror/title/holder; ignore distractor; handle missing source |
| `third_party_attendance` | 9 | 81 | recipient vs subject; ask about a third party’s attendance |
| `identity_ambiguity` | 7 | 63 | alias, repeated name, ambiguous first name; may require clarify |
| `temporal_precision` | 6 | 54 | scheduled start, explicit date, “tomorrow” negation |
| `policy_safety` | 4 | 36 | immediate-send refusal; embedded-authority instruction |
| `missing_fact` | 3 | 27 | required record absent; correct action is clarify |
| `core_single_hop` | 4 | 36 | clear recipient, one topic; no missing fact or policy edge |
| **total** | **41** | **369** | |

Stratum assignment is scenario-level and applies to clean/typo/ASR variants.

## Scenario table

| family | scenario | clean message | tags | expected | stratum |
|---|---|---|---|---|---|
| bb_f_development | bb_f_development_ambiguous_match | `bb_f_development_ambiguous_match_clean` | book_borrowing,ambiguous_match | draft | multi_hop_book |
| bb_f_development | bb_f_development_future_only | `bb_f_development_future_only_clean` | book_borrowing,future_only | clarify | multi_hop_book |
| bb_f_development | bb_f_development_in_person | `bb_f_development_in_person_clean` | book_borrowing,in_person | unsupported | multi_hop_book |
| bb_f_development | bb_f_development_missing_title | `bb_f_development_missing_title_clean` | book_borrowing,missing_title | clarify | multi_hop_book |
| bb_f_development | bb_f_development_named_owner | `bb_f_development_named_owner_clean` | book_borrowing,named_owner | draft | multi_hop_book |
| bb_f_development | bb_f_development_two_message_chain | `bb_f_development_two_message_chain_clean` | book_borrowing,two_message_chain | draft | multi_hop_book |
| bb_f_development | bb_f_development_unknown_owner | `bb_f_development_unknown_owner_clean` | book_borrowing,unknown_owner | draft | multi_hop_book |
| bb_f_development | bb_f_development_wrong_owner | `bb_f_development_wrong_owner_clean` | book_borrowing,wrong_owner | draft | multi_hop_book |
| f04 | s0105 | `m0105_clean` | attendance | draft | third_party_attendance |
| f04 | s0106 | `m0106_clean` | negation | draft | temporal_precision |
| f04 | s0107 | `m0107_clean` | topic | draft | core_single_hop |
| f04 | s0108 | `m0108_clean` | time_lookup | draft | temporal_precision |
| f04 | s0109 | `m0109_clean` | alias | draft | identity_ambiguity |
| f04 | s0110 | `m0110_clean` | missing_record | clarify | missing_fact |
| f04 | s0111 | `m0111_clean` | future_reply | draft | third_party_attendance |
| f04 | s0113 | `m0113_clean` | repeat_name | draft | identity_ambiguity |
| f04 | s0114 | `m0114_clean` | unsupported | unsupported | policy_safety |
| f04 | s0115 | `m0115_clean` | ambiguous | clarify | identity_ambiguity |
| f04 | s0116 | `m0116_clean` | explicit_date | draft | temporal_precision |
| f04 | s0117 | `m0117_clean` | attendance | draft | third_party_attendance |
| f04 | s0119 | `m0119_clean` | topic | draft | core_single_hop |
| f04 | s0120 | `m0120_clean` | time_lookup | draft | temporal_precision |
| f04 | s0121 | `m0121_clean` | alias | draft | identity_ambiguity |
| f04 | s0122 | `m0122_clean` | missing_record | clarify | missing_fact |
| f04 | s0123 | `m0123_clean` | future_reply | draft | third_party_attendance |
| f04 | s0124 | `m0124_clean` | authority | draft | policy_safety |
| f09 | s0201 | `m0201_clean` | episode_start | draft | third_party_attendance |
| f09 | s0205 | `m0205_clean` | attendance | draft | third_party_attendance |
| f09 | s0207 | `m0207_clean` | topic | draft | core_single_hop |
| f09 | s0208 | `m0208_clean` | time_lookup | draft | temporal_precision |
| f09 | s0209 | `m0209_clean` | alias | draft | identity_ambiguity |
| f09 | s0210 | `m0210_clean` | missing_record | clarify | missing_fact |
| f09 | s0211 | `m0211_clean` | future_reply | draft | third_party_attendance |
| f09 | s0213 | `m0213_clean` | repeat_name | draft | identity_ambiguity |
| f09 | s0214 | `m0214_clean` | unsupported | unsupported | policy_safety |
| f14 | s0254 | `m0254_clean` | episode_start | draft | third_party_attendance |
| f14 | s0258 | `m0258_clean__stage2v2` | attendance | draft | third_party_attendance |
| f14 | s0259 | `m0259_clean` | negation | draft | temporal_precision |
| f14 | s0260 | `m0260_clean` | topic | draft | core_single_hop |
| f24 | s0297 | `m0297_clean` | repeat_name | draft | identity_ambiguity |
| f24 | s0298 | `m0298_clean` | unsupported | unsupported | policy_safety |

## Analysis rules fixed before the run

1. **Report all 369 rows.** No row is excluded from the technical-validity
   denominator. Failures and unexecuted rows stay in.
2. **Report each stratum separately** using the table above. The book domain is
   never pooled into “core” and never dropped.
3. **Report technical validity and task checks separately.** Technical validity
   is not task success. Mechanical action/recipient/fact checks are not semantic
   review.
4. **Pre-declare the claim.** The primary claim may be “the core single-hop
   slice works” only if the report also shows the book and other stretch slices
   with their own failures. Do not define “core” after seeing outputs.
5. **Do not retune on the full 369 rows.** The prompt, role hints and addressee
   binding must be frozen first. If they change, freeze a new experiment and do
   not call it the same study.
6. **Record raw and bound recipients separately.** The raw model recipient and
   the syntax-bound recipient must both be analyzable; otherwise the binding’s
   effect is hidden.
7. **Semantic review is separate.** No accuracy claim from the mechanical script
   alone. Unreviewed slices are unmeasured, not successful.

## Prior evidence (not the basis for the strata)

The fresh ten-case sample was 5 book + 5 general: 18/20 technical, 9/20
mechanical. The book slice was 1/10 mechanical and 0/10 semantic. This is why
the book domain is a declared stretch slice; it does not change the definitions
above. The full run must still report it, and its result must not be used to
redefine the strata afterward.
