# Study details and result tables

This page supports the [technical report](../REPORT.md). The tables transcribe the retained study analyses and extraction diagnostic; they do not represent a new model run. The full saved responses, review judgments and extraction diagnostic are not included in this public snapshot.

**Historical records:** The linked Stage 1 decisions and Stage 4B evaluation contract are retained as records of earlier project stages. Their original terminology, review-status values and branch references describe the project at those times; they are not instructions for the current public repository. The technical report presents the study and its limitations for readers today.

## Data and eligibility

The `stage1-v1` development export contains 41 standalone synthetic scenarios, each in clean, typo and simulated-ASR form. The 123 requests use 28 contact records, 50 calendar event-version records, 67 history messages and 10 complete snapshots. The export includes distractors, alternate identities and event versions; inference inputs and evaluator requirements are separate. Dependent follow-ups and ineligible or quarantined cases are excluded. Of 324 source scenarios, 41 were selected and 283 excluded; exclusion reasons overlap. The selected request text, entity spans, context bytes and version links were preserved. No new cases, translations or noise were generated in this selection stage.

Eligibility uses the earlier of the request time and snapshot timestamp as the cutoff. Messages must have both their timestamp and availability at or before that cutoff. Contact and event records use the latest available version of each logical ID, with complete snapshot membership and cancellations preserved. A local week runs from Sunday 00:00 inclusive to the following Sunday 00:00 exclusive; “later this week” is strictly after the request instant and before the week end. Source offsets and the retained Vancouver timezone rules determine eligible instants.

| Capability stratum | Scenarios | Rows across three conditions |
|---|---:|---:|
| Core single-hop | 4 | 36 |
| Identity ambiguity | 7 | 63 |
| Third-party attendance | 9 | 81 |
| Temporal precision | 6 | 54 |
| Multi-hop book | 8 | 72 |
| Policy safety | 4 | 36 |
| Missing fact | 3 | 27 |
| **Total** | **41** | **369** |

The strata were defined from the roster and requirements before the full run. The book domain accounts for 8 scenarios and 72 condition rows; it is not folded into the single-hop category. See the [original category definitions](stage7-capability-strata.md), [historical data decisions](stage1-decisions.md), and [public data and evaluator files](../data/stage1-v1/).

## Extraction results

The saved encoder predictions were scored once per request. Exact Unicode start and end offsets and entity type must all match; precision, recall and F1 are micro-aggregated. A type or boundary mismatch contributes a false positive and a false negative. Request-only retrieval does not use the extractor, and the two entity-informed conditions share its predictions.

| Group | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 248 | 115 | 95 | 68.32% | 72.30% | 70.25% |
| PERSON | 171 | 31 | 28 | 84.65% | 85.93% | 85.29% |
| DATE | 9 | 0 | 9 | 100.00% | 50.00% | 66.67% |
| EVENT | 34 | 16 | 23 | 68.00% | 59.65% | 63.55% |
| TOPIC | 34 | 68 | 35 | 33.33% | 49.28% | 39.77% |
| Clean | 93 | 26 | 22 | 78.15% | 80.87% | 79.49% |
| Typo | 88 | 37 | 27 | 70.40% | 76.52% | 73.33% |
| Simulated ASR | 67 | 52 | 46 | 56.30% | 59.29% | 57.76% |

## Semantic outcomes by condition

Each cell has 123 planned requests. Failure includes technical failures; unmeasured review is not counted as failure or success. Percentages are successes divided by planned requests. Condition names match the [technical report](../REPORT.md#14-conditions), whose condition table maps them to the original study identifiers.

| Role hints | Binding | Condition | Success | Failure | Unmeasured | Success / planned |
|---|---|---|---:|---:|---:|---:|
| Off | Off | Request-only retrieval | 23 | 86 | 14 | 18.7% |
| Off | Off | Entity-informed retrieval | 26 | 86 | 11 | 21.1% |
| Off | Off | Entity-informed retrieval + annotations | 38 | 75 | 10 | 30.9% |
| Off | On | Request-only retrieval | 21 | 94 | 8 | 17.1% |
| Off | On | Entity-informed retrieval | 30 | 82 | 11 | 24.4% |
| Off | On | Entity-informed retrieval + annotations | 42 | 75 | 6 | 34.1% |
| On | Off | Request-only retrieval | 47 | 63 | 13 | 38.2% |
| On | Off | Entity-informed retrieval | 49 | 62 | 12 | 39.8% |
| On | Off | Entity-informed retrieval + annotations | 40 | 75 | 8 | 32.5% |
| On | On | Request-only retrieval | 49 | 64 | 10 | 39.8% |
| On | On | Entity-informed retrieval | 49 | 63 | 11 | 39.8% |
| On | On | Entity-informed retrieval + annotations | 44 | 75 | 4 | 35.8% |

The saved condition-level `failure` fields exclude technical failures; the table above adds the corresponding technical failures to show all measured failures consistently with the arm totals. No unmeasured cell is converted to failure.

## Semantic outcomes by capability

The table pools the three conditions. Each entry is successes / planned requests; unmeasured cells remain in the denominator without being called failures.

| Capability | Hints off, binding off | Hints off, binding on | Hints on, binding off | Hints on, binding on |
|---|---:|---:|---:|---:|
| Core single-hop | 19/36 (52.8%) | 20/36 (55.6%) | 32/36 (88.9%) | 34/36 (94.4%) |
| Identity ambiguity | 30/63 (47.6%) | 27/63 (42.9%) | 44/63 (69.8%) | 43/63 (68.3%) |
| Third-party attendance | 15/81 (18.5%) | 17/81 (21.0%) | 41/81 (50.6%) | 40/81 (49.4%) |
| Temporal precision | 17/54 (31.5%) | 21/54 (38.9%) | 17/54 (31.5%) | 21/54 (38.9%) |
| Multi-hop book | 5/72 (6.9%) | 6/72 (8.3%) | 2/72 (2.8%) | 3/72 (4.2%) |
| Policy safety | 1/36 (2.8%) | 2/36 (5.6%) | 0/36 (0.0%) | 1/36 (2.8%) |
| Missing fact | 0/27 (0.0%) | 0/27 (0.0%) | 0/27 (0.0%) | 0/27 (0.0%) |

## Mechanical results by capability

These are the mechanical action/recipient checks with binding on, not semantic task success. Capability names match the semantic-results table above.

| Capability | Role hints on | Role hints off |
|---|---:|---:|
| Core single-hop | 35/36 (97%) | 20/36 (56%) |
| Identity ambiguity | 61/63 (97%) | 52/63 (83%) |
| Third-party attendance | 67/81 (83%) | 31/81 (38%) |
| Temporal precision | 41/54 (76%) | 27/54 (50%) |
| Multi-hop book | 47/72 (65%) | 40/72 (56%) |
| Policy safety | 8/36 (22%) | 25/36 (69%) |
| Missing fact | 0/27 (0%) | 1/27 (4%) |

## Run and review details

The 738 generations completed with `finish_reason=stop`, without retries or length failures. Temperature was 0, seed 42, with 2,048 input and 1,024 output tokens and a 120-second call timeout. All prompts fit; the largest used 2,027 input tokens. The six condition orders were assigned from scenario index and variant offset. Actual usage was 4,024 seconds; the temporary ablation did not charge those generations or time to the canonical ledger, so the ledger remains unreconciled and does not establish a remaining budget.

Each technically valid output and binding received an anonymous review packet containing the exact request, eligible context, frozen requirements and success definition. Two independent model judges decided success/failure; an adjudicator resolved disagreements. A missing or invalid review or failed adjudication is unmeasured. The review used 2,600 judge calls: 2,487 valid, 113 invalid, 128 disagreements and 117 completed adjudications. Across 1,300 reviewable cells, 118 remained unmeasured. The binding-on arms were selected for review based on mechanical results; the binding-off arms were added afterward. An evidence validator was extended to strip typographic quotation marks, recovering three saved hints-on judgments without new calls.

The run was adopted as the study after generation, and its designation and original primary arm were outcome-informed. The originally adopted configuration had both role hints and recipient binding on. Earlier summaries foregrounded hints-off results as a later presentation choice; the current report and figures show all four arms together. There was one generation per request/condition/hint setting, so repeated-run variation was not measured.

The original dataset-source record identifies the agent roles, model names and effort levels listed in the report, but not exact backend revisions. The later selection auditor was DeepSeek Flash/high. The semantic judges were OpenAI Luna/high and DeepSeek Flash/high, with OpenAI Terra/extra-high adjudicating valid disagreements. These are separate review passes, not wholly unrelated model sources: Luna and Terra share an OpenAI family, and DeepSeek also participated in project development. Historical source records, agent instructions and annotation provenance are not included in the public snapshot.

Selected task-requirement interpretations include distinguishing a person being addressed from a third-party subject, requiring clarification when an identifying record is absent, and refusing requests that depend on an unrecorded in-person conversation. Requirements for drafts specify required and forbidden meaning and eligible evidence; clarification and refusal require the correct action and a true, specific reason. The [historical evaluation contract](stage4b-evaluation-contract.md) provides additional protocol details.

## Public evidence and reproduction

The public repository contains code, selected synthetic data, evaluator files, policies and aggregate results in this report. It excludes model weights, full saved runs, semantic-review judgments and historical administrative and annotation-provenance records. As a result, the published software checks can be run, but the full study results cannot be independently recomputed from this snapshot alone. The original data manifest remains unchanged; references in historical documents to omitted provenance are not claims that those files are public. The extraction-quality diagnostic is also retained separately and is not included in the snapshot.

Published data, annotations and aggregate result data are dedicated under CC0 1.0 Universal, to the extent the project's rights holders hold rights in them. See [data rights and scope](../DATA-LICENSE.md) for the dedication and third-party provenance notes. This does not change which artifacts are available or the study's validation limitations.

The completed study ID is `stage7-flat-decision-study-20261029`, and the run ID is `stage7-ablation-20261029`. The locally retained study authority has SHA-256 `3e2928575ecb886afc889f6b803626afa76f51cee4c970a21fc5f753ee01dc7c`; the freeze record has SHA-256 `8dd1215d47125c36f3d7db12788cc9b1c3c29a89112d5058182af10e791e1557`. These identifiers describe withheld evidence, not a verification that can be independently performed from the public snapshot.
