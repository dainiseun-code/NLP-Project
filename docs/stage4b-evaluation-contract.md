# Stage 4B evaluation and offline reproduction contract

Authorization: user's 2026-10-28 instruction to review 4A and continue through 5.
This stage uses fictional outputs/judgments only. No study quality calculation,
review call, model execution, research-data change, or historical score import.
Stage 4A source hashes match its acceptance record; independent read-only review
found no material defect. Its acceptance remains software evidence, not model quality.

## Ownership and interfaces

The orchestrator owns this contract, review binding and reproduction integration.
`evaluation.py` contains pure kernels; its one writer may add fictional kernel tests.
`reviews.py` owns blinded packets and strict bound judgments. `reproduction.py`
consumes the actual `load_row_receipt` API, never an independent run-row schema.
CLI adds one offline `reproduce` entry point. No new dependencies are necessary.

Pure kernel interfaces (JSON-compatible returns):

- `normalize_human_review(value: str) -> str`: only `pending` and
  `pending_not_human_annotation` and roster's `pending_not_human_annotated`
  normalize to `pending`; unknown values fail closed. The third spelling was
  verified directly in the retained roster; this normalizes metadata, not data.
- `score_extraction(text: str, gold: list[Entity], result: ExtractionResult | None)`:
  exact Unicode code-point (start,end,type), validate gold/predicted text slices;
  return status, tp/fp/fn and precision/recall/f1. Missing/failed prediction is
  unmeasured with null counts/rates, NOT an empty extraction. Valid both-empty has
  tp=fp=fn=0 and rates=1. Baseline is not scored; B/A+B counted once per request.
- `score_structure(row: RunRow | None, requirement: dict)`: separate action and
  draft-recipient equality, selected-event ID and explicit event-start instant
  checks; false clarification fails action. Missing/invalid output is unmeasured,
  not a correct action. Non-draft recipient is `not_applicable` irrespective of
  eventual-draft metadata. Missing expected draft selection fails applicable checks.
  Only explicitly representable normalized facts are deterministic; book roles,
  meaning, authorship and true blocker/reason require semantic review.
- `evidence_coverage(candidate_record_ids: list[str] | None, alternatives: list[list[str]])`:
  complete iff any sufficient source set is a subset; empty alternative is valid;
  missing candidate result is unmeasured; report best fraction separately.
  `[[]]` is a sufficient empty source set; `[]` declares no admissible sufficient
  alternative and is incomplete/0.0, not proof that no support is needed.
- `paired_summary(rows: list[dict], *, repetitions=2000, seed=42)`: rows contain
  scenario_id, family_id, variant_type, condition and success (bool or None).
  Reject duplicate scenario/variant/condition and inconsistent family linkage.
  Report each condition's planned, measured, successful and unmeasured counts,
  success/planned as **credited success coverage**, measured-only descriptive rate
  separately (null if none), never label unexecuted work a measured zero.
  Condition differences use only same-scenario/same-variant resolved pairs; retain
  incomplete-pair counts.
  All scenario/variant pairs in the supplied roster count as incomplete when one
  or both contrast arms are missing (including pairs seen only in the third arm).
  Cluster bootstrap by family, uniform family sampling with replacement; preserve
  every paired scenario/variant row within the sampled
  family; shared draws for all contrasts; 2000 repetitions, Python Random seed42,
  percentile95% interval (linear interpolation). No complete pairs => null estimate
  and interval; fewer than two represented families => interval null. Five source
  families limit uncertainty interpretation, not an independence guarantee.
  Draw from all planned families jointly across contrasts; a contrast with no
  resolved pairs in a draw has an undefined mean, so omit that draw for that
  contrast and disclose `bootstrap_samples`. This conditional complete-pair
  analysis does not impute unmeasured results or assume missingness is random.

## Semantic review protocol

Freeze seven dimensions from study policy verbatim. Every completed technically
valid row needs both frozen judges on all variants/categories/conditions. Technical
failures remain failures and require no fabricated reviewable output. Missing
receipt, missing parsed output or pending persistence remains explicit/unmeasured.
Action-conditional recipient/blocker guidance belongs in the rubric. Ratings are
pass/fail/unresolved, never syntax-as-success. Each verdict must cite an exact output
quote, or identify an omission, with rationale and eligible opaque source IDs.

Judges remain GPT-6.1 Sol/high and DeepSeek V4 Pro/high; unavailable identities fail
closed, no substitution. Development reviewer agents are NOT outcome judges.
No judge is invoked in 4B/5. At Stage7 use fresh isolated sessions with only one
anonymous packet, not repository/context/history, sibling output or other judgment.
Retain actual native identity receipt bytes and validate declared provider/model/
effort against frozen judge policy. Bind judgments to exact packet, request,
eligible full context, requirements, rubric, raw and resolved output, and identity
receipt bytes. No hash-only unavailable evidence. Keep originals immutable.
For each disagreement the orchestrator may add a separate evidence-bound written
adjudication; never replace originals. Missing/stale/unresolved dimensions cannot
pass. A fail dimension fails overall; otherwise unresolved stays unmeasured.

Judge-visible bytes omit condition/model/category/scenario/family/request/snapshot
IDs, provenance and annotation metadata. Keep exact original text and request clock,
full eligible context (not just retrieved candidates), source-bound requirements,
rubric and one output. Use condition-neutral opaque evidence aliases with a separate
reversible private binding manifest. Packet IDs are random and carry no condition.
`judge_assignment` supplies only that anonymous packet, opaque binding hashes and
the response schema to a judge; it never exposes private paths/manifests. Binding
hashes must be supplied because judges cannot inspect private source files.
Exact raw output stays retained privately; the packet uses the anonymous output
whose structured IDs are aliased without changing prose. Never show entity labels
or condition-specific preprocessing to the judge.

## Offline reproduction and acceptance

One manifest binds all retained source/config/data/run/review bytes and expected
row roster, including pending-row index and missing receipts. Resolve run artifact
paths under the retained run root, and frozen inputs under its `frozen/` subtree.
Validate actual plan/order/request and every supplied artifact hash; duplicate or
foreign receipts fail. Missing planned work remains visible. Changes to artifacts,
review bindings, expectations or policy fail rather than silently rescore.
Installed package source must match the retained source inventory before scoring;
otherwise restore/install those actual frozen bytes. An unavailable commit ID is
not an executable dependency and a changed scorer is not the same reproduction.
Retain necessary bytes (including evaluator inputs and rubric) in a portable bundle;
no old git show, archive dependency, model weights, reviewer calls or fresh retrieval.
The offline path never imports optional model runtimes. Costs report per-condition
Qwen response elapsed/charges and unique shared preparation separately; do not
allocate one preprocessing charge twice, sum medians, or infer live E2E latency.

Independent fictional tests cover exact/noisy spans, false clarifications, wrong
limitations, contradictory events/times, condition-conditional roles, missing/zero
executions, partial persistence, stale reviews, artifact mutation, paired variants,
family draws, denominators, cost attribution and a retained fake-run offline replay.
Run the full offline gate and obtain independent review after integration. Only the
orchestrator interprets gates and accepts the stage. Stage5 must seal this protocol,
rubric and settings before any Qwen generation. Actual report calculations stay in7
after asking whether authoritative measurements already exist.

### Confirmed offline integration API

`build_bundle(run_root, destination, *, requirements_path, roster_path, gold_path,
policy_path, review_records=None, adjudications=None, preparation_costs_path=None)`
copies actual retained run bytes to `destination/run/`, evaluator inputs to explicit
`destination/evaluator/` names and seals `manifest.json`. The destination must be
fresh, disjoint from the source, never follow symlinks, and never copy live model
directories. run_root is the durable runner output (contains plan.json, frozen/,
rows/, status.json), NOT the project root. Bundles may retain frozen non-weight
model metadata/tokenizer/template bytes under
`frozen/models/`; these are offline input identities, not a live model dependency.
The builder refuses weight payloads and must never copy the project's live model
directory. Original native identity and judgment artifacts are retained unchanged.
Review binding/judgment files are already inside run_root; review_records maps
row_id to lists of JudgmentRecord
dictionaries with run-root-relative references; adjudications maps row_id to
ArtifactRef. Persist these maps in the manifest. Do not invoke packet creation.

The manifest is a strict versioned contract: `schema_version`, `files` (ArtifactRef
inventory of every copied file with bundle-relative paths), `plan` (ArtifactRef),
`run_root` (`run`), `requirements`, `roster`, `gold`, `policy` (ArtifactRefs),
`review_records`, `adjudications`, `preparation_costs` (ArtifactRef or null).
Retain all bytes, not just refs. Hash-check the entire inventory before reading
scores, require indexed refs to agree with inventory and reject escaping paths.
`reproduce(manifest_path) -> dict` is the pure offline consumer. CLI prints it;
it writes nothing into the immutable bundle and calls no model/reviewer.

For every planned row, use `run/rows/{ordinal:06d}/receipt.json` if present through
`load_row_receipt`. Validate ordinal/row_id/condition/request against the top plan.
Missing receipt is `pending_persistence` if listed in status.json `pending_rows`,
otherwise `missing_receipt`; validate PendingRowRecord identities/refs when present.
Never promote pending inline RunRow bytes to success. Validate all original frozen
files against run/frozen/ paths. Reconcile roster order and requirement request
hash/snapshot/scenario/family/variant against each actual planned request. Model
identity is retained metadata only; do not use load_and_validate (it requires
weights and installed-source equality, unsuitable for offline scoring).

Successful rows receive structural scores and full semantic resolution only when
bound review inputs match the exact requirement, policy, row raw output, resolved
envelope and full eligible context. No binding means semantic unmeasured. Missing
judges remain unmeasured; supplied stale judgments raise. Attempted technical
failures have success=False for operational task success; unexecuted/missing/
pending rows and unresolved review have success=None. All remain planned counts.
Overall success=True requires technical ok, structural applicable dimensions pass,
and both/adjudicated semantic pass. Preserve separate dimensions/reasons.

Extraction diagnostics consume each unique assisted extraction from frozen plan
preparation bytes, not only completed Qwen rows; failed/missing extraction is
unmeasured, not empty. Count B/A+B once per request; never baseline. Coverage uses
the same frozen preprocessing candidates, independent of Qwen completion. No
quality values over real selected data may be computed during4B/5: tests and
retained fake replay only. Preparation-cost bytes are passed through separately;
Qwen costs are summed from row receipts by condition, startup/overhead ledger
charges disclosed separately when retained, never recombined into live latency.

## Primary acceptance — 2026-10-28

Accepted as software/fictional evidence only. The actual4A schema integration,
blinded packet/judgment/adjudication bindings, frozen source/expectation checks,
missing/pending denominator accounting and paired statistics satisfy this contract.
Primary corrected the workers' omitted aggregate/status integration and a visible-
but-uncommitted receipt path before acceptance. Independent review plus two
follow-ups completed with no remaining material findings. Coverage limits remain:
adjudication is unit-tested plus remapping-tested, not a separate full replay;
the full fake replay uses fabricated pass/pass judgments and proves no model quality.

Final primary gates:1260tests+22subtests and812fictional-only tests with zero forbidden
model/archive/runtime (and fictional-gate data) attempts; DATA22; Ruff; all three
offline locks/main sync; installed CLI; retained fake3-row runner/review/offline
replay and relocation. Evidence is under `runs/stage4b-evaluation-20261028/`, including
`primary-final-*.log`, original gate collection and `retained-fake-final/`.
No real encoder/tokenizer, Qwen, semantic judge, study score, budget reset, research
edit, archive deletion or publication occurred. Stage5 follows on its own branch.
