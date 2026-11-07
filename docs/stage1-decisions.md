# Stage 1 decisions

Authority: the user authorized Stage 1 on 2026-10-26 and delegated decisions
while away. This record does not authorize later stages or study inference.
The user explicitly resumed Stage 1 on 2026-10-27. The paused snapshot was
hash-verified before work resumed; its incomplete audit is not an accepted verdict.

## Time interpretation

Use the user's Sunday 00:00 inclusive to next Sunday 00:00 exclusive week in the
request's IANA timezone. “Later this week” requires an event start strictly after
the request instant and strictly before the week end. Construct local calendar
boundaries; a daylight-saving week need not contain 168 elapsed hours.

Preserve source timestamps verbatim as offset-bearing instants. Convert the
instant using the verified IANA timezone rules. Vancouver remains at `-07:00`
in November 2026 under B.C.'s permanent daylight-time change, confirmed by the
installed timezone database and the official announcement:
https://news.gov.bc.ca/releases/2026AG0013-000209 .
For example, `2026-11-02T09:00:00-07:00` remains `09:00` in Vancouver that day.
The initial assumption of a November `-08:00` offset was rejected before the
independent annotation pass; it must not create false time-conflict labels.
Bind timezone rules for reproduction rather than relying on an unrecorded host DB.
An explicit request clock that conflicts with the normalized calendar time needs
a targeted clarification. Equivalent draft wording must express the same instant.

## Data and scope

Select existing eligible standalone development requests; preserve their IDs,
text and clean/typo/ASR links. Exclude dependent follow-ups with recorded reasons.
Keep full eligible snapshot context and distractors, rather than selecting
context from the expected answer. Source rows remain byte-preserved where
possible; explicit public projections keep evaluator metadata out of inference.
Keep descriptive source IDs internally for provenance, but omit them from future
model/reviewer prompts when they reveal scenario categories. Use opaque evidence
aliases with a retained reversible mapping at those prompt boundaries.

No new research cases, translations or noise are generated. Missing source
coverage is documented, with fictional software tests used for implementation
boundaries. Keep optional gold diagnostics deferred. This is synthetic,
model-reviewed development material; full human review remains pending.

## Matched conditions and order

Baseline, B and A+B retain the original request and common instructions. B/A+B
share the exact saved extraction and Python result bytes. A+B additionally gets
the extracted entity list. No extra comparison arms or retries are introduced.

Use the six permutations in the study policy. For sorted scenario index `i`,
choose permutation `(i + offset) % 6`, with clean/typo/ASR offsets `0/2/4`.
Persist each order in the roster before execution. This cycles all six orders
within each noise slice, avoiding a fixed association between noise and order.

## Review

New task requirements receive an independent source-only annotation pass before
the annotator sees proposed labels. Preserve disagreements and adjudicate against
the request and eligible evidence. Development review is not outcome scoring.

For later outcome review, freeze independent GPT-6.1 Sol/high and DeepSeek Pro/high
judges, fresh blinded sessions, exact packet bindings and the policy rubric.
Retain original judgments; Astra adjudicates with reasons. Missing, stale or
unresolved review is unmeasured, never success. Pin model identity receipts at
execution; unavailable models do not authorize silent substitution.

### Source-only annotation adjudication (2026-10-27)

The completed independent native `nlp_data_auditor` annotation is retained verbatim
under `data/stage1-v1/evaluator/provenance/annotation-audit/`, with the exact
source-only assignment, input hashes and native DeepSeek Flash/high identity
receipt. It inspected all 123 original variants without proposed requirements or
assistant outputs. This is model-authored development annotation, not human
review, outcome scoring or a replacement for the original source entity labels.
The interrupted earlier audit remains historical and is not credited as a verdict.

The primary adjudicates the flagged interpretations before any generation:

- **`m0105_asr`: retain `draft` to Maya Patel (`c04`).** The omitted name is not
  restored from the clean sibling. The exact ASR still names the family brunch;
  `event_c04_v1` names only the owner and `c04`, and the eligible incoming
  `h_c04_1` explicitly says "For our family brunch" under Maya Patel's authorship.
  These independently available links resolve the intermediary. Alex Morgan is
  the subject of the attendance question, not an automatically selected recipient.
  The auditor's preference for clarification and its stated contextual alternative
  are both preserved. Missing request text alone is not an actual blocker when
  eligible evidence resolves the role.
- **`bb_f_development_ambiguous_match_*`: change the proposed action from
  `clarify` to `draft` to Victor about Elena's *The Paper Observatory*.** All three
  requests identify the book Victor is **reading**. Elena's eligible offer says
  "Victor is reading my copy now" and explicitly supplies this title. Evan's
  competing offer says only "once Victor returns my copy"; possession does not
  establish the same current-reading predicate. The positive explicit match is
  sufficient; do not invent a second confirmed reading relationship and then
  require clarification. Identify Elena as offeror, ask Victor when he will
  finish, and do not invent a finish date. Retain the descriptive source category
  unchanged as provenance, not as an expected-action label. The auditor's
  clarification label and its strong explicit-reading counterinterpretation remain
  in the original annotation. No request, context, roster membership or source
  entity annotation is changed.
- **`bb_f_development_in_person_*`: retain reason-only `unsupported`.** The
  request itself locates the offer in an unrecorded face-to-face conversation,
  "never in our messages". Explain that specific inaccessible source; do not
  falsely deny available message access. Remove the proposed requirement to ask
  the owner a question: that conflicts with the reason-only unsupported contract.
  No fallback draft, selection or clarification payload is required or allowed.
- **`bb_f_development_future_only_*`: retain `clarify` for the missing title or
  an available identifying record.** No eligible offer/title exists in the exact
  snapshot. Neither a descriptive source ID nor an unprovided future record can
  justify telling the owner that a future offer exists or changing this to a
  capability limitation. Absence is assessed against the complete eligible
  snapshot; the missing required fact can be supplied by the owner.
- **Other flagged readings:** retain clarification for the missing winter-seminar
  start information and for both the ambiguous Maya and unspecified question;
  retain drafts for topic-specified open questions, duplicate mentions of the same
  recipient, and "Address the request to MP". These are not immediate-send
  instructions. The explicit 3 pm request agrees with the early eligible 15:00
  Vancouver event and needs no timezone-based clarification.

The evaluation owner must bind these decisions to every exact variant and source
hash, preserve superseded v2 requirements, compare every remaining requirement
against the complete independent annotation, and return any additional discrepancy.
Stage 1 acceptance still requires the installed software gate and independent
code/protocol review. The roster is not yet an execution freeze.

The statistical procedure and all choices affecting interpretation must be frozen
**before generation**, not chosen after inspecting outputs. Stage 4B implements
the paired family-cluster procedure; Stage 5 seals its concrete settings.

### Additional required-content adjudication (2026-10-27)

The evaluation comparison correctly flagged an overly narrow requirement for
`s0109`, `s0124` and `s0209`. The primary inspected all nine exact variants:
"ask about photo album", "with a question about the photo album", and "ask about
seed order" do not mandate a news, progress or opinion question specifically.
Broaden those requirements to a genuine request-consistent topical question to
the supported recipient; news/update/view questions are examples, not exclusive
answers. Do not require an unavailable answer or factual status assertion, and
do not accept an unrelated topic, invented fact or unnecessary clarification.
Keep `draft`, recipient identity and support alternatives unchanged. Preserve the
pending v3 requirements/comparison history and bind this final content decision in
a new version before the independent integrated review. No source request/context,
entity gold, category, roster member, condition order or assistant output changes.

## Software-review decisions (2026-10-27)

The first independent Pro review covered the installed software and fictional
tests, not the still-integrating data/protocol. Its original findings and native
identity receipt are retained in `runs/stage1-resume-20261027/`. It is not an
outcome judge or a study-wide acceptance. The primary authorizes these dispositions:

- Allow multiple source-distinct `name` and `message_text` selected facts so a
  recipient plus a third-party mention, or a two-message loan chain, is expressible.
  Reject exact duplicate facts (source order is immaterial), and retain one fact
  per selected-event field (`title`, `start`, `end`, `location`). Literal source
  equality is still not role or draft-semantic validation; Stage 3 owns that layer.
- Add fictional version-availability-order regressions and make the listed-future
  tests assert the chosen specific fail-closed errors rather than accepting any
  exception. Recheck the revised schema and tests before acceptance.
- The primary read **only** the retained checkpoint's small configuration, not
  weights or predictions: `models/xlmr-final-block-head/block-head-config.json`,
  SHA-256 `1448aafb1814250f8efc49c3891e244342031fc5bc980a09a84f85558fb1886e`.
  Its order is `O`, `B-PERSON`, `I-PERSON`, `B-DATE`, `I-DATE`, `B-EVENT`,
  `I-EVENT`, `B-TOPIC`, `I-TOPIC`. Removing BIO prefixes maps exactly to the four
  existing entity types; no taxonomy expansion or model execution is needed.
  Stage 2 must still validate the complete checkpoint and decoder identities.
- Request construction currently requires a system-recognized IANA zone name.
  Retained TZif bytes, not system offsets, govern frozen civil normalization.
  Document the name-catalog prerequisite rather than adding a second timezone
  loader or silently relaxing validation; a host without the name fails closed.
- Keep the established dependency set for now. Final unused-dependency pruning
  belongs to Stage 8, not an unrelated dependency/CLI rewrite in this checkpoint.

## Stage 1 acceptance and review dispositions

The primary accepted Stage 1 after independently rerunning the complete locked,
offline gate: 265 tests and 22 subtests, the 22-test DATA/installed/relocated gate,
Ruff lint/format, lock/sync and installed public validation. The Pro software
review/follow-up and separate fresh DATA/PROTOCOL review found no material remaining
defects within their declared boundaries. Original review texts/native receipts
and gate logs are retained in `runs/stage1-resume-20261027/`.

Low-severity protocol questions are resolved without rewriting source evidence:
human-review spellings remain historical pending statuses and require explicit
normalization at Stage 4B; non-draft `recipient_contact_ids` is intended eventual
draft metadata, not an output-selection requirement. Unsupported output stays
reason-only. Ten native API pages containing eleven read tools is valid pagination,
not an incomplete input capture. Semantic annotation remains model-authored and
human review pending; this acceptance is neither model quality nor execution freeze.

The user authorizes continuation through Stage 3 on separate `stage-*` branches,
with verified stage checkpoints on `main`; no full-model preparation or Qwen run.

## Retained resources

Continue the existing development budget without transferring the historical
training allowance. The portable opening preserves prior reservations, failed
charges, caps and registered prompt versions, without legacy lock ownership.
Its remaining ceilings are operational accounting, not new execution permission.

Retain existing decoding/token/timeout and memory/disk limits. Stage 1 performs
software/data preparation only. Later preparation must fit the remaining budget
and full prompts without truncation; Qwen still requires Stage 6 authorization
after the restart. Keep failures and partial evidence without repair or retry.
