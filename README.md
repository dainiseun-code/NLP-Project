# Does entity extraction help a local messaging assistant?

For a messaging assistant to prepare a useful response, it needs to identify the people and topics in a request, retrieve the relevant records and use them correctly. This project investigates whether a trained XLM-R entity extractor helps a small local language model complete these tasks, compared with retrieving evidence from the request alone.

Given a request and read-only contacts, messages and calendar records, the assistant returns a structured draft, a clarification question or an explanation that the request is unsupported. For example, it can prepare a question to Maya about the photo album, or recover a book title from earlier messages before drafting a question to its current holder. It is **draft-only and does not send messages**.

The [technical report](REPORT.md) explains the design, example requests, agent workflow and results. The project covers entity extraction, retrieval, local inference, structured output validation and blinded model-based evaluation.

## System and experiment

The assistant uses XLM-R base with a fine-tuned final encoder block and classification head, a Python retrieval and validation pipeline, and a local `Qwen3.5-2B-OptiQ-4bit` model. The extractor identifies people (`PERSON`), dates (`DATE`), events (`EVENT`) and topics (`TOPIC`). The comparison retains the original request in all three conditions:

| Condition | Retrieval | Model input |
|---|---|---|
| **Request-only retrieval** (originally baseline) | Search using the original request. | Request and retrieved evidence. |
| **Entity-informed retrieval** (originally B) | Use extracted entities to inform retrieval. | Request and retrieved evidence. |
| **Entity-informed retrieval + annotations** (originally A+B) | Use the same extraction and retrieval artifacts as entity-informed retrieval. | The same input plus the extracted entities. |

Each condition is tested with two further factors turned off and on:

- **Role hints:** syntax-derived labels shown to the model to distinguish the person being addressed from someone discussed in the request. In “Ask Maya if Alex is coming”, Maya is addressed and Alex is the subject.
- **Recipient binding:** a Python rule applied after generation that can replace the model-selected recipient with the single unambiguously addressed displayed contact. It does not write or repair the message.

Together these give four hint/binding **arms**, each evaluated under all three conditions. Agents and sub-agents also supported data preparation, implementation and semantic review; they were part of the development workflow, not the messaging assistant itself.

## Data and run counts

The study uses **41 synthetic development scenarios**, each with three versions of the same underlying request:

- **Clean:** text without deliberately introduced errors.
- **Typo:** text containing typing errors.
- **Simulated ASR (automatic speech recognition):** text altered to resemble transcription errors, including substitutions and omissions; no audio was transcribed.

Noisy versions can lose useful information and are processed independently of their clean counterparts. The resulting **123 requests** are evaluated using model-based task annotation and semantic review.

| Stage | Calculation | Total |
|---|---|---:|
| Requests | 41 scenarios × 3 text variants | 123 |
| Generations | 123 requests × 3 conditions × 2 hint settings | 738 |
| Evaluation cells | 738 saved responses × 2 binding settings | 1,476 |

Binding off and on reuse the same generated response, rather than require separate model calls.

## Selected results

Semantic success requires the correct action, recipient and meaning, not just a valid response structure. The full comparison across the four arms was:

| Role hints | Binding | Request-only retrieval | Entity-informed retrieval | Entity-informed retrieval + annotations |
|---|---|---:|---:|---:|
| Off | Off | 23/123 (18.7%) | 26/123 (21.1%) | **38/123 (30.9%)** |
| Off | On | 21/123 (17.1%) | 30/123 (24.4%) | **42/123 (34.1%)** |
| On | Off | 47/123 (38.2%) | **49/123 (39.8%)** | 40/123 (32.5%) |
| On | On | **49/123 (39.8%)** | **49/123 (39.8%)** | 44/123 (35.8%) |

Values are successes / planned requests. Missing or invalid semantic reviews remain in the denominator without being counted as confirmed failures; the report gives [review coverage and detailed counts](REPORT.md#21-metrics-and-review).

Entity-informed retrieval alone matched or scored above request-only retrieval in every arm. Adding annotations produced the highest success rates **without role hints**, under both binding settings. With hints present, however, annotations scored below both alternatives. The benefit of supplying entities therefore depended on the other guidance given to the model, rather than simply on recipient binding.

The extractor achieved **70.25% exact-span-and-type micro F1**, with person names strongest and topics weakest. The assistant struggled particularly with multi-hop book reasoning, missing facts and policy safety. Role hints helped ordinary drafting tasks but also encouraged drafts when refusal was required. These weaknesses keep the system a research prototype rather than a tool for real messaging. See the [results and figures](REPORT.md#2-evaluation-and-results) for the task-category breakdowns.

Available compute limited model choice and the amount of inference possible. The quantized 2B model's limited capacity likely contributed to the low overall success rates, although its contribution was not isolated. All conditions and arms shared these resource constraints, allowing a like-for-like comparison within this system. A stronger model could improve overall performance, but whether it would preserve the same ordering of conditions remains to be tested.

## Install and run software checks

The historical `infer` and `ablation` commands are disabled in this public snapshot. Public validation, software checks, figure generation and offline `reproduce` scoring of a supplied evidence bundle remain available.

With Python 3.11 and [uv](https://docs.astral.sh/uv/) installed:

```sh
uv sync --locked
uv run --locked messaging-nlp --help
uv run --locked messaging-nlp validate-public data/stage1-v1/public --owner-id owner_001
uv run --locked ruff check src/messaging_nlp tests
uv run --locked ruff format --check src/messaging_nlp tests
uv run --locked pytest -q
```

The public suite checks published data against the applicable original manifest hashes. It skips **262 historical tests** that depend on private provenance and administrative fixtures; their IDs are listed in `tests/private-evidence-tests.txt`. These checks test the software and published data, separately from model quality. Regenerate the figures from the published aggregate values without model inference using:

```sh
uv run --locked python docs/figures/render.py
```

## Evidence and availability

The repository contains the code, selected synthetic data, evaluation policies, aggregate tables and report. Model weights, full run outputs, semantic-review judgments, the extraction diagnostic and historical provenance records are not included. A clone can run the public software checks and regenerate the figures, but cannot independently recompute the study results or run the full model workflow without those additional artifacts. See [study details](docs/study-details.md#public-evidence-and-reproduction) for the available evidence and historical records.

## Explore the project

- [`REPORT.md`](REPORT.md) — experiment, results, agent workflow and discussion.
- [`docs/study-details.md`](docs/study-details.md) — data selection, detailed tables and review/evidence notes.
- [`src/messaging_nlp/encoder.py`](src/messaging_nlp/encoder.py) and [`encoder_decode.py`](src/messaging_nlp/encoder_decode.py) — XLM-R adapter and span decoding.
- [`src/messaging_nlp/retrieval.py`](src/messaging_nlp/retrieval.py) and [`prompts.py`](src/messaging_nlp/prompts.py) — evidence retrieval and prompts.
- [`src/messaging_nlp/drafting.py`](src/messaging_nlp/drafting.py), [`evaluation.py`](src/messaging_nlp/evaluation.py) and [`reviews.py`](src/messaging_nlp/reviews.py) — decision validation and scoring.
- [`data/stage1-v1/`](data/stage1-v1/) — synthetic public inputs and separate evaluator data.
- [`tests/`](tests/) — model-free behavioral and public-data integrity checks.

## License

The project's original code and documentation are available under the [MIT License](LICENSE).

All published data, annotations and aggregate result data are dedicated to the public domain under [CC0 1.0 Universal](LICENSE-DATA), to the extent the project's rights holders hold rights in them. Reuse, modification, redistribution and commercial use require no permission or attribution under this dedication. See [data rights and scope](DATA-LICENSE.md).

CC0 does not override third-party rights or applicable service terms. The data-rights notice records the retained timezone file's provenance and the limits of its verification. Model weights are not included and retain their own applicable terms.
