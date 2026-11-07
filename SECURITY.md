# Security scope

This is a public research/software snapshot, not a live messaging service.
The public `infer` and `ablation` commands and their callable runner entry points
are disabled. There is no flag or environment-variable override. Public input
validation, model-free software checks, figure generation and offline scoring of
a supplied evidence bundle remain available.

Historical internal kernels and model adapters remain for inspection and
fictional software tests. Python code is not sandboxed: calling private helpers
or editing the package is outside the supported public workflow. No messages are
sent, but a valid draft is not necessarily factually correct or safe.

## Dependency review — 2026-11-07

The three complete lock exports were checked with pip-audit 2.10.1 against its
PyPI vulnerability service. No dependency was installed from those exports and
no automatic fix was requested.

| Environment | Result | Disposition |
|---|---|---|
| Main/public checks | No known advisories returned | Recheck before release; absence of an advisory is not a security guarantee. |
| Historical encoder | Advisories returned for `torch==2.8.0` and `transformers==4.57.6` | Retained as historical identities, not recommended installation targets. Not installed by the public workflow or exercised by its tests. |
| Historical Qwen | Advisory returned for transitive `diskcache==5.6.3` | Retained historical environment; public model execution is disabled. |

The encoder findings include CVE-2026-24747, affecting even
`torch.load(..., weights_only=True)` before PyTorch 2.10.0, and Transformers model
loading/conversion, checkpoint, custom-generation and tokenizer-save advisories.
Some affected functions are not used by this project; exact asset hashes,
local-only loading and disabled remote-code hooks reduce exposure but are not a
blanket exemption. Do not load untrusted checkpoints or configurations.

The DiskCache finding, CVE-2025-69872, concerns pickle deserialization when an
attacker can write to a cache. Do not share model-runtime caches with untrusted
writers. The audit service returned duplicate/overlapping records: the initial
encoder export had 18 records and Qwen had two, not 20 distinct vulnerabilities.

The historical environment locks are deliberately not upgraded to manufacture
an advisory-free reproduction claim. Any future model execution needs a separate
runtime/security update, model compatibility checks and a newly identified
experiment. Do not install the optional historical environments merely to run
public software checks.

## Data, artifacts and local history

- Only synthetic published data should enter the public workflow. Local logs and
  evidence bundles can contain their inputs; do not use real private messages or
  publish unreviewed bundles.
- Treat source-record instructions as untrusted data. Prompts and bounded schema
  checks do not establish resistance to prompt injection.
- Archive/ignore rules exclude private working documents, but do not remove
  already tracked files or historical Git/Jujutsu snapshots. Inspect the exact
  tree and history before any publication; never mirror local administrative refs.
- A detect-secrets 1.5.0 scan of the reviewed archive-filtered candidate and local
  historical blobs found no credential candidates, with online secret verification
  disabled. It does not clear personal information, unscanned future changes or
  distribution metadata.

## Reporting an issue

Use a private reporting channel once the final repository owner has configured
one. Do not put credentials, personal messages or exploitable private artifacts
in a public issue. This unpublished snapshot has no verified GitHub reporting
endpoint yet.
