# Data rights and public-domain dedication

The project's rights holders dedicate all copyright and related rights they hold
in the published data to the public domain under
[CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/).
The complete legal text is in [LICENSE-DATA](LICENSE-DATA).

## Scope

This dedication covers the project's rights in:

- All published data under `data/`, including synthetic requests and context,
  entity annotations, task requirements, rosters, exclusions, manifests and
  diagnostic records.
- Published aggregate result data, including the numerical results in the README,
  report, study-details tables and figures.

You may copy, modify, redistribute and use these data for any purpose, including
commercial purposes, without seeking permission or providing attribution under
this dedication. Citation is welcome but is not a condition of reuse. No
noncommercial, share-alike or research-only restriction is imposed by the project.

Original code and documentation/presentation material other than the data covered
above remain under the [MIT License](LICENSE).

## Third-party rights and provenance

CC0 applies only to rights the project's rights holders can waive. It does not
override third-party rights or applicable service terms, guarantee that every
AI-generated element is copyrightable, or independently establish redistribution
clearance for third-party material. Data are provided as-is without warranties,
as set out in CC0.

The retained `data/stage1-v1/provenance/timezones/America_Vancouver.tzif` was copied
from the host's timezone installation; its recorded origin and hash are in
[the timezone identity record](data/stage1-v1/provenance/timezones/identity.json).
On 2026-11-07, compiling IANA's published `tzdata2026e` `northamerica` source with
`zic -b fat` reproduced this retained file byte-for-byte, with SHA-256
`e182e20ec67c46a7dd0aedb3d45fb1395edcb14ff2af2ac694a5776b307bed31`.
[The third-party notice](THIRD-PARTY-NOTICES.md) records the upstream archive hash,
verification command and applicable public-domain statement. This verifies the
retained timezone content without identifying the original host distribution.
The pinned file, identity record and manifest were not changed. Any applicable
third-party terms remain in effect; the project asserts no additional restrictions
on the timezone data.

Model weights are not distributed here and retain their own applicable terms.
This dedication does not publish withheld runs, review judgments, private
provenance or other excluded artifacts, and does not change any frozen data or
manifest hashes.
