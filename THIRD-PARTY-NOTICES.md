# Third-party notices

## Retained IANA timezone data

File: `data/stage1-v1/provenance/timezones/America_Vancouver.tzif`.

The file was originally copied from the host timezone installation. On
2026-11-07 its complete bytes were independently reproduced from IANA's
[`tzdata2026e.tar.gz`](https://data.iana.org/time-zones/releases/tzdata2026e.tar.gz).

- Upstream archive SHA-256:
  `b26882805f26aac59d5b222978e6580484b834ccdc98be89df2f05a6dc53a652`.
- Archive `version`: `2026e`.
- Source: `northamerica`; compile with
  `zic -b fat -d <temporary-output> <extracted-source>/northamerica`.
- Result `America/Vancouver` SHA-256:
  `e182e20ec67c46a7dd0aedb3d45fb1395edcb14ff2af2ac694a5776b307bed31`,
  equal to the retained file. The host compiler was `/usr/sbin/zic`.

The archive's complete `LICENSE` text is:

> Unless specified below, all files in the tz code and data (including
> this LICENSE file) are in the public domain.
>
> If the files date.c, newstrftime.3, and strftime.c are present, they
> contain material derived from BSD and use the BSD 3-clause license.

Those three exception files are not distributed by this project. The retained
timezone data comes from `northamerica`; no timezone compiler code is included.
This verification does not replace or modify the frozen dataset, its manifest
or the identity record's original host-copy provenance.

## Model runtimes and weights

Dependency packages retain their respective licenses. The encoder and Qwen
environment definitions identify historical runtimes, not bundled third-party
packages. No model weights are distributed. See [security scope](SECURITY.md)
before considering any historical runtime installation.
