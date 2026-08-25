# Results Integrity Policy

This repository is intended to support externally reviewable results.

## Allowed

- Machine-generated score files.
- Scripts that derive tables and figures from raw run records.
- Manual failure labels when the supporting run/log ID is recorded.
- Corrections to analysis code followed by full regeneration of derived outputs.

## Not allowed

- Typing benchmark scores into a publication table from memory.
- Replacing a failed run with a successful rerun under the same run ID.
- Deleting inconvenient failures.
- Changing the selected task subset after seeing model performance.
- Mixing runs from different model snapshots/settings under one condition label.
- Reporting partial batches as if they were complete.

## Provenance

Each evaluated prediction should have:

- task ID
- model/provider
- condition
- repeat number
- score
- prediction SHA-256
- run status
- timestamp
- path to raw artifacts/logs when available

The final README should link every headline number to a generated file in `results/derived/`.
