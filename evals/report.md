# Guardrail evaluation (rules only (offline))

Items: 10 · accuracy **90%** · precision **100%** · recall **83%**

Confusion (positive = should be blocked): TP 5 · FP 0 · FN 1 · TN 4

| id | expected pass | predicted pass | issues | note |
|---|---|---|---|---|
| g1 | True | True | - | clean, grounded |
| g2 | True | True | - | clean paraphrase |
| g3 | True | True | - | clean, other topic |
| g4 | True | True | - | numbers that ARE in the facts |
| g5 | False | False | numbers not found in the facts (possible hallucination): ['500'] | invented number |
| g6 | False | False | numbers not found in the facts (possible hallucination): ['1', '100'] | invented benefit + numbers |
| g7 | False | False | must contain the registration form link | missing form link |
| g8 | False | False | hashtags must include #FPTSoftware | missing #FPTSoftware |
| g9 | False | False | too long: 1462 > 1200 characters | too long |
| g10 | False | True | - | unsupported claim WITHOUT numbers: rules alone miss this, needs the LLM judge |
