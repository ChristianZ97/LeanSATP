# miniF2F-test proofs — satp-policy-v2 (92/244)

Machine-generated proofs for the 92 miniF2F test problems solved by
[`ChristianZ97/satp-policy-v2`](https://huggingface.co/ChristianZ97/satp-policy-v2)
(0.2B byt5 aesop-configuration policy, greedy decode) on
[`ChristianZ97/minif2f-satp`](https://huggingface.co/datasets/ChristianZ97/minif2f-satp)
`split=test`. 92/244 = 37.70% matches the model card; the solved set is
identical across four independent harnesses (torch decode + kimina server,
torch decode + plain `lake`, the `satp?` Lean tactic end-to-end, and
`satp?` with `maxHeartbeats 0`).

Two views of the same 92 theorems:

- **`config/`** — the policy-emitted `aesop (config := …)` invocation,
  verbatim as decoded. This is the model's actual output.
- **`minimal/`** — distilled minimal form: the tactic script aesop itself
  reports via `aesop? … "Try this:"` for the config's successful search
  (mean 1.5 tactics per proof; 61/92 are a single tactic; −94% bytes
  vs the config view).

Every file in both views:

- imports only `Mathlib` — no LeanSATP, no inference service required;
- compiles on stock Mathlib v4.26.0 (`2df2f0150c`): `lake env lean` exit 0,
  no `sorry`;
- depends only on the standard axioms (`propext`, `Classical.choice`,
  `Quot.sound`), checked per file via `#print axioms`.

File names are theorem names. `manifest.jsonl` maps each theorem to its
dataset row index, uuid, distilled tactic sequence, and axiom set.

Reproduce from scratch: `reproduce.py` in the model's Hugging Face repo
(kimina or plain-lake backend), or run the `satp` / `satp?` tactic from this
repository on the dataset statements.
