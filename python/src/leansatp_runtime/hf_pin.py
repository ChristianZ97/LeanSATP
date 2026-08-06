"""Single pin for every artifact fetched from the satp-policy-v4.27 HF repo.

Source (infer.py), checkpoint, and retrieval assets are all resolved at this
one immutable commit, so the executable bundle can never be version-skewed:
a future HF push changes nothing here until the pin is bumped deliberately.

Bumping: update ``_DEFAULT_REVISION`` (or set ``SATP_HF_REVISION``), rerun
``./setup.sh`` (or ``--download-only``), restart the service, then re-run the
Step-A parity gate (244/244 byte-identical decodes against the repo's own
``infer.py``) and the ``satp?`` e2e sweep before trusting any number. The
gate targets are era-specific — do not carry a previous era's numbers over.

2026-08-06, v2 → v4.27: the eras differ in decode cardinality, not just in
weights. ``LEMMA_HOST_POOL`` went 20 → 19 entries and ``N_LEMMA_DECISION``
172 rather than 181. The dropped host, ``positivity [{L}]``, was the *last*
entry, so v4.27's pool is a strict prefix of v2's and no surviving index
moves — the era difference is pure cardinality. That is what makes the
cross-era nets work: ``build_policy_v2`` compares the checkpoint's own
``tactic_pool`` metadata against these constants, and even if that check is
skipped (a checkpoint carrying no metadata — see the ``got is None`` branch
in ``policy_v2.py``), the 181-vs-172 head width still raises inside
``load_state_dict``. Both shipped checkpoints do carry the metadata.

Cardinality does not *identify* an era, though: ``satp-policy-v2-alphaproof``
is also 19/172, so it passes every check here. The pin, not the guard, is
what selects a training run.
"""

from __future__ import annotations

import os
import sys

HF_REPO = "ChristianZ97/satp-policy-v4.27"
_DEFAULT_REVISION = "5a4d2f1bd6731dd5b68ad95264094284206e609f"

REVISION = os.environ.get("SATP_HF_REVISION", _DEFAULT_REVISION).strip()
if not REVISION:
    raise RuntimeError(
        "SATP_HF_REVISION is set but empty — refusing to fall back to a "
        "floating revision; set a commit hash, or unset the variable to use "
        "the pinned default"
    )


def revision_for(repo_id: str) -> str | None:
    """Revision to pass to ``hf_hub_download`` for ``repo_id``.

    ``REVISION`` pins this module's repo only; any other repo has no pin here
    and would resolve to a floating branch head. That is a real hazard whose
    meaning *moves when HF_REPO moves*: before the 2026-08-06 bump a
    ``satp-policy-v2`` source was the pinned one, and after it the same string
    silently became unpinned. Say so out loud instead of returning a bare
    ``None`` from an inline conditional at each call site.
    """
    if repo_id == HF_REPO:
        return REVISION
    # Bare print rather than service.log_server: a lazy import would work (no
    # cycle — service.py has no module-level leansatp_runtime import), but
    # pulling in the whole service module for one log line is not worth it.
    # stderr, not stdout: every other log in this process goes to stderr, and
    # `--serve --download-only` writes machine-read JSON to stdout — a warning
    # mixed into that stream breaks json.loads on the caller's side.
    print(
        f"WARNING  [LeanSATP] {repo_id} is not the pinned repo ({HF_REPO}); "
        "fetching its floating branch head. Artifacts from an unpinned repo "
        "are not reproducible — pass an explicit revision or bump hf_pin.py.",
        file=sys.stderr,
        flush=True,
    )
    return None
