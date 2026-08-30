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

2026-08-12, 5a4d2f1b → 8ed997e1: same era, same executable bundle. ``infer.py``
and both retrieval assets (``cache/mathlib4_premises.txt``,
``cache/premise_embeddings.npy``) are byte-identical across the two revisions —
sha256 compared at each — so the decode surface, and the Step-A parity gate that
targets it, are unchanged. What moves is *which file* ``best_checkpoint.pt`` is:
upstream flattened the checkpoint names and re-pointed the default from the
10-epoch validation-best (``ckpt_10ep_val104_test99.pt``, sha256 ``867372b6…``)
to the 3-epoch test-best (``ckpt_3ep_val103_test101.pt``, sha256 ``1c03fdd5…``).
That new default is the checkpoint already in service and the one every v4.27
band number was produced with, so this bump makes the pin agree with the runs
rather than changing them. The seed-named files the old revision shipped
(``ckpt_8964.pt``, ``ckpt_1827.pt``) no longer exist upstream.
"""

from __future__ import annotations

import os
import sys

HF_REPO = "ChristianZ97/satp-policy-v4.27"
_DEFAULT_REVISION = "8ed997e1526e58106a960966b06721dbf313c6a4"

# ``lfs.sha256`` that HfApi().repo_info(HF_REPO, revision=_DEFAULT_REVISION,
# files_metadata=True) reports for best_checkpoint.pt. Lives beside the
# revision because it is the same fact: a revision names weights, and this is
# how you check you have them. Keeping it here rather than in the test is not
# tidiness — on 2026-08-12 the revision was bumped and the test's own copy of
# the digest was not, so the guard went on approving the superseded file.
CHECKPOINT_SHA256 = "1c03fdd59943318e439f942e0f1d03f7a428ab54cd8ea6f16ac8d71a0e80f262"

REVISION = os.environ.get("SATP_HF_REVISION", _DEFAULT_REVISION).strip()
if not REVISION:
    raise RuntimeError(
        "SATP_HF_REVISION is set but empty — refusing to fall back to a "
        "floating revision; set a commit hash, or unset the variable to use "
        "the pinned default"
    )


def is_default_revision() -> bool:
    """Whether ``REVISION`` is still the pin, not an ``SATP_HF_REVISION`` override.

    ``CHECKPOINT_SHA256`` is the digest *of the pinned revision*, so any code
    enforcing it has to know whether the revision in play is that one. An
    override is a legitimate thing to do (bisecting a bad bump, checking an
    older artifact); it just means nothing here can vouch for what lands.
    """
    return REVISION == _DEFAULT_REVISION


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
