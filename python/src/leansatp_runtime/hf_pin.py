"""Single pin for every artifact fetched from the satp-policy-v2-alphaproof HF repo.

Source (infer.py), checkpoint, and retrieval assets are all resolved at this
one immutable commit, so the executable bundle can never be version-skewed:
a future HF push changes nothing here until the pin is bumped deliberately.

Bumping: update ``_DEFAULT_REVISION`` (or set ``SATP_HF_REVISION``), rerun
``./setup.sh`` (or ``--download-only``), restart the service, then re-run the
Step-A parity gate (244/244 byte-identical decodes) and the ``satp?`` e2e
sweep (92/244) before trusting any number.
"""

import os

HF_REPO = "ChristianZ97/satp-policy-v2-alphaproof"
_DEFAULT_REVISION = "a654c92d305730b8f0b251e01392a5934483e6a4"

REVISION = os.environ.get("SATP_HF_REVISION", _DEFAULT_REVISION).strip()
if not REVISION:
    raise RuntimeError(
        "SATP_HF_REVISION is set but empty — refusing to fall back to a "
        "floating revision; set a commit hash, or unset the variable to use "
        "the pinned default"
    )
