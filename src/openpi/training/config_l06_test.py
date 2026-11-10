import sys

import pytest

from openpi.training import config as _config

# The FREEZE list of runbook 2 (docs/reports/phase2-rerun-runbook.zh-TW.md, $W/r2/env2.sh).
_P = "paligemma_with_expert"
_L = f"{_P}.paligemma.model.language_model"
RUNBOOK2_FREEZE = (
    f"{_P}.paligemma.model.vision_tower. {_P}.paligemma.model.multi_modal_projector. {_L}.embed_tokens. "
    f"{_P}.paligemma.lm_head. {_P}.gemma_expert.lm_head. {_L}.norm. {_L}.layers.5.mlp. "
    f"{_L}.layers.5.self_attn.q_proj. {_L}.layers.5.self_attn.o_proj. {_L}.layers.5.post_attention_layernorm."
).split()


@pytest.mark.parametrize("name", ["pi05_libero_l06", "pi05_libero_l06_taskonly"])
def test_l06_configs_freeze_runbook2_prefixes(name: str):
    assert _config.get_config(name).pytorch_freeze_prefixes == tuple(RUNBOOK2_FREEZE)


def test_cli_freeze_prefixes_override_replaces_default(monkeypatch: pytest.MonkeyPatch):
    # The runbooks pass --pytorch-freeze-prefixes explicitly; it must replace the default, not extend it.
    monkeypatch.setattr(
        sys, "argv", ["train", "pi05_libero_l06", "--exp-name", "t", "--pytorch-freeze-prefixes", "a.", "b."]
    )
    assert _config.cli().pytorch_freeze_prefixes == ("a.", "b.")
