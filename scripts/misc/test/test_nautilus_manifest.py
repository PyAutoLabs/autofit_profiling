from dataclasses import replace

import pytest
from profiling.nautilus_manifest import RunConfig, assert_matching, fingerprint


def test_adapters_share_resolved_contract():
    config = RunConfig()
    assert_matching(config.matching(), replace(config, adapter="pyautofit", trace=False).matching())
    assert config.constructor()["n_networks"] == 4
    assert config.matching()["pool"]["sampler_workers"] == 0


@pytest.mark.parametrize(
    "change",
    [{"n_eff": 10000}, {"n_batch": 101}, {"checkpoint": False}, {"workers": 2}, {"seed": 1}],
)
def test_confounded_comparison_rejected(change):
    config = RunConfig()
    with pytest.raises(ValueError, match="unmatched"):
        assert_matching(config.matching(), replace(config, **change).matching())


@pytest.mark.parametrize(
    "change", [{"dimension": 1}, {"dimension": 200}, {"work": -1}, {"update_interval": 100}]
)
def test_invalid_setup_rejected(change):
    with pytest.raises(ValueError):
        RunConfig(**change)


def test_fingerprint_is_order_independent():
    assert fingerprint({"a": 1, "b": 2}) == fingerprint({"b": 2, "a": 1})


def test_relocated_dependencies_keep_identity_but_revisions_do_not():
    from profiling.nautilus_manifest import source_pins

    left = {"PyAutoFit": {"root": "/old", "revision": "abc", "code_sha256": "def"}}
    moved = {"PyAutoFit": {"root": "/new", "revision": "abc", "code_sha256": "def"}}
    assert_matching(source_pins(left), source_pins(moved))
    moved["PyAutoFit"]["revision"] = "ghi"
    with pytest.raises(ValueError):
        assert_matching(source_pins(left), source_pins(moved))
