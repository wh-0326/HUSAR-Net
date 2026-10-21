"""Check portable training/checkpoints and the optimized decoder."""
from dataclasses import asdict

import numpy as np
import pytest
import torch

from environment import HUSAREnv
from fast_decode import FastDecoder
from inference import decode
from model import HUSARNet, HUSARNetConfig
from objective import husar_objective
from synthetic import make_hierarchy, make_instance
from train import TrainConfig, train


def test_impossible_hierarchy_is_rejected():
    with pytest.raises(ValueError, match="n_cities"):
        make_hierarchy(n_firms=10, n_cities=20, n_provs=2, seed=0)


@pytest.mark.parametrize("agg", ["mean", "min", "cvar"])
@pytest.mark.parametrize("scenario_branches", [False, True])
def test_fast_greedy_matches_reference(agg, scenario_branches):
    torch.set_num_threads(1)
    torch.manual_seed(12)
    config = HUSARNetConfig(d_model=32, d_mlp=64,
                            use_scenario_encoder=scenario_branches,
                            use_scenario_projection=scenario_branches)
    model = HUSARNet(config).eval()
    instance = make_instance(family="B", n=60, n_scen=10, seed=700001)
    env = HUSAREnv(instance, 8, agg=agg, device="cpu")
    reference = decode(model, env, mode="greedy")
    fast = FastDecoder(model, env.T, 8, agg=agg).greedy(block=1)
    assert fast["selection"] == reference["selection"]
    assert husar_objective(instance, fast["selection"], agg) == pytest.approx(
        reference["objective"])


@pytest.mark.parametrize("scenario_branches", [False, True])
def test_training_checkpoint_retains_architecture(tmp_path, scenario_branches):
    torch.set_num_threads(1)
    torch.manual_seed(1)
    config = HUSARNetConfig(d_model=32, d_mlp=64,
                            use_scenario_encoder=scenario_branches,
                            use_scenario_projection=scenario_branches)
    settings = TrainConfig(iters=1, instances_per_update=2, device="cpu",
                           n_min=60, n_max=80, n_scen_choices=(10,),
                           val_instances=2, val_every=1, out_dir=str(tmp_path))
    model, summary = train(settings, HUSARNet(config), log_fn=lambda _: None)
    checkpoint = torch.load(tmp_path / "best.pt", weights_only=True)
    assert checkpoint["net_cfg"] == asdict(config)
    assert checkpoint["val"]["val_ratio_vs_lcb"] > 0
    assert np.isfinite(summary["history"][0]["loss"])
    restored = HUSARNet(HUSARNetConfig(**checkpoint["net_cfg"]))
    restored.load_state_dict(checkpoint["model"])
    for key, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])
