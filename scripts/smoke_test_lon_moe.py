"""Smoke test for Lat+Lon MoE module.

Run from repo root with the sparsedrive conda env active:
    conda activate sparsedrive
    python scripts/smoke_test_lon_moe.py
"""
import torch
from projects.mmdet3d_plugin.models.motion.motion_blocks import LatLonPredModuleV13MoE


def main():
    torch.manual_seed(0)
    mod = LatLonPredModuleV13MoE(
        embed_dims=256,
        plan_config={'lat': {}, 'lon': {}, 'traj': {}, 'collision': {}},
        ffn_cfg=None,
        num_path_experts=5,
        num_vel_experts=6,
        router_aux_dim=None,
        router_routing_dim=64,
        ffn_hidden_dim=512,
        ffn_drop=0.0,
        tau=1.0,
    ).train()

    B, NP, NV, D = 2, 64, 16, 256
    path_embed = torch.randn(B, NP, D, requires_grad=True)
    vel_embed = torch.randn(B, NV, D, requires_grad=True)
    path_vocab = torch.randn(B, NP, 15, 2)
    vel_vocab = torch.randn(B, NV, 6)
    traj_vocab = torch.randn(B, NP, NV, 6, 2)
    traj_mask = torch.ones(B, NP, NV, 6)

    out = mod(path_embed, vel_embed, path_vocab, vel_vocab,
              traj_vocab, traj_mask, filter_num=(32, 8))
    print(f"forward OK, n_outputs={len(out)}")
    print(f"last_path_q  : {tuple(mod.last_path_q.shape)}")
    print(f"last_vel_q   : {tuple(mod.last_vel_q.shape)}")
    print(f"path weights row-sums = {mod.last_path_weights.sum(-1).tolist()}")
    print(f"vel  weights row-sums = {mod.last_vel_weights.sum(-1).tolist()}")

    path_labels = torch.tensor([0, 2])
    vel_labels = torch.tensor([1, 5])
    dr_p = mod.path_router.dr_loss(mod.last_path_q, path_labels)
    dr_v = mod.vel_router.dr_loss(mod.last_vel_q, vel_labels)
    print(f"DR loss path={dr_p.item():.4f}  vel={dr_v.item():.4f}")

    # Gradient flow
    total = dr_p + dr_v + out[6].sum() * 1e-4
    total.backward()
    assert path_embed.grad is not None and path_embed.grad.abs().sum() > 0
    assert vel_embed.grad is not None and vel_embed.grad.abs().sum() > 0
    print("grad flow OK (path_embed, vel_embed)")

    # Tau set works
    mod.tau = 0.3
    mod(path_embed.detach(), vel_embed.detach(), path_vocab, vel_vocab,
        traj_vocab, traj_mask, filter_num=(32, 8))
    print("tau=0.3 forward OK")
    print("ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
