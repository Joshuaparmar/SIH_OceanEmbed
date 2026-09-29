"""
Software-level shape and sanity tests for the OceanEmbed CBAM U-Net.

IMPORTANT: All tensors here are random (torch.randn) and used ONLY to
verify that the model's forward pass runs, produces the correct output
shape, and is numerically well-behaved, and that it can participate in
a training computation graph (forward -> loss -> backward -> optimizer
step). This is SOFTWARE-LEVEL VALIDATION ONLY. It is NOT scientific
validation, NOT training on real data, and NOT evidence of predictive
skill — no real ocean data is used or implied anywhere in this script.

Run with:
    python scripts/test_model.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn

from configs.model_config import ModelConfig
from models.cbam_unet import CBAMUNet


# ---------------------------------------------------------------------------
# 1. Forward-pass shape/sanity test (existing coverage, retained)
# ---------------------------------------------------------------------------

def run_case(name: str, config: ModelConfig, batch_size: int, height: int, width: int):
    print(f"\n--- Test case: {name} ---")
    print(
        f"config: in_channels={config.in_channels}, out_depths={config.out_depths}, "
        f"num_stages={config.num_stages}, base_channels={config.base_channels}, "
        f"norm={config.norm}, downsample_mode={config.downsample_mode}, "
        f"upsample_mode={config.upsample_mode}"
    )
    print(f"input shape target: [B={batch_size}, C={config.in_channels}, H={height}, W={width}]")

    model = CBAMUNet(config)
    model.eval()

    x = torch.randn(batch_size, config.in_channels, height, width)

    start = time.time()
    with torch.no_grad():
        y = model(x)
    elapsed = time.time() - start

    expected_shape = (batch_size, config.out_depths, height, width)
    assert tuple(y.shape) == expected_shape, (
        f"Shape mismatch: expected {expected_shape}, got {tuple(y.shape)}"
    )
    assert torch.isfinite(y).all(), "Output contains NaN or Inf values"

    param_count = model.count_parameters()

    print(f"output shape:        {tuple(y.shape)}  (matches expected [B, D, H, W]: PASS)")
    print(f"no NaN/Inf:           PASS")
    print(f"forward pass time:    {elapsed:.4f}s")
    print(f"parameter count:      {param_count:,}")

    return model, param_count


# ---------------------------------------------------------------------------
# 2. Backward-pass / optimizer-step training-graph sanity test
# ---------------------------------------------------------------------------

def run_backward_pass_case(name: str, config: ModelConfig, batch_size: int, height: int, width: int):
    """
    Verifies the model can participate in a training computation graph:
    forward -> MSE loss -> backward -> gradient sanity checks -> one Adam
    step. This checks only that gradients flow and are numerically sane —
    it is NOT a claim that the model learns anything meaningful, since the
    "data" here is random noise with no real relationship to the "target".
    """
    print(f"\n--- Backward-pass test case: {name} ---")

    model = CBAMUNet(config)
    model.train()

    x = torch.randn(batch_size, config.in_channels, height, width)
    target = torch.randn(batch_size, config.out_depths, height, width)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()

    pred = model(x)
    assert tuple(pred.shape) == (batch_size, config.out_depths, height, width), (
        f"Shape mismatch before loss: got {tuple(pred.shape)}"
    )

    loss_fn = nn.MSELoss()
    loss = loss_fn(pred, target)
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"

    loss.backward()

    # Every trainable parameter that participates in the forward pass should
    # receive a gradient, and that gradient must be finite.
    missing_grad = []
    nan_or_inf_grad = []
    total_params_checked = 0
    for pname, param in model.named_parameters():
        if not param.requires_grad:
            continue
        total_params_checked += 1
        if param.grad is None:
            missing_grad.append(pname)
        elif not torch.isfinite(param.grad).all():
            nan_or_inf_grad.append(pname)

    grad_ok = (len(missing_grad) == 0) and (len(nan_or_inf_grad) == 0)

    # One optimizer step should run cleanly against these gradients.
    optimizer_step_ok = True
    optimizer_error = None
    try:
        optimizer.step()
    except Exception as exc:  # pragma: no cover - only triggered on real failure
        optimizer_step_ok = False
        optimizer_error = str(exc)

    print(f"loss value:              {loss.item():.6f}  (finite: PASS)")
    print(f"parameters checked:      {total_params_checked}")
    print(f"parameters missing grad: {len(missing_grad)}"
          + (f"  -> {missing_grad[:5]}{'...' if len(missing_grad) > 5 else ''}" if missing_grad else ""))
    print(f"parameters with NaN/Inf grad: {len(nan_or_inf_grad)}"
          + (f"  -> {nan_or_inf_grad[:5]}{'...' if len(nan_or_inf_grad) > 5 else ''}" if nan_or_inf_grad else ""))
    print(f"gradient check:          {'PASS' if grad_ok else 'FAIL'}")
    print(f"Adam optimizer.step():   {'PASS' if optimizer_step_ok else 'FAIL (' + str(optimizer_error) + ')'}")

    overall = grad_ok and optimizer_step_ok
    print(f"BACKWARD-PASS TEST:      {'PASS' if overall else 'FAIL'}")

    assert overall, "Backward-pass / optimizer-step sanity test failed"
    return loss.item()


# ---------------------------------------------------------------------------
# 3. Embedding-extraction test
# ---------------------------------------------------------------------------

def run_embedding_case(name: str, config: ModelConfig, batch_size: int, height: int, width: int):
    """
    Exercises forward_with_embedding() and reports the shapes of the final
    prediction and the bottleneck embedding. The embedding is a spatial
    feature map (not a flattened vector) — see CBAMUNet.forward_with_embedding
    docstring for the exact shape semantics.
    """
    print(f"\n--- Embedding-extraction test case: {name} ---")

    model = CBAMUNet(config)
    model.eval()

    x = torch.randn(batch_size, config.in_channels, height, width)
    print(f"input shape:    {tuple(x.shape)}")

    with torch.no_grad():
        pred, embedding = model.forward_with_embedding(x)

    expected_out_shape = (batch_size, config.out_depths, height, width)
    assert tuple(pred.shape) == expected_out_shape, (
        f"Prediction shape mismatch: expected {expected_out_shape}, got {tuple(pred.shape)}"
    )
    assert embedding.dim() == 4, f"Expected a 4D spatial embedding [B, C, H, W], got {embedding.dim()}D"
    assert embedding.size(0) == batch_size, "Embedding batch dimension mismatch"
    assert torch.isfinite(embedding).all(), "Embedding contains NaN or Inf values"
    assert torch.isfinite(pred).all(), "Prediction contains NaN or Inf values"

    print(f"embedding shape: {tuple(embedding.shape)}  (spatial feature map, not a flattened vector)")
    print(f"output shape:    {tuple(pred.shape)}  (matches expected [B, D, H, W]: PASS)")
    print("EMBEDDING-EXTRACTION TEST: PASS")

    return tuple(embedding.shape)


# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(0)

    results = []

    # Case 1: current working spec from the project brief — [B, 5, H, W] -> [B, 13, H, W]
    cfg1 = ModelConfig(in_channels=5, out_depths=13, num_stages=3, base_channels=32)
    _, p1 = run_case("current spec (C=5, D=13), square HxW", cfg1, batch_size=2, height=64, width=64)
    results.append(("current spec (C=5, D=13)", p1))

    # Case 2: different configurable channel/depth counts, to prove nothing is hardcoded
    # (also close to the currently intended 8-channel surface variable set)
    cfg2 = ModelConfig(in_channels=9, out_depths=20, num_stages=3, base_channels=24)
    _, p2 = run_case("alternate config (C=9, D=20)", cfg2, batch_size=1, height=96, width=96)
    results.append(("alternate config (C=9, D=20)", p2))

    # Case 3: non-square spatial size
    cfg3 = ModelConfig(in_channels=5, out_depths=13, num_stages=3, base_channels=32)
    _, p3 = run_case("non-square spatial size", cfg3, batch_size=2, height=80, width=120)
    results.append(("non-square spatial size", p3))

    # Case 4: spatial size NOT divisible by 2^num_stages (tests robust skip alignment)
    #   301 and 481 are the real dataset dims mentioned in the brief, used here ONLY
    #   as an odd/non-divisible shape stress test — no real data is involved.
    cfg4 = ModelConfig(in_channels=5, out_depths=13, num_stages=3, base_channels=16)
    _, p4 = run_case("odd, non-power-of-2-divisible size (301x481)", cfg4, batch_size=1, height=301, width=481)
    results.append(("odd size 301x481", p4))

    # Case 5: stride-conv downsample + transposed-conv upsample variant, CBAM on
    cfg5 = ModelConfig(
        in_channels=5, out_depths=13, num_stages=4, base_channels=16,
        downsample_mode="stride", upsample_mode="transpose",
    )
    _, p5 = run_case("stride/transpose variant, 4 stages", cfg5, batch_size=2, height=67, width=53)
    results.append(("stride/transpose variant", p5))

    # Case 6: CBAM disabled, to confirm the toggle works end-to-end
    cfg6 = ModelConfig(in_channels=5, out_depths=13, num_stages=3, base_channels=32, use_cbam=False)
    _, p6 = run_case("CBAM disabled", cfg6, batch_size=2, height=64, width=64)
    results.append(("CBAM disabled", p6))

    # Case 7: GroupNorm instead of BatchNorm (useful for small batch sizes)
    cfg7 = ModelConfig(
        in_channels=8, out_depths=13, num_stages=3, base_channels=24,
        norm="group", group_norm_groups=8,
    )
    _, p7 = run_case("GroupNorm, batch size 1", cfg7, batch_size=1, height=64, width=64)
    results.append(("GroupNorm (B=1)", p7))

    print("\n" + "=" * 70)
    print("FORWARD-PASS TEST CASES: ALL PASSED")
    print("=" * 70)
    for name, params in results:
        print(f"  {name:<45}{params:>15,} params")

    # --- Backward-pass / optimizer-step tests -------------------------------
    print("\n" + "=" * 70)
    print("BACKWARD-PASS / OPTIMIZER-STEP TESTS")
    print("(software-level training-graph check only — NOT scientific validation)")
    print("=" * 70)
    run_backward_pass_case("current spec (C=5, D=13), square HxW", cfg1, batch_size=2, height=64, width=64)
    run_backward_pass_case("odd size 301x481 (small config for speed)", cfg4, batch_size=1, height=61, width=97)
    run_backward_pass_case("GroupNorm, batch size 1", cfg7, batch_size=1, height=64, width=64)

    # --- Embedding-extraction tests ------------------------------------------
    print("\n" + "=" * 70)
    print("EMBEDDING-EXTRACTION TESTS")
    print("=" * 70)
    run_embedding_case("current spec (C=5, D=13), square HxW", cfg1, batch_size=2, height=64, width=64)
    run_embedding_case("odd spatial dims (301x481)", cfg4, batch_size=1, height=301, width=481)

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED (forward shape, backward/optimizer, embedding extraction)")
    print("This validates software correctness only — NOT scientific model validation.")
    print("=" * 70)

    # Print a full architecture summary for the primary spec (Case 1)
    print("\n")
    model1 = CBAMUNet(cfg1)
    print(model1.summary())


if __name__ == "__main__":
    main()
