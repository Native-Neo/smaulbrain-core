"""Centralized CLI: every entry point uses these definitions + validation."""
import argparse

from . import config as _cfg
from . import precision as _precision


def add_common_args(ap: argparse.ArgumentParser, need_train: bool = False):
    ap.add_argument("--preset", default="moe-tiny",
                    help="full|tiny (with --moe/--dense) or moe-full|moe-tiny|dense-full|dense-tiny")
    ap.add_argument("--moe", action="store_true")
    ap.add_argument("--dense", action="store_true")
    ap.add_argument("--paging", default="R", choices=list(_cfg.PAGING_MODES))
    ap.add_argument("--paging-cache-experts", type=int, default=8)
    ap.add_argument("--paging-staging-bytes", type=int, default=1 << 24)
    ap.add_argument("--forward-quant", default="fp32", choices=list(_cfg.FORWARD_QUANT_CHOICES))
    ap.add_argument("--backward-mode", default="fp32", choices=list(_cfg.BACKWARD_MODE_CHOICES))
    ap.add_argument("--activation-storage", default="fp32",
                    choices=list(_precision.ACTIVATION_STORAGE_MODES),
                    help="storage format for differentiable activation caches")
    ap.add_argument("--gradient-storage", default="fp32",
                    choices=list(_precision.GRADIENT_STORAGE_MODES),
                    help="storage format for persistent parameter gradient buffers")
    ap.add_argument("--trainer", default="backprop", choices=("backprop", "dust"),
                    help="training algorithm: ordinary backpropagation (default) "
                         "or the experimental forward-only Dust estimator")
    ap.add_argument("--dust-writer-draws", type=int, default=2,
                    help="Dust draws per block for the attention/FFN residual writers")
    ap.add_argument("--dust-hidden-draws", type=int, default=2,
                    help="Dust draws per block for the FFN hidden projections")
    ap.add_argument("--dust-hub-draws", type=int, default=2,
                    help="Dust draws per block for the attention-output error targets")
    ap.add_argument("--dust-embed-draws", type=int, default=2,
                    help="Dust draws for the token embedding")
    ap.add_argument("--dust-head-draws", type=int, default=1024,
                    help="Dust head draws (slab cross-entropy evaluations)")
    ap.add_argument("--dust-local-draws", type=int, default=2,
                    help="Dust draws per block for each attention internal (r/k/v)")
    ap.add_argument("--dust-scalar-draws", type=int, default=0,
                    help="two-sided ES draws for non-matrix parameters "
                         "(0 freezes norm gains, attn.w and attn.u)")
    ap.add_argument("--dust-chunk", type=int, default=8,
                    help="independent draws evaluated together in one batched forward")
    ap.add_argument("--dust-sigma", type=float, default=0.05,
                    help="Dust noise scale for writers, FFN hidden and embedding")
    ap.add_argument("--dust-hub-sigma", type=float, default=0.05,
                    help="Dust noise scale for the shallow-half attention-output hubs")
    ap.add_argument("--dust-hub-sigma-deep", type=float, default=0.05,
                    help="Dust noise scale for the deep-half attention-output hubs")
    ap.add_argument("--dust-local-sigma", type=float, default=0.05,
                    help="Dust noise scale for attention internals")
    ap.add_argument("--dust-head-sigma", type=float, default=0.05,
                    help="Dust noise scale for head logit slabs")
    ap.add_argument("--dust-scalar-sigma", type=float, default=0.03,
                    help="two-sided ES noise scale for non-matrix parameters")
    ap.add_argument("--dust-gamma", type=float, default=0.5,
                    help="future-token credit decay for keys/values")
    ap.add_argument("--dust-head-slab", type=int, default=256,
                    help="number of vocabulary columns jittered per head draw")
    ap.add_argument("--dust-lr", type=float, default=0.05,
                    help="Dust SGD learning rate (separate from the backprop --lr)")
    ap.add_argument("--dust-momentum", type=float, default=0.9,
                    help="Dust SGD momentum")
    ap.add_argument("--dust-grad-clip", type=float, default=0.0,
                    help="global gradient-norm clip for Dust updates (0 disables; "
                         "the paper does not clip)")
    ap.add_argument("--dust-seed", type=int, default=None,
                    help="Dust perturbation seed (default: seed + 0xD057)")
    ap.add_argument("--device", default="cpu", help="cpu (accelerator optional, must not break CPU)")
    ap.add_argument("--checkpoint", default="checkpoints/run")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.98)
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--growth-every-steps", default="10000")
    ap.add_argument("--growth-loss-nats", default="0.5")
    ap.add_argument("--growth-loss-beta", type=float, default=0.98,
                    help="EMA decay for loss smoothing")
    ap.add_argument("--growth-min-eval-samples", type=int, default=256)
    ap.add_argument("--growth-noise-scale", type=float, default=0.01,
                    help="Gaussian mutation sigma as fraction of source RMS")
    ap.add_argument("--growth-seed", type=int, default=None)
    ap.add_argument("--expert-capacity-factor", type=float, default=1.25)
    ap.add_argument("--mtp-depth", default="10",
                    help="multi-token prediction depth: depth k predicts k "
                         "tokens ahead (0 disables; each depth adds a shared "
                         "V3-style module). Default 10.")
    ap.add_argument("--mtp-loss-weight", default="0.1",
                    help="weight applied to every MTP depth's cross-entropy")
    if need_train:
        ap.add_argument("--max-steps", type=int, default=50)
        ap.add_argument("--corpus", default=None, help="path to raw byte corpus")
        ap.add_argument("--synth-bytes", type=int, default=1 << 16)
        ap.add_argument("--grad-clip", type=float, default=1.0)
        ap.add_argument("--log-every", type=int, default=10)
        ap.add_argument("--ckpt-every", type=int, default=0)


def validate_args(args) -> str:
    preset = _cfg.resolve_preset(args.preset, args.moe, args.dense)
    _cfg.validate_quant_combo(args.forward_quant, args.backward_mode)
    _precision.validate_activation_storage(args.activation_storage)
    _precision.validate_gradient_storage(args.gradient_storage)
    args.growth_every_steps = _cfg.parse_growth_every_steps(args.growth_every_steps)
    args.growth_loss_nats = _cfg.parse_growth_loss_nats(args.growth_loss_nats)
    if args.seq_len < 1:
        raise ValueError("--seq-len must be >= 1")
    if args.paging_cache_experts < 1:
        raise ValueError("--paging-cache-experts must be >= 1")
    if args.expert_capacity_factor <= 0:
        raise ValueError("--expert-capacity-factor must be > 0")
    args.mtp_depth = _cfg.parse_mtp_depth(args.mtp_depth)
    args.mtp_loss_weight = _cfg.parse_mtp_loss_weight(args.mtp_loss_weight)
    if args.trainer == "dust":
        spec = _cfg.PRESETS[preset]
        if spec.kind != "dense":
            raise ValueError(
                "Dust supports dense models only; mixture-of-experts routing is "
                "not covered by the verified estimator")
        if args.mtp_depth != 0:
            raise ValueError("Dust requires --mtp-depth 0")
        if args.forward_quant != "fp32" or args.backward_mode != "fp32":
            raise ValueError("Dust requires --forward-quant fp32 --backward-mode fp32")
        if args.activation_storage != "fp32" or args.gradient_storage != "fp32":
            raise ValueError(
                "Dust requires --activation-storage fp32 --gradient-storage fp32; "
                "it uses its own clean-activation cache")
        if args.grad_clip != 1.0 and args.dust_grad_clip == 0.0:
            raise ValueError(
                "--grad-clip applies to backprop; use --dust-grad-clip for Dust")
        from . import dust as _dust
        _dust.config_from_args(args).validate(spec.vocab_size)
    elif getattr(args, "trainer", "backprop") != "backprop":
        raise ValueError(f"unknown trainer {args.trainer!r}")
    return preset
