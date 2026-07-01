#!/usr/bin/env python3
"""Automated ablation study for COSMOBridge v5.

Runs 6 experiments (A0-A5) defined in configs/ablation.yaml, each with 10 seeds.
Measures the marginal contribution of each image component.

Experiments:
    A0: v4 baseline (no images, gated fusion)
    A1: + Multi-View ViT (random init, no SimCLR)
    A2: + Multi-View ViT (SimCLR pre-trained)
    A3: + Cation-Anion Siamese
    A4: + Cross-Modal Attention (replace gates)
    A5: Full v5 (all components)

Usage:
    python ablation_study.py --config configs/ablation.yaml
    python ablation_study.py --experiment A2 --seeds 0-4
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
V5_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(V5_ROOT))


def build_ablation_model(experiment_config, base_config, device):
    """Build a model variant for a specific ablation experiment.

    Args:
        experiment_config: dict with enable/disable flags
        base_config: full v5 config dict
        device: torch device

    Returns:
        model: COSMOBridgeV5 (possibly with disabled modalities)
    """
    from models.cosmobridge_v5 import COSMOBridgeV5

    mc = base_config.get("model", {})
    model = COSMOBridgeV5(
        embed_dim=mc.get("embed_dim", 256),
        n_properties=mc.get("n_properties", 7),
        n_views=mc.get("multiview_vit", {}).get("n_views", 36),
        graph_dim=mc.get("graph", {}).get("dim", 300),
        surface_dim=mc.get("pointcloud", {}).get("dim", 256),
        thermo_dim=mc.get("tabular", {}).get("dim", 25),
        dropout=mc.get("dropout", 0.2),
    ).to(device)

    # Apply ablation flags
    ec = experiment_config

    if not ec.get("enable_multiview_vit", True):
        # Zero out ViT pathway: freeze and zero weights
        for param in model.multiview_vit.parameters():
            param.requires_grad = False
            param.data.zero_()
        for param in model.vit_proj.parameters():
            param.requires_grad = False
            param.data.zero_()

    if not ec.get("enable_siamese", True):
        for param in model.siamese.parameters():
            param.requires_grad = False
            param.data.zero_()
        for param in model.siamese_proj.parameters():
            param.requires_grad = False
            param.data.zero_()

    if not ec.get("enable_cross_modal_attention", True):
        # Zero out cross-attention blocks
        for name, param in model.fusion.named_parameters():
            if "vit_graph" in name or "vit_surface" in name or \
               "siamese_graph" in name or "graph_surface" in name:
                param.requires_grad = False
                param.data.zero_()

    # Load SimCLR weights if enabled
    if ec.get("vit_pretrained", False):
        simclr_path = mc.get("multiview_vit", {}).get("pretrained_checkpoint",
                              str(V5_ROOT / "checkpoints/simclr/vit_pretrained.pt"))
        if Path(simclr_path).exists():
            model.load_simclr_weights(simclr_path)
        else:
            print(f"  WARNING: SimCLR checkpoint not found: {simclr_path}")

    # Initialize routing
    routing_init = base_config.get("model", {}).get("routing_init")
    if routing_init:
        model.fusion.init_routing_from_domain_knowledge(routing_init)

    return model


def run_single_experiment(exp_name, exp_config, base_config, seeds, device):
    """Run one ablation experiment across multiple seeds.

    Returns:
        dict with per-seed and aggregate metrics
    """
    from train_v5 import set_seed, train_stage, evaluate, compute_metrics
    from data.dataset import build_dataloader

    print(f"\n{'='*60}")
    print(f"  ABLATION: {exp_name}")
    print(f"  {exp_config.get('description', '')}")
    print(f"  Seeds: {seeds}")
    print(f"{'='*60}")

    tc = base_config.get("training", {})
    criterion = nn.MSELoss()

    all_metrics = {}

    for seed in seeds:
        set_seed(seed)
        print(f"\n  --- Seed {seed} ---")

        model = build_ablation_model(exp_config, base_config, device)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"  Trainable: {trainable:,} / {total:,}")

        train_loader = build_dataloader("train", base_config, seed)
        val_loader = build_dataloader("val", base_config, seed)
        test_loader = build_dataloader("test", base_config, seed)

        ckpt_dir = V5_ROOT / "results" / "ablation" / exp_name / f"seed_{seed}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        anchor_weights = model.fusion.routing_logits.detach().clone()

        # Stage 1: Freeze encoders
        model.freeze_encoders()
        s1 = tc.get("stage1", {})
        opt1 = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=s1.get("lr", 1e-3),
            weight_decay=tc.get("weight_decay", 1e-3),
        )
        _, val_r2_1 = train_stage(
            model, train_loader, val_loader, opt1, None,
            criterion, device, f"S1 ({exp_name}, seed {seed})",
            epochs=s1.get("epochs", 5), patience=s1.get("epochs", 5),
            anchor_loss_lambda=tc.get("anchor_loss_lambda", 0.05),
            anchor_weights=anchor_weights, checkpoint_dir=ckpt_dir,
        )

        # Stage 2: Unfreeze image encoders (if enabled)
        if exp_config.get("enable_multiview_vit", True) or \
           exp_config.get("enable_siamese", True):
            model.unfreeze_image_encoders()

        s2 = tc.get("stage2", {})
        param_groups = model.get_parameter_groups(
            image_lr=s2.get("image_lr", 1e-4),
            fusion_lr=s2.get("fusion_lr", 1e-3),
        )
        opt2 = torch.optim.AdamW(param_groups, weight_decay=tc.get("weight_decay", 1e-3))
        sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=s2.get("epochs", 50))
        _, val_r2_2 = train_stage(
            model, train_loader, val_loader, opt2, sched2,
            criterion, device, f"S2 ({exp_name}, seed {seed})",
            epochs=s2.get("epochs", 50),
            patience=s2.get("early_stopping_patience", 20),
            checkpoint_dir=ckpt_dir,
        )

        # Evaluate
        test_metrics, test_preds, test_targets = evaluate(
            model, test_loader, criterion, device,
        )

        all_metrics[seed] = test_metrics
        print(f"  Seed {seed}: avg R² = {test_metrics['avg_r2']:.4f}")

        # Save predictions
        np.savez(
            ckpt_dir / "predictions.npz",
            predictions=test_preds.numpy(),
            targets=test_targets.numpy(),
        )

    # Aggregate
    property_names = ["gamma1", "gamma2", "G_E", "H_E", "G_mix", "H_vap", "P"]
    summary = {"experiment": exp_name, "description": exp_config.get("description", "")}

    for prop in property_names:
        key = f"{prop}_r2"
        vals = [m[key] for m in all_metrics.values()]
        summary[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    avg_r2s = [m["avg_r2"] for m in all_metrics.values()]
    summary["avg_r2"] = {"mean": float(np.mean(avg_r2s)), "std": float(np.std(avg_r2s))}
    summary["per_seed"] = {str(k): v for k, v in all_metrics.items()}

    return summary


def main():
    parser = argparse.ArgumentParser(description="Ablation study")
    parser.add_argument("--config", type=str, default="configs/ablation.yaml")
    parser.add_argument("--base_config", type=str, default="configs/v5_full.yaml")
    parser.add_argument("--experiment", type=str, default=None,
                        help="Run single experiment (e.g., A2)")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Seed range (e.g., 0-9)")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    import yaml

    # Load configs
    ablation_path = V5_ROOT / args.config
    base_path = V5_ROOT / args.base_config

    if ablation_path.exists():
        with open(ablation_path) as f:
            ablation_config = yaml.safe_load(f)
    else:
        print(f"Ablation config not found: {ablation_path}")
        return

    if base_path.exists():
        with open(base_path) as f:
            base_config = yaml.safe_load(f)
    else:
        base_config = {}

    experiments = ablation_config.get("experiments", {})
    common = ablation_config.get("common", {})

    # Seeds
    if args.seeds:
        start, end = map(int, args.seeds.split("-"))
        seeds = list(range(start, end + 1))
    else:
        seeds = common.get("seeds", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    device = torch.device(args.device)

    # Select experiments
    if args.experiment:
        exp_keys = [k for k in experiments if args.experiment in k]
        if not exp_keys:
            print(f"Experiment '{args.experiment}' not found. Available: {list(experiments.keys())}")
            return
    else:
        exp_keys = list(experiments.keys())

    print(f"Ablation Study: {len(exp_keys)} experiments x {len(seeds)} seeds")
    print(f"Device: {device}")

    # Run experiments
    all_results = {}
    for exp_name in exp_keys:
        exp_config = experiments[exp_name]
        result = run_single_experiment(
            exp_name, exp_config, base_config, seeds, device,
        )
        all_results[exp_name] = result

    # Save results
    output_dir = Path(common.get("output_dir", str(V5_ROOT / "results/ablation")))
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "ablation_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print comparison table
    print(f"\n{'='*70}")
    print("ABLATION RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Experiment':25s} {'avg R²':>10s} {'gamma1':>8s} {'gamma2':>8s} "
          f"{'G_E':>8s} {'H_E':>8s} {'G_mix':>8s}")
    print("-" * 70)
    for exp_name, result in all_results.items():
        avg = result["avg_r2"]
        g1 = result["gamma1_r2"]
        g2 = result["gamma2_r2"]
        ge = result["G_E_r2"]
        he = result["H_E_r2"]
        gm = result["G_mix_r2"]
        print(f"{exp_name:25s} {avg['mean']:>6.4f}±{avg['std']:.3f} "
              f"{g1['mean']:>6.4f} {g2['mean']:>6.4f} "
              f"{ge['mean']:>6.4f} {he['mean']:>6.4f} {gm['mean']:>6.4f}")

    print(f"\nResults saved to: {output_dir / 'ablation_results.json'}")


if __name__ == "__main__":
    main()
