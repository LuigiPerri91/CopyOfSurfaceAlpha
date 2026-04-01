"""
Training entry point.

Usage:
    python scripts/train.py                    # run all folds
    python scripts/train.py --fold 2           # run a single fold only
    python scripts/train.py --start-fold 2     # resume from fold 2 to end
    python scripts/train.py --output ./runs    # custom output directory
"""

import argparse
import logging
import sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/ 'src'))

from volregime.utils.config import load_config, get_project_root
from volregime.data.dataset import SurfaceAlphaDataset
from volregime.training.walk_forward import WalkForwardOrchestrator

def main():
    parser = argparse.ArgumentParser(description="SurfaceAlpha walk-forward training")
    parser.add_argument("--fold",       type=int, default=None,
                        help="Run a single fold index (default: all folds)")
    parser.add_argument("--start-fold", type=int, default=None,
                        help="Resume from this fold index, running through all remaining folds")
    parser.add_argument("--output", type=str, default=None,
                        help="Output root directory (default: project root)")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    logger = logging.getLogger("train")

    # load config
    project_root = get_project_root()
    cfg = load_config()

    # dataset
    sample_index = Path(cfg['paths']['processed_dir']) / 'sample_index.parquet'
    processed_dir = Path(cfg['paths']['processed_dir'])

    logger.info('Loading dataset from %s', sample_index)
    dataset = SurfaceAlphaDataset(sample_index, processed_dir)
    logger.info('Dataset: %d samples, %d symbols, dates %s -> %s', len(dataset), len(dataset.get_symbols()), dataset.get_dates()[0], dataset.get_dates()[-1])

    # Sync model input dimensions to processed tensor shapes to avoid config drift.
    if len(dataset) > 0:
        sample = dataset[0]
        model_cfg = cfg.setdefault("model", {})
        ret_cfg = model_cfg.setdefault("returns_encoder", {})
        ctx_cfg = model_cfg.setdefault("context_encoder", {})
        surf_cfg = model_cfg.setdefault("surface_encoder", {})

        observed_ret_dim = int(sample["returns"].shape[-1])
        configured_ret_dim = int(ret_cfg.get("input_dim", observed_ret_dim))
        if configured_ret_dim != observed_ret_dim:
            logger.warning(
                "returns_encoder.input_dim mismatch (config=%d, data=%d). Using data shape.",
                configured_ret_dim,
                observed_ret_dim,
            )
        ret_cfg["input_dim"] = observed_ret_dim

        observed_ctx_dim = int(sample["vol_history"].numel() + sample["market_state"].numel())
        configured_ctx_dim = int(ctx_cfg.get("input_dim", observed_ctx_dim))
        if configured_ctx_dim != observed_ctx_dim:
            logger.warning(
                "context_encoder.input_dim mismatch (config=%d, data=%d). Using data shape.",
                configured_ctx_dim,
                observed_ctx_dim,
            )
        ctx_cfg["input_dim"] = observed_ctx_dim

        observed_surface_channels = int(sample["surface"].shape[0])
        configured_surface_channels = int(surf_cfg.get("num_input_channels", observed_surface_channels))
        if configured_surface_channels != observed_surface_channels:
            logger.warning(
                "surface_encoder.num_input_channels mismatch (config=%d, data=%d). Using data shape.",
                configured_surface_channels,
                observed_surface_channels,
            )
        surf_cfg["num_input_channels"] = observed_surface_channels

    # walk-forward
    output_dir = args.output or str(project_root)
    orchestrator = WalkForwardOrchestrator(cfg, dataset, output_dir=output_dir)

    if args.fold is not None and args.start_fold is not None:
        logger.error("--fold and --start-fold are mutually exclusive")
        sys.exit(1)

    if args.fold is not None:
        specs = orchestrator.compute_fold_specs()
        if args.fold >= len(specs):
            logger.error("Fold %d does not exist (max=%d)", args.fold, len(specs) - 1)
            sys.exit(1)
        spec = specs[args.fold]
        logger.info("Running fold %d only: test %s -> %s", args.fold, spec.test_start, spec.test_end)
        results = orchestrator.run(start_fold=args.fold, end_fold=args.fold + 1)
    elif args.start_fold is not None:
        specs = orchestrator.compute_fold_specs()
        if args.start_fold >= len(specs):
            logger.error("--start-fold %d does not exist (max=%d)", args.start_fold, len(specs) - 1)
            sys.exit(1)
        logger.info("Resuming from fold %d", args.start_fold)
        results = orchestrator.run(start_fold=args.start_fold)
    else:
        results = orchestrator.run()

    logger.info('Training complete.')
    for r in results:
        logger.info(
            '   Fold %d | val_loss=%.4f | test_samples=%d | preds=%s',
            r['fold_idx'], r['best_val_loss'], r['n_test_samples'],
            r['predictions_path']
        )

if __name__ == "__main__":
    main()
