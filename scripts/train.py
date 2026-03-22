"""
Training entry point.

Usage:
    python scripts/train.py
    python scripts/train.py --fold 2         # run a single fold only
    python scripts/train.py --output ./runs  # custom output directory
"""

import argparse
import logging
import sys
from pathlib import Path
import yaml
import os

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/ 'src'))

from volregime.utils.config import load_config, get_project_root
from volregime.data.dataset import SurfaceAlphaDataset
from volregime.training.walk_forward import WalkForwardOrchestrator

def main():
    parser = argparse.ArgumentParser(description="SurfaceAlpha walk-forward training")
    parser.add_argument("--fold",    type=int, default=None,
                        help="Run a single fold index (default: all folds)")
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
    data_dir = Path(os.environ.get('DATA_DIR', project_root / 'data'))
    sample_index = data_dir / 'processed' / 'sample_index.parquet'
    processed_dir = data_dir / 'processed'

    logger.info('Loading dataset from %s', sample_index)
    dataset = SurfaceAlphaDataset(sample_index, processed_dir)
    logger.info('Dataset: %d samples, %d symbols, dates %s -> %s', len(dataset), len(dataset.get_symbols()), dataset.get_dates()[0], dataset.get_dates()[-1])

    # walk-forward
    output_dir = args.output or str(project_root)
    orchestrator = WalkForwardOrchestrator(cfg, dataset, output_dir=output_dir)

    if args.fold is not None:
        # single fold mode
        specs = orchestrator.compute_fold_specs()
        if args.fold >= len(specs):
            logger.error("Fold %d does not exist (max=%d)", args.fold, len(specs)-1)
            sys.exit(1)
        
        # Temporarily override num_folds and step_days to run just this fold
        # by running the orchestrator with a single-fold dataset view
        spec = specs[args.fold]
        logger.info("Running fold %d only: test %s -> %s", args.fold, spec.test_start, spec.test_end)
        # Slice dataset to the fold range and run orchestrator
        # Simplest: run all folds but only save results for the requested one
        results = orchestrator.run()
        results = [r for r in results if r['fold_idx'] == args.fold]
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