"""
src/run_pipeline.py
=====================

End-to-end pipeline orchestrator for the OmniStyle Customer Intelligence
Platform. Runs every stage in order, using the actual callable entry points
that already exist in each module (inspected directly rather than assumed):

  1. Data ingestion          -> src.data_ingestion.main()
  2. Data cleaning           -> src.data_cleaning.main()
  3. Feature engineering     -> src.feature_engineering.main()
  4. MySQL loading           -> src.load_to_mysql.main()            (skippable)
  5. Customer segmentation   -> src.customer_segmentation.main()
  6. Churn model training    -> src.train_model.train_churn_models() +
                                 src.train_model.run_shap_explainability()
  7. Business recommendations-> src.generate_recommendations.main()
  8. Power BI export         -> src.export_dashboard_data.main()

Note on stage 6: `src.train_model.main()` itself runs segmentation, churn
training, SHAP, a first-pass dashboard export, and a first-pass
recommendations report all in one call. To avoid re-running segmentation
twice and producing duplicate/overwritten reports in the wrong order, this
orchestrator calls the churn-specific functions from `src.train_model`
directly for stage 6, and lets stages 5, 7, and 8 be handled by their own
dedicated, standalone modules. This keeps each stage's logging and failure
handling independent and avoids doing duplicate work.

Run as:
    python -m src.run_pipeline
    python -m src.run_pipeline --skip-mysql
"""

from __future__ import annotations

import argparse
import sys
import time

from src.config import get_logger

logger = get_logger(__name__)


class PipelineStageError(RuntimeError):
    """Raised when a pipeline stage fails critically and the run cannot continue."""


def _run_stage(stage_number: int, stage_name: str, fn, *args, **kwargs):
    """
    Run a single pipeline stage with consistent logging and error handling.

    Stages are invoked as plain Python function calls, not subprocesses.
    Some stage entry points (currently `src.data_ingestion.main`) build
    their own `argparse.ArgumentParser` and call `parse_args()` with no
    explicit arguments, which reads `sys.argv` of *this* process. Since
    `run_pipeline.py` has its own top-level flags (e.g. `--skip-mysql`),
    we temporarily clear `sys.argv` down to just the program name for the
    duration of each stage call so a stage's internal argparse never sees
    run_pipeline's own CLI flags.
    """
    logger.info("=" * 70)
    logger.info("STAGE %d: %s", stage_number, stage_name)
    logger.info("=" * 70)
    start = time.time()

    original_argv = sys.argv
    sys.argv = [original_argv[0]] if original_argv else ["run_pipeline"]
    try:
        result = fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        logger.error("STAGE %d (%s) FAILED: %s", stage_number, stage_name, exc)
        raise PipelineStageError(f"Stage {stage_number} ({stage_name}) failed: {exc}") from exc
    finally:
        sys.argv = original_argv

    elapsed = time.time() - start
    logger.info("STAGE %d (%s) complete in %.1fs", stage_number, stage_name, elapsed)
    return result


def run(skip_mysql: bool = False) -> int:
    pipeline_start = time.time()
    logger.info("#" * 70)
    logger.info("# OmniStyle Customer Intelligence Platform - Full Pipeline Run")
    logger.info("#" * 70)

    try:
        # Stage 1: Data ingestion
        from src.data_ingestion import main as ingestion_main
        exit_code = _run_stage(1, "Data Ingestion", ingestion_main)
        if exit_code:
            raise PipelineStageError("Stage 1 (Data Ingestion) returned a non-zero exit code.")

        # Stage 2: Data cleaning
        from src.data_cleaning import main as cleaning_main
        exit_code = _run_stage(2, "Data Cleaning", cleaning_main)
        if exit_code:
            raise PipelineStageError("Stage 2 (Data Cleaning) returned a non-zero exit code.")

        # Stage 3: Feature engineering
        from src.feature_engineering import main as feature_main
        exit_code = _run_stage(3, "Feature Engineering", feature_main)
        if exit_code:
            raise PipelineStageError("Stage 3 (Feature Engineering) returned a non-zero exit code.")

        # Stage 4: MySQL loading (skippable)
        if skip_mysql:
            logger.info("=" * 70)
            logger.info("STAGE 4: MySQL Loading -- SKIPPED (--skip-mysql)")
            logger.info("=" * 70)
        else:
            from src.load_to_mysql import main as mysql_main
            exit_code = _run_stage(4, "MySQL Loading", mysql_main)
            if exit_code:
                raise PipelineStageError(
                    "Stage 4 (MySQL Loading) returned a non-zero exit code. "
                    "If you don't have MySQL configured, re-run with --skip-mysql."
                )

        # Stage 5: Customer segmentation
        from src.customer_segmentation import run_segmentation
        _run_stage(5, "Customer Segmentation", run_segmentation, save_outputs=True)

        # Stage 6: Churn model training (+ SHAP explainability)
        from src.train_model import _load_churn_dataset, train_churn_models, run_shap_explainability

        def _churn_stage():
            churn_bundle = train_churn_models(_load_churn_dataset())
            try:
                run_shap_explainability(churn_bundle)
            except Exception as exc:  # noqa: BLE001
                logger.warning("SHAP explainability failed (%s). Continuing without it.", exc)
            return churn_bundle

        _run_stage(6, "Churn Model Training", _churn_stage)

        # Stage 7: Business recommendations
        from src.generate_recommendations import main as recommendations_main
        exit_code = _run_stage(7, "Business Recommendations", recommendations_main)
        if exit_code:
            raise PipelineStageError("Stage 7 (Business Recommendations) returned a non-zero exit code.")

        # Stage 8: Power BI export
        from src.export_dashboard_data import main as export_main
        exit_code = _run_stage(8, "Power BI Export", export_main)
        if exit_code:
            raise PipelineStageError("Stage 8 (Power BI Export) returned a non-zero exit code.")

    except PipelineStageError as exc:
        logger.error("Pipeline halted: %s", exc)
        return 1

    total_elapsed = time.time() - pipeline_start
    logger.info("#" * 70)
    logger.info("# Pipeline complete in %.1fs. See data/, models/, reports/, and dashboard/sample_data/.", total_elapsed)
    logger.info("#" * 70)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the full OmniStyle Customer Intelligence pipeline end to end."
    )
    parser.add_argument(
        "--skip-mysql",
        action="store_true",
        help="Skip the MySQL loading stage (useful if MySQL is not configured locally).",
    )
    args = parser.parse_args()
    return run(skip_mysql=args.skip_mysql)


if __name__ == "__main__":
    sys.exit(main())
