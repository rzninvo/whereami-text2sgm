"""Localization subpackage for dense-grid camera localization.

Public API re-exports for convenient access::

    from whereami.localization import evaluate_scene, EvalMode, SceneMetrics
"""
from whereami.localization.evaluation import EvalMode, evaluate_scene, run_evaluation
from whereami.localization.metrics import SceneMetrics
from whereami.localization.grid import load_scene, sample_grid, first_hit_is_object
from whereami.localization.matching import topk_matched_objects
from whereami.localization.pipeline import run_loc_pipeline
from whereami.localization.prediction import select_prediction_point
from whereami.localization.frame_io import FrameSelection

__all__ = [
    "EvalMode",
    "evaluate_scene",
    "run_evaluation",
    "SceneMetrics",
    "FrameSelection",
    "load_scene",
    "sample_grid",
    "first_hit_is_object",
    "topk_matched_objects",
    "run_loc_pipeline",
    "select_prediction_point",
]
