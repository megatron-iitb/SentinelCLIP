# Pipeline modules for Experiment 1
"""
Pipeline modules organize the experiment workflow into logical stages:
- data_preparation: Model & data loading, embedding extraction
- calibration: Temperature, questions, conformal prediction
- uncertainty: Ensemble and entropy-based confidence
- policy: Threshold optimization and decision-making
- evaluation: Metrics, visualization, and audit logging
"""

from .data_preparation import setup_model_and_data
from .calibration import run_calibration_pipeline
from .uncertainty import run_uncertainty_estimation
from .policy import run_policy_pipeline
from .evaluation import run_evaluation

__all__ = [
    'setup_model_and_data',
    'run_calibration_pipeline',
    'run_uncertainty_estimation',
    'run_policy_pipeline',
    'run_evaluation'
]
