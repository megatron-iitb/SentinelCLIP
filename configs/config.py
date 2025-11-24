"""
Configuration settings for Experiment 1: Accountable CLIP-based Classification
"""
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOG_DIR = PROJECT_ROOT / "logs"
CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub"

# Model configuration
MODEL_NAME = "ViT-B-32"
MODEL_PRETRAINED = "openai"
BATCH_SIZE = 256
NUM_WORKERS = 2

# Dataset configuration
DATASET_NAME = "CIFAR10"
VAL_SPLIT = 0.1  # 10% for validation
RANDOM_SEED = 42

# Calibration configuration
TEMPERATURE_INIT = 1.0
TEMPERATURE_MIN = 0.01
TEMPERATURE_MAX = 100.0
TEMPERATURE_PATIENCE = 10
TEMPERATURE_MAX_EPOCHS = 500
TEMPERATURE_LR = 0.01

# Conformal prediction
CONFORMAL_ALPHA = 0.10  # 90% coverage target

# Augmentation ensemble
AUGMENTATION_SCALES = [0.9, 0.8]  # Crop scales

# Question prompts for semantic reasoning
QUESTION_PROMPTS = {
    "Q1_man_made": (
        "A photo of a man-made manufactured object.",
        "A photo of a natural living thing or animal."
    ),
    "Q2_wings": (
        "There are clear wings or flight surfaces visible.",
        "No wings or flight surfaces visible."
    ),
    "Q3_watercraft": (
        "This is a watercraft/ship or the object is clearly on water.",
        "Not a watercraft; not clearly on water."
    ),
    "Q4_wheels": (
        "Road wheels/tires or wheel assemblies are visible.",
        "No visible wheels or tires."
    ),
    "Q5_amphibian_like": (
        "Looks like a small smooth-bodied amphibian (frog-like).",
        "Not a frog/amphibian-like."
    ),
    "Q6_hooves": (
        "Hooves or hoof-like feet are visible.",
        "No hooves visible."
    ),
    "Q7_feline_signature": (
        "Feline features: short muzzle, pointed ears, visible whiskers.",
        "No feline facial signature visible."
    ),
    "Q8_cargo_bed": (
        "Vehicle has an open cargo bed or truck-like cargo area.",
        "No cargo bed visible (not a truck-bed)."
    ),
    "Q9_mane_equine": (
        "Horse-like mane or equine neck/pose is visible.",
        "No mane/equine features visible."
    ),
}

# CIFAR-10 specific: Question ground truth mappings for isotonic calibration
# Classes: airplane(0), automobile(1), bird(2), cat(3), deer(4), dog(5), frog(6), horse(7), ship(8), truck(9)
QUESTION_GT_MAP = {
    'Q1_man_made': [0, 1, 8, 9],  # airplane, automobile, ship, truck
    'Q2_wings': [0, 2],  # airplane, bird
    'Q3_watercraft': [8],  # ship
    'Q4_wheels': [1, 9],  # automobile, truck
    'Q5_amphibian_like': [6],  # frog
    'Q6_hooves': [4, 7],  # deer, horse
    'Q7_feline_signature': [3],  # cat
    'Q8_cargo_bed': [9],  # truck
    'Q9_mane_equine': [7],  # horse
}

# Non-critical questions (used for additional context, not safety-critical)
NONCRITICAL_QUESTIONS = [
    'Q2_wings', 'Q4_wheels', 'Q5_amphibian_like', 
    'Q6_hooves', 'Q7_feline_signature', 'Q8_cargo_bed', 'Q9_mane_equine'
]

# Policy thresholds (can be optimized on validation)
TAU_CRITICAL_LOW = 0.45  # Minimum critical signal threshold
ACTION_AUTO_DEFAULT = 0.90  # Auto-execute threshold
ACTION_CLARIFY_DEFAULT = 0.60  # Clarification threshold

# Threshold optimization
OPTIMIZE_THRESHOLDS = True
THRESHOLD_COST_HUMAN = 1.0  # Cost per human intervention
THRESHOLD_COST_ERROR = 10.0  # Cost per error
THRESHOLD_GRID_SIZE = 5  # Grid resolution (n^3 combinations)
VAL_AUGMENTATION_SUBSET = 1000  # Limit val augmentations for speed

# Simulation settings
SIM_HUMAN_ACCURACY = 1.0  # Perfect humans by default (set < 1.0 for realistic)

# Evaluation
ECE_BINS = 15  # Expected Calibration Error bins
RELIABILITY_BINS = 10  # Reliability diagram bins

# Visualization
FIGURE_DPI = 150
SAVE_PLOTS = True

# Ensemble confidence strategy
# Options: "std", "entropy", "combined", "geometric"
ENSEMBLE_CONF_STRATEGY = "combined"

# Device
DEVICE = "cuda"  # Will auto-detect if available
