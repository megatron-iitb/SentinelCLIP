# Error Documentation & Troubleshooting Guide

**Comprehensive reference for common errors, warnings, and their resolutions.**

---

## 📋 Error Categories

1. [Model Loading Errors](#model-loading-errors)
2. [Data Loading Errors](#data-loading-errors)
3. [Calibration Errors](#calibration-errors)
4. [SLURM Job Errors](#slurm-job-errors)
5. [Dependency Errors](#dependency-errors)
6. [Performance Warnings](#performance-warnings)

---

## 🔴 Model Loading Errors

### Error 1: FileNotFoundError - Model Not Found

**Symptom**:
```
FileNotFoundError: [Errno 2] No such file or directory: '/content/clip_model/...'
```

**Cause**: Colab-specific paths hardcoded when running on HPC cluster.

**Fix**:
```python
# In config.py, change:
DATA_DIR = "/content/data"          # ❌ Wrong
DATA_DIR = "./data"                 # ✅ Correct

MODEL_CACHE = "/content/models"     # ❌ Wrong  
MODEL_CACHE = None  # Use default   # ✅ Correct
```

**Prevention**: Always use relative paths or environment variables.

---

### Error 2: OSError - Cannot Download Model (Offline Mode)

**Symptom**:
```
OSError: We couldn't connect to 'https://huggingface.co' to load this model.
You are offline or have no internet connection.
```

**Cause**: Running in offline mode (`HF_HUB_OFFLINE=1`) without pre-cached model.

**Fix**:
```bash
# Step 1: On login node (with internet), download model
python3 -c "
import open_clip
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
print('✓ Model cached at:', model.visual.conv1)
"

# Step 2: Verify cache location
ls -lh ~/.cache/huggingface/hub/models--timm*

# Step 3: Now safe to run offline
export HF_HUB_OFFLINE=1
python run_experiment.py
```

**Prevention**: Add model download step to setup script.

---

### Error 3: RuntimeError - CUDA Out of Memory

**Symptom**:
```
RuntimeError: CUDA out of memory. Tried to allocate 2.34 GiB
```

**Cause**: Batch size too large for GPU memory.

**Fix**:
```python
# In config.py
BATCH_SIZE = 128  # Reduce from 256
N_AUGMENTATIONS = 8  # Reduce from 16

# Or use gradient accumulation (advanced)
```

**Memory Usage Reference**:
| GPU | Max Batch Size (ViT-B-32) |
|-----|---------------------------|
| T4 (16GB) | 128 |
| L40 (48GB) | 512 |
| A100 (80GB) | 1024 |

---

## 🟡 Data Loading Errors

### Error 4: Dataset Download Fails

**Symptom**:
```
URLError: <urlopen error [Errno 111] Connection refused>
```

**Cause**: No internet access when CIFAR-10 tries to download.

**Fix**:
```bash
# On login node (with internet):
python3 -c "
from torchvision.datasets import CIFAR10
CIFAR10(root='./data', train=True, download=True)
CIFAR10(root='./data', train=False, download=True)
print('✓ CIFAR-10 downloaded to ./data')
"

# Verify
ls -lh data/cifar-10-batches-py/
```

**Expected Size**: ~170 MB (all splits)

---

### Error 5: Corrupted Dataset

**Symptom**:
```
RuntimeError: invalid header or archive is corrupted
```

**Cause**: Incomplete download or disk corruption.

**Fix**:
```bash
# Remove corrupted files
rm -rf data/cifar-10-batches-py*

# Re-download
python3 -c "
from torchvision.datasets import CIFAR10
CIFAR10(root='./data', train=True, download=True)
"

# Verify MD5
md5sum data/cifar-10-python.tar.gz
# Should match: c58f30108f718f92721af3b95e74349a
```

---

## 🟠 Calibration Errors

### Error 6: Temperature Hits Bounds

**Symptom**:
```
Learned temperature T = 0.0100 (log_T = -4.6052)
Temperature hit lower bound 0.01. Using grid-search fallback...
```

**Status**: ⚠️ **Not an error** - this is expected behavior!

**Interpretation**:
- Model probabilities are already well-calibrated or over-confident
- Grid-search automatically activates to verify optimal T
- If grid-search returns same value, no issue

**When to investigate**:
- If T consistently hits upper bound (100.0) → model under-confident
- If grid-search finds significantly different T → optimizer failed

**Fix** (if needed):
```python
# In config.py, adjust bounds
TEMPERATURE_MIN = 0.001  # Allow sharper
TEMPERATURE_MAX = 50.0   # Lower ceiling

# Or adjust optimizer
TEMPERATURE_LR = 0.001   # Lower from 0.01
TEMPERATURE_MAX_EPOCHS = 1000  # More iterations
```

---

### Error 7: Isotonic Regression Fails

**Symptom**:
```
ValueError: X must be sorted in increasing order
```

**Cause**: Validation similarities have identical values (no variance).

**Fix**:
```python
# In src/calibration/isotonic.py, add check:
if len(np.unique(val_sims)) < 2:
    print(f"Warning: Question '{question}' has no variance, skipping")
    calibrated_funcs.append(None)
    continue

# Use identity function as fallback
from scipy.interpolate import interp1d
calibrated_funcs.append(interp1d([0, 1], [0, 1]))
```

**Prevention**: Ensure validation set has diverse examples per question.

---

### Error 8: Negative Conformal Set Size

**Symptom**:
```
RuntimeError: Found conformal set with -1 labels (impossible!)
```

**Cause**: Bug in quantile calculation or threshold application.

**Fix**:
```python
# In src/calibration/conformal.py, add debug:
print(f"Scores percentile: {np.percentile(scores, [0, 50, 100])}")
print(f"Threshold: {threshold}, Min score: {scores.min()}")

# Ensure threshold is reasonable
assert 0 <= threshold <= 1, f"Invalid threshold: {threshold}"
```

---

## 🔵 SLURM Job Errors

### Error 9: Job Fails to Start

**Symptom**:
```
sbatch: error: Batch job submission failed: Invalid account or account/partition combination specified
```

**Cause**: Wrong partition or account specified.

**Fix**:
```bash
# Check available partitions
sinfo

# Check your account
sacctmgr show user $USER withassoc

# Update job_experiment_1.sh
#SBATCH --partition=l40  # Use your partition
#SBATCH --account=<your_account>  # If required
```

---

### Error 10: Job Killed by OOM

**Symptom**:
```
slurmstepd: error: Detected 1 oom-kill event(s) in StepId=40159.batch
```

**Cause**: Job exceeded memory limit (32GB requested, >32GB used).

**Fix**:
```bash
# In job_experiment_1.sh
#SBATCH --mem=64G  # Increase from 32G

# Or reduce batch size in config.py
BATCH_SIZE = 128
```

**Memory Profiling**:
```python
# Add to run_experiment.py
import tracemalloc
tracemalloc.start()
# ... your code ...
current, peak = tracemalloc.get_traced_memory()
print(f"Peak memory: {peak / 1024**3:.2f} GB")
```

---

### Error 11: Module Not Found in SLURM Job

**Symptom**:
```
ModuleNotFoundError: No module named 'open_clip'
```

**Cause**: Conda environment not activated correctly.

**Fix**:
```bash
# In job_experiment_1.sh, ensure proper activation:
source ~/miniconda3/etc/profile.d/conda.sh  # Critical!
conda activate myenv

# Verify before running
which python  # Should show: ~/miniconda3/envs/myenv/bin/python
python -c "import open_clip; print(open_clip.__version__)"
```

---

## 🟢 Dependency Errors

### Error 12: ImportError - sklearn Not Found

**Symptom**:
```
ImportError: No module named 'sklearn'
```

**Fix**:
```bash
conda activate myenv
pip install scikit-learn
```

---

### Error 13: Version Mismatch - PyTorch

**Symptom**:
```
RuntimeError: Detected that PyTorch and torch_geometric versions do not match
```

**Fix**:
```bash
# Check versions
python -c "import torch; print(torch.__version__)"

# Reinstall matching versions
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
```

---

## 🟣 Performance Warnings

### Warning 1: Slow Data Loading

**Symptom**:
```
Extracting embeddings: 0%|          | 0/157 [00:30<?, ?it/s]
```

**Cause**: num_workers=0 or I/O bottleneck.

**Fix**:
```python
# In src/data/dataset.py
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    num_workers=4,      # Increase from 0
    pin_memory=True,    # Add for GPU
    persistent_workers=True  # Reuse workers
)
```

---

### Warning 2: Low Intervention Precision

**Symptom**:
```
Intervention precision: 0.1384 (expected >0.3)
```

**Status**: ⚠️ **Not an error** - indicates policy needs tuning.

**Interpretation**:
- 86% of human interventions were false positives
- System is over-cautious (high recall, low precision)

**Fix**: See [LOG_ANALYSIS.md](LOG_ANALYSIS.md#8-policy-threshold-optimization)
```python
# In config.py
THRESHOLD_COST_ERROR = 3.0  # Reduce penalty
THRESHOLD_AUTO_RANGE = (0.7, 0.9)  # Increase thresholds
```

---

### Warning 3: All Decisions = 'human'

**Symptom**:
```
Action coverage:
  auto:    0.0000
  clarify: 0.0104  
  human:   0.9896
```

**Status**: ⚠️ **System working but not useful** - no automation achieved.

**Root Causes** (in order of likelihood):
1. Low question confidences after calibration
2. Overly conservative threshold optimization
3. Incorrect QUESTION_GT_MAP
4. Cost function imbalance

**Diagnostic Steps**:
```python
# Step 1: Check question confidences
import json
logs = [json.loads(line) for line in open('outputs/audit_log.jsonl')]
q_confs = [log['question_confidences'] for log in logs]
print(f"Q conf: min={min(q_confs)}, mean={np.mean(q_confs)}, max={max(q_confs)}")
# Expected: mean >0.4, if <0.2 → problem with questions

# Step 2: Check thresholds
from configs.config import THRESHOLD_TAU_LOW, THRESHOLD_AUTO, THRESHOLD_CLARIFY
print(f"Thresholds: tau={THRESHOLD_TAU_LOW}, auto={THRESHOLD_AUTO}, clarify={THRESHOLD_CLARIFY}")
# If auto >0.8 → too high

# Step 3: Validate ground truth map
from configs.config import QUESTION_GT_MAP, CIFAR10_CLASSES
for q, classes in QUESTION_GT_MAP.items():
    print(f"{q} → {[CIFAR10_CLASSES[i] for i in classes]}")
# Verify semantic alignment
```

**Fix Priority Order**:
1. **First**: Adjust cost ratio (easiest)
   ```python
   THRESHOLD_COST_ERROR = 3.0  # down from 10.0
   ```
2. **Second**: Constrain threshold search
   ```python
   THRESHOLD_TAU_RANGE = (0.3, 0.6)
   THRESHOLD_AUTO_RANGE = (0.7, 0.9)
   ```
3. **Third**: Fix question map (hardest)
   ```python
   # Review each question's semantic alignment
   QUESTION_GT_MAP = {
       "Is this a vehicle?": [0, 1, 8, 9],  # Validate!
       # ...
   }
   ```

---

## 🔧 Debugging Tools

### Enable Verbose Logging
```bash
# Set environment variable
export LOGLEVEL=DEBUG
python run_experiment.py

# Or modify code
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Interactive Debugging
```python
# Add breakpoint
import pdb; pdb.set_trace()

# Or use ipdb (better)
pip install ipdb
import ipdb; ipdb.set_trace()
```

### Profile Performance
```bash
# Time profiling
python -m cProfile -o profile.stats run_experiment.py
python -c "import pstats; p=pstats.Stats('profile.stats'); p.sort_stats('cumtime').print_stats(20)"

# Memory profiling
pip install memory_profiler
python -m memory_profiler run_experiment.py
```

---

## 📞 Getting Help

### Before Asking for Help

1. ✅ Check this error documentation
2. ✅ Review [LOG_ANALYSIS.md](LOG_ANALYSIS.md)
3. ✅ Search logs: `grep -i "error\|warning" logs/*.log`
4. ✅ Verify environment: `python --version`, `pip list`

### When Reporting Issues

Include:
- **Error message**: Full traceback
- **Log file**: `logs/Exp_1.*.log`
- **Environment**: `conda list > env.txt`
- **Configuration**: `configs/config.py` relevant sections
- **Steps to reproduce**: Minimal example

### Contact

**Anupam Rawat**  
Email: anupam.rawat@iitb.ac.in  
Lab: MEDAL, IIT Bombay

---

## 📚 Additional Resources

- **PyTorch Debugging**: https://pytorch.org/docs/stable/notes/debugging.html
- **SLURM Troubleshooting**: https://slurm.schedmd.com/troubleshoot.html
- **CLIP Issues**: https://github.com/openai/CLIP/issues

---

**Last Updated**: November 24, 2025  
**Version**: 1.0
