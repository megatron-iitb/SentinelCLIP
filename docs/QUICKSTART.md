# Quick Start Guide: Running Improved Experiment 1

## Overview
The improved `experiment_1.py` contains 6 major enhancements (A, B, C + 3 robustness fixes). This guide explains how to run it.

## Important Note
**The code is designed for Google Colab/Jupyter notebooks** and includes cell markers and `!pip` commands. You have two options:

### Option 1: Run in Google Colab (Recommended)
1. Upload `experiment_1.py` to Google Colab
2. Convert to notebook format or run cells sequentially
3. The code will automatically install dependencies with `!pip`

### Option 2: Convert to Standalone Script
Remove the Colab-specific lines (lines with `!pip` and cell markers).

## Quick Syntax Validation ✓

All improvements have been validated for Python syntax:
```bash
✓ Core Python syntax valid
✓ Temperature grid fallback added
✓ Log-space geometric mean implemented  
✓ Simulated human accuracy integrated
✓ Threshold optimizer (Feature A) added
✓ Isotonic calibration (Feature B) added
✓ Entropy ensemble (Feature C) added
```

## What Was Added

### 1. Temperature Scaling Robustness (~30 lines)
- Grid-search fallback when optimizer hits bounds
- More reliable temperature selection
- Location: After `fit_temperature()` function

### 2. Numerically Stable Geometric Mean (1 line change)
- Log-space computation prevents NaN/overflow
- Location: Line ~565 in action confidence computation

### 3. Simulated Human Accuracy (~20 lines)
- Realistic human error simulation
- Configurable via `SIM_HUMAN_ACCURACY` variable
- Location: Top of file and in decision loop

### 4. Feature A: Threshold Optimizer (~120 lines)
- Validates and optimizes policy thresholds
- Grid-searches over validation set
- Minimizes `cost = c_human * interventions + c_error * errors`
- Location: Before test set evaluation

### 5. Feature B: Question Calibration (~80 lines)
- Isotonic regression per question prompt
- Domain-specific ground-truth mapping for CIFAR-10
- Location: After question confidence computation

### 6. Feature C: Entropy Ensemble (~60 lines)
- Mutual information proxy for disagreement
- More principled uncertainty quantification
- Location: After augmentation ensemble

## Configuration Parameters

### In the Code (Top Section)
```python
# Line ~33
SIM_HUMAN_ACCURACY = 1.0  # Change to 0.95 for 95% human accuracy

# Line ~542 (in optimize_policy_thresholds call)
c_human = 1.0   # Cost per intervention
c_error = 10.0  # Cost per error
n_grid = 5      # Resolution (5^3 = 125 combinations)

# Line ~522 (ensemble confidence choice)
ens_conf = 0.5 * ens_conf_std + 0.5 * ens_conf_entropy  # Combined
# OR:
# ens_conf = ens_conf_entropy  # Entropy-based only
```

## Expected Runtime

On a typical setup (GPU-enabled Colab):
- Temperature fitting: ~10 seconds
- Question calibration: ~5 seconds  
- Augmentation ensemble (test): ~3-5 minutes
- Threshold optimization (val): ~2-3 minutes
- **Total: ~10-15 minutes**

## Output Explained

### 1. Temperature Fitting
```
Learned temperature T = 1.2345
```
Or if fallback triggered:
```
Temperature hit clamp bounds; running grid-search fallback...
Grid-search found T = 2.1234; using that value
```

### 2. Question Calibration
```
Calibrated 8/9 questions
Mean question confidence before: 0.6543
Mean question confidence after: 0.6789
```

### 3. Ensemble Confidence
```
Std-based ensemble conf: mean=0.8123, std=0.1234
Entropy-based ensemble conf: mean=0.7956, std=0.0987
Correlation: 0.8234
Using combined ensemble confidence: mean=0.8040
```

### 4. Threshold Optimization (NEW!)
```
Best thresholds found:
  tau_critical_low: 0.4500
  ACTION_AUTO: 0.8500
  ACTION_CLARIFY: 0.6500
  Cost: 1234.56 (interventions=456, errors=78)
  Val accuracy: 0.9234
```

### 5. Final Test Results
```
Baseline (CLIP calibrated) accuracy: 0.8617
Post-pipeline accuracy (simulated human): 0.9782 (or lower if SIM_HUMAN_ACCURACY < 1.0)
Action coverage: auto=0.45 clarify=0.35 human=0.20
Errors baseline: 1383 after: 218 prevented: 1165
Intervention precision: 0.42 recall: 0.95
```

## Interpreting Results

### Good Signs
- ✅ Temperature in range [0.1, 10.0]
- ✅ ECE < 0.1 (well-calibrated)
- ✅ Most questions calibrated successfully
- ✅ Intervention recall > 0.9 (catches most errors)
- ✅ Post-pipeline accuracy > baseline accuracy

### Potential Issues
- ⚠️ Temperature hits bound (0.01 or 100) → grid-search fallback activates
- ⚠️ Few questions calibrated (<5) → check ground-truth mappings
- ⚠️ Low intervention precision (<0.2) → too many false alarms; increase thresholds
- ⚠️ High human fraction (>0.5) → policy too conservative; adjust costs

## Comparison With Original Run

### Original Results (from results.md)
```
Learned temperature T = 0.01 (hit lower bound)
ECE: 0.0425
Avg conformal set size: 1.11
Coverage: auto=0.0, clarify=0.71, human=0.29
Post-pipeline accuracy: 0.9782 (perfect humans)
Intervention precision: 0.14, recall: 1.0
```

### Expected Improved Results
With improvements, you should see:
1. **Better temperature**: T in [0.5, 5.0] range (not hitting bounds)
2. **Calibrated questions**: 7-9 questions successfully calibrated
3. **Optimized thresholds**: Data-driven, not hand-tuned
4. **Balanced policy**: Some auto-execution (>0%), less over-intervention
5. **Realistic accuracy**: Lower if SIM_HUMAN_ACCURACY < 1.0
6. **Higher precision**: >0.20 with similar recall

## Troubleshooting

### Issue: "Temperature hit clamp bounds"
**Solution**: This is expected! The grid-search fallback will activate automatically. If it persists, the logit scale may need adjustment.

### Issue: "Could not fit isotonic for Q_xyz"
**Solution**: Not enough positive examples for that question. Check `question_gt_map` mappings. Non-fatal - code continues.

### Issue: "All decisions forced to human"
**Solution**: Check `tau_critical_low` - may be too high. Threshold optimizer should fix this automatically.

### Issue: "Correlation between std and entropy is negative"
**Solution**: Check augmentation implementation. Both should agree on uncertainty direction.

## Running Experiments

### Experiment 1: Baseline with Perfect Humans
```python
SIM_HUMAN_ACCURACY = 1.0
c_human = 1.0
c_error = 10.0
```

### Experiment 2: Realistic Human Performance
```python
SIM_HUMAN_ACCURACY = 0.95  # 95% human accuracy
c_human = 1.0
c_error = 10.0
```

### Experiment 3: High Intervention Cost
```python
SIM_HUMAN_ACCURACY = 1.0
c_human = 5.0   # Interventions are expensive
c_error = 10.0
```

### Experiment 4: Safety-Critical (Low Error Tolerance)
```python
SIM_HUMAN_ACCURACY = 1.0
c_human = 1.0
c_error = 50.0  # Errors are very costly
```

## Files Generated

After running, you'll find in `/content/clip_accountable_experiment/`:
- `reliability_diagram.png` - Calibration plot
- `coverage_vs_error.png` - Policy trade-off curve
- `audit_log_test.jsonl` - Per-sample decisions and confidences

## Next Steps

1. **Run baseline**: Default settings to establish improved performance
2. **Sweep parameters**: Try different cost ratios and human accuracy
3. **Analyze audit log**: Inspect per-sample decisions for patterns
4. **Extend to other datasets**: Modify question GT mappings
5. **Add visualization**: Plot threshold optimization results (Pareto frontier)

## Summary

All improvements have been implemented and validated for syntax. The code is ready to run in Google Colab or Jupyter. Key improvements:

✅ Temperature scaling robustness (grid fallback)  
✅ Numerical stability (log-space geometric mean)  
✅ Realistic evaluation (simulated human errors)  
✅ (A) Validation-driven threshold optimization  
✅ (B) Per-question isotonic calibration  
✅ (C) Entropy-based ensemble confidence  

**Total additions**: ~310 lines of code, 6 major functions

For detailed documentation, see `IMPROVEMENTS.md`.
