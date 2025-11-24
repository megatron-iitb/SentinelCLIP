# Running Experiment 1 Offline

## Prerequisites (Already Completed)

✅ CLIP ViT-B-32 model downloaded to: `~/.cache/huggingface/hub/models--timm--vit_base_patch32_clip_224.openai/`
✅ CIFAR-10 dataset downloaded to: `./data/cifar-10-batches-py/`

## How to Submit the Job

### Option 1: Submit via SLURM (Recommended)
```bash
cd /home/medal/anupam.rawat/Experiment_1
sbatch job_experiment_1.sh
```

### Option 2: Interactive Test (with internet access)
```bash
cd /home/medal/anupam.rawat/Experiment_1
conda activate myenv
python experiment_1.py
```

## Job Configuration

The job script (`job_experiment_1.sh`) is configured with:
- **Partition**: l40
- **GPU**: 1x L40
- **Memory**: 32GB
- **Time limit**: 4 hours
- **CPUs**: 8 cores

### Key Environment Variables Set by Job Script

```bash
HF_HUB_OFFLINE=1          # Force offline mode for HuggingFace
TRANSFORMERS_OFFLINE=1     # Force offline mode for transformers
HF_DATASETS_OFFLINE=1      # Force offline mode for datasets
HF_HUB_CACHE=~/.cache/huggingface/hub  # Use local cache
```

## Monitoring the Job

### Check job status
```bash
squeue -u $USER
```

### View live output
```bash
tail -f /home/medal/anupam.rawat/Experiment_1/Exp_1.*.log
```

### View errors
```bash
tail -f /home/medal/anupam.rawat/Experiment_1/Exp_1.*.err
```

### Cancel job
```bash
scancel <job_id>
```

## Expected Runtime

Based on the code structure:
- Temperature fitting: ~1-2 minutes
- Val embeddings: ~1 minute
- Test embeddings: ~2 minutes
- Question calibration: ~30 seconds
- Augmentation ensemble (test): ~5-8 minutes
- Threshold optimization (val): ~3-5 minutes
- Final evaluation: ~1 minute

**Total estimated time**: ~15-20 minutes on L40 GPU

## Output Files

After completion, check these directories:

### Results
```bash
ls -lh ./clip_accountable_experiment/
```

Should contain:
- `reliability_diagram.png`
- `coverage_vs_error.png`
- `audit_log_test.jsonl`

### Logs
```bash
ls -lh Exp_1.*.log Exp_1.*.err
```

## Configuration Options

To adjust experiment parameters, edit `experiment_1.py`:

```python
# Line ~35: Simulated human accuracy
SIM_HUMAN_ACCURACY = 1.0  # Change to 0.95 for realistic humans

# Line ~682: Threshold optimization costs
c_human = 1.0   # Cost per intervention
c_error = 10.0  # Cost per error
n_grid = 5      # Grid resolution (5^3 = 125 combinations)

# Line ~563: Ensemble confidence type
ens_conf = 0.5 * ens_conf_std + 0.5 * ens_conf_entropy  # Combined (default)
```

## Troubleshooting

### If job fails with "model not found"
The models should already be cached. Verify:
```bash
ls -lh ~/.cache/huggingface/hub/models--timm--vit_base_patch32_clip_224.openai/
```

### If CIFAR-10 not found
Verify dataset:
```bash
ls -lh ./data/cifar-10-batches-py/
```

### Re-download if needed (from login node with internet)
```bash
cd /home/medal/anupam.rawat/Experiment_1
conda activate myenv
python3 -c "
import open_clip
model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
print('CLIP model downloaded')
"
```

## Quick Submit Command

```bash
cd /home/medal/anupam.rawat/Experiment_1 && sbatch job_experiment_1.sh && squeue -u $USER
```

This will submit the job and immediately show its status.
