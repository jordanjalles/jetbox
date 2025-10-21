# Overnight Autonomous Testing - Quick Start

## What This Does

Runs ~8 hours of autonomous agent testing while you sleep:
- **Hours 0-2:** Extended stress tests (L3-L5) - 10 runs each
- **Hour 2-3:** L6 extreme challenges - 5 runs each
- **Hours 3-6:** HRM-JEPA development tasks
- **Hours 6-7:** L7-L8 algorithmic/system design tests
- **Hours 7-8:** Analysis, statistics, and report generation

**Output:** Comprehensive pass/fail logs, statistics, and improvement recommendations

## Prerequisites

1. **Ollama running** with gpt-oss:20b model
2. **At least 5GB free disk space**
3. **Python 3.11+** with pytest and ruff installed
4. **Conda environment activated** (if using)

## Quick Start

### Option 1: Simple Start (Recommended)

```bash
# Start overnight testing (will run for ~8 hours)
python run_overnight.py
```

### Option 2: Background Start (Linux/WSL)

```bash
# Start in background and log to file
nohup python run_overnight.py > overnight.log 2>&1 &

# Check progress
tail -f overnight.log

# Check if still running
ps aux | grep run_overnight
```

### Option 3: Screen/Tmux (Recommended for SSH)

```bash
# Using screen
screen -S overnight
python run_overnight.py
# Detach: Ctrl+A, then D
# Reattach later: screen -r overnight

# Using tmux
tmux new -s overnight
python run_overnight.py
# Detach: Ctrl+B, then D
# Reattach later: tmux attach -t overnight
```

## What Gets Created

```
overnight_results/
├── master_log.json              # Overall statistics
├── overnight_report_YYYY-MM-DD.md  # Comprehensive analysis
├── runs/                        # Individual test results (JSON)
│   ├── L3-1-run-0001.json
│   ├── L3-1-run-0002.json
│   └── ...
├── logs/                        # Detailed output logs
│   ├── L3-1-run-0001.txt
│   ├── L3-1-run-0001-ledger.log
│   └── ...
├── plots/                       # Visualizations (if generated)
│   └── ...
└── workspaces/
    └── failed/                  # Snapshots of failed test workspaces
```

## Monitoring Progress

### Check Overall Status

```bash
# Count completed runs
ls overnight_results/runs/ | wc -l

# Check last few results
ls -lt overnight_results/runs/ | head -10

# View current statistics
cat overnight_results/master_log.json | python -m json.tool
```

### Check Specific Test

```bash
# Find all runs of a test
ls overnight_results/runs/ | grep "L3-1"

# View latest run of L3-1
cat overnight_results/runs/L3-1-run-*.json | tail -1 | python -m json.tool

# View output log
cat overnight_results/logs/L3-1-run-0001.txt | tail -50
```

### Monitor in Real-Time

```bash
# Watch results directory
watch -n 10 'ls overnight_results/runs/ | wc -l'

# Monitor disk usage
watch -n 30 'df -h . | tail -1'
```

## Expected Timeline

| Time | Phase | Activity |
|------|-------|----------|
| 0:00 | Start | Environment validation |
| 0:00-2:00 | Phase 1 | L3-L5 tests (10x each) ~90 tests |
| 2:00-3:00 | Phase 2 | L6 extreme tests (5x each) ~30 tests |
| 3:00-6:00 | Phase 3 | HRM-JEPA development (6 tasks) |
| 6:00-7:00 | Phase 4 | L7-L8 tests (3x each) ~18 tests |
| 7:00-8:00 | Phase 5 | Analysis and report generation |
| 8:00 | Complete | Report ready in `overnight_results/` |

**Total expected:** ~145 test runs + 6 HRM tasks + analysis

## Stopping Early

If you need to stop:

```bash
# Find the process
ps aux | grep run_overnight

# Kill it
kill <PID>

# Or use Ctrl+C if in foreground
```

The script is designed to be crash-resilient. Partial results are saved as tests complete.

## Reading the Report

After completion, read:

```bash
# Main report
cat overnight_results/overnight_report_*.md | less

# Quick summary
head -50 overnight_results/overnight_report_*.md

# Statistics
cat overnight_results/master_log.json | python -m json.tool | less
```

## What to Look For

### In the Report

1. **Overall Pass Rate** - Should improve from baseline 67%/89%
2. **Flaky Tests** - Tests with inconsistent results (need investigation)
3. **Failure Modes** - Common failure patterns
4. **HRM-JEPA Progress** - Which development tasks completed
5. **Top Recommendations** - Action items for improvement

### Key Metrics

- **Pass rate by level** - Shows agent capability at each difficulty
- **Test stability** - Identifies unreliable tests
- **Performance** - Duration and round counts
- **Failure modes** - Most common issues

## Troubleshooting

### Script Won't Start

```bash
# Check Python
python --version  # Should be 3.11+

# Check dependencies
pip install pytest ruff

# Check Ollama
ollama list  # Should show gpt-oss:20b
```

### Out of Disk Space

```bash
# Clean up old workspaces
rm -rf .agent_workspace/
rm -rf .agent_context/

# Remove old results
rm -rf overnight_results/  # If from previous run
```

### Tests Failing Immediately

```bash
# Verify base tests work
python run_stress_tests.py 1  # Run L1 tests

# Check agent works
python agent.py "Create hello.py with hello world"
```

### Ollama Not Responding

```bash
# Restart Ollama
ollama serve  # In a separate terminal

# Or restart Ollama service (Windows)
# Restart from Ollama system tray icon
```

## Advanced Options

### Run Specific Phases Only

Edit `run_overnight.py` and comment out phases you don't want:

```python
# Skip L3-L5 tests
# if l3_tests:
#     results = run_test_batch(l3_tests, repetitions=10, phase_name="L3 Extended (x10)")
#     all_results.extend(results)
```

### Adjust Repetitions

Edit test counts in the script:

```python
# Reduce from 10 to 5 repetitions
results = run_test_batch(l3_tests, repetitions=5, phase_name="L3 Extended (x5)")
```

### Run Only HRM-JEPA Tasks

```bash
# Edit run_overnight.py to skip test phases
# Keep only the HRM-JEPA section
```

## What Happens Next Morning

1. **Check completion:**
   ```bash
   ls overnight_results/overnight_report_*.md
   ```

2. **Read the report:**
   ```bash
   cat overnight_results/overnight_report_*.md
   ```

3. **Review recommendations** in the report

4. **Check for quick wins:**
   - Consistently failing tests → need fixes
   - Flaky tests → need investigation
   - HRM-JEPA tasks → continue development

5. **Commit results** (optional):
   ```bash
   git add overnight_results/
   git commit -m "Add overnight testing results"
   ```

## Expected Deliverables

✓ **145+ test runs** across L3-L8
✓ **6 HRM-JEPA development tasks** attempted
✓ **Statistical analysis** of agent performance
✓ **Failure pattern analysis**
✓ **Improvement recommendations**
✓ **Workspace snapshots** of failed tests

## Support

If something goes wrong, check:
1. `overnight_results/master_log.json` - Overall status
2. Last few entries in `overnight_results/runs/` - Recent test results
3. Agent logs: `agent_v2.log` and `agent_ledger.log`

Good luck! Check back in ~8 hours for comprehensive results.
