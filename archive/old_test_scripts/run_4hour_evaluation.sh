#!/bin/bash
# 4-Hour Comprehensive Evaluation Run
# Can be run in background: nohup bash run_4hour_evaluation.sh > eval.log 2>&1 &

echo "========================================="
echo "Starting 4-Hour Jetbox Evaluation"
echo "========================================="
echo "Start time: $(date)"
echo ""

# Run without input prompt
PYTHONPATH=. python3 comprehensive_evaluation.py --no-prompt

echo ""
echo "========================================="
echo "Evaluation Complete"
echo "End time: $(date)"
echo "========================================="
