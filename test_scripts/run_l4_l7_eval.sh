#!/bin/bash

# L4-L7 Evaluation with Context Inspection
# Runs each task twice, saving context inspection reports separately

EVAL_DIR="evaluation_results/l4_l7_context_inspection_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$EVAL_DIR/successful_runs"
mkdir -p "$EVAL_DIR/failed_runs"

echo "========================================================================"
echo "L4-L7 Task Evaluation with Context Inspection"
echo "========================================================================"
echo "Output directory: $EVAL_DIR"
echo ""

# L4 Tasks
L4_TASKS=(
  "rest_api_mock|Create api.py with Flask app having GET /users and POST /users endpoints with in-memory storage"
  "sqlite_manager|Create db.py with Database class: create_table, insert, query, update, delete methods for SQLite"
)

# L5 Tasks
L5_TASKS=(
  "multi_service_system|Create a multi-service system with user_service.py, auth_service.py, and api_gateway.py that coordinates between them"
  "event_driven_system|Build an event-driven system with event_bus.py, event_handlers.py, and event_producer.py"
)

# L6 Tasks
L6_TASKS=(
  "design_patterns|Implement Factory, Observer, and Strategy design patterns in patterns.py with examples"
  "refactor_legacy|Take a monolithic legacy_code.py and refactor it into clean modular architecture"
)

# L7 Tasks
L7_TASKS=(
  "performance_optimization|Optimize slow_algorithm.py using caching, vectorization, and algorithmic improvements"
  "production_api|Build a production-ready REST API with rate limiting, authentication, monitoring, and comprehensive error handling"
)

run_task() {
  local level=$1
  local task_name=$2
  local task_goal=$3
  local run_num=$4

  echo ""
  echo "========================================================================"
  echo "Running: L${level} - ${task_name} (Run ${run_num})"
  echo "========================================================================"
  echo "Goal: ${task_goal}"
  echo ""

  # Create unique workspace
  WORKSPACE=$(mktemp -d -t eval_L${level}_${task_name}_run${run_num}_XXXXXX)
  echo "Workspace: $WORKSPACE"

  # Run agent with context inspection
  cd "$WORKSPACE" || exit 1

  timeout 600 python /workspace/agent.py "${task_goal}" --ContextInspector > "${EVAL_DIR}/L${level}_${task_name}_run${run_num}.log" 2>&1
  EXIT_CODE=$?

  cd /workspace || exit 1

  # Determine success/failure
  if [ $EXIT_CODE -eq 0 ]; then
    STATUS="SUCCESS"
    TARGET_DIR="$EVAL_DIR/successful_runs"
  else
    STATUS="FAILED"
    TARGET_DIR="$EVAL_DIR/failed_runs"
  fi

  echo ""
  echo "Status: $STATUS (exit code: $EXIT_CODE)"

  # Copy context inspection if it exists
  if [ -d "$WORKSPACE/.context_inspection" ]; then
    INSPECTION_DIR="${TARGET_DIR}/L${level}_${task_name}_run${run_num}_inspection"
    cp -r "$WORKSPACE/.context_inspection" "$INSPECTION_DIR"
    echo "✓ Context inspection saved: $INSPECTION_DIR"

    # Generate analysis report
    python tools/analyze_context.py "$INSPECTION_DIR" 2>&1 | tee "${TARGET_DIR}/L${level}_${task_name}_run${run_num}_analysis.log"
  fi

  # Copy agent logs
  [ -f "$WORKSPACE/agent.log" ] && cp "$WORKSPACE/agent.log" "${TARGET_DIR}/L${level}_${task_name}_run${run_num}_agent.log"
  [ -f "$WORKSPACE/agent_ledger.log" ] && cp "$WORKSPACE/agent_ledger.log" "${TARGET_DIR}/L${level}_${task_name}_run${run_num}_ledger.log"
  [ -d "$WORKSPACE/.agent_context" ] && cp -r "$WORKSPACE/.agent_context" "${TARGET_DIR}/L${level}_${task_name}_run${run_num}_context"

  # Keep workspace for debugging
  echo "Workspace preserved: $WORKSPACE"
  echo "$WORKSPACE" >> "$EVAL_DIR/workspaces.txt"
}

# Run first 2 L4 tasks (x2 runs each = 4 total)
echo ""
echo "Starting evaluation: First 2 L4 tasks..."
echo ""

task_count=0
for task_spec in "${L4_TASKS[@]}"; do
  IFS='|' read -r task_name task_goal <<< "$task_spec"

  for run in 1 2; do
    task_count=$((task_count + 1))
    run_task 4 "$task_name" "$task_goal" "$run"

    # Check-in after 2 problems
    if [ $task_count -eq 2 ]; then
      echo ""
      echo "========================================================================"
      echo "CHECK-IN: First 2 problems complete"
      echo "========================================================================"
      echo "Successful runs: $(ls -1 $EVAL_DIR/successful_runs/*_inspection 2>/dev/null | wc -l)"
      echo "Failed runs: $(ls -1 $EVAL_DIR/failed_runs/*_inspection 2>/dev/null | wc -l)"
      echo ""
      echo "Press Enter to continue, or Ctrl+C to stop..."
      read -r
    fi
  done
done

echo ""
echo "========================================================================"
echo "EVALUATION COMPLETE"
echo "========================================================================"
echo "Results directory: $EVAL_DIR"
echo "Successful runs: $(ls -1 $EVAL_DIR/successful_runs/*_inspection 2>/dev/null | wc -l)"
echo "Failed runs: $(ls -1 $EVAL_DIR/failed_runs/*_inspection 2>/dev/null | wc -l)"
echo ""
