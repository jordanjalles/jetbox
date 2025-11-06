#!/bin/bash
# Quick launcher for workspace reuse and WTN evaluation

set -e

echo "========================================"
echo "Workspace Reuse & WTN Eval Launcher"
echo "========================================"
echo ""

# Check if orchestrator exists
if [ ! -f "orchestrator_agent.py" ]; then
    echo "ERROR: orchestrator_agent.py not found in current directory"
    exit 1
fi

# Parse command line args
MODE="${1:-quick}"

case "$MODE" in
    quick)
        echo "Mode: QUICK TEST (calculator only, 3 rounds)"
        echo "Estimated time: ~10 minutes"
        echo ""
        python eval_workspace_reuse_and_wtn.py --projects calculator --cleanup
        ;;

    fast)
        echo "Mode: FAST TEST (2 projects, 6 rounds)"
        echo "Estimated time: ~20 minutes"
        echo ""
        python eval_workspace_reuse_and_wtn.py --projects calculator web_scraper --cleanup
        ;;

    full)
        echo "Mode: FULL EVALUATION (5 projects, 15 rounds)"
        echo "Estimated time: ~45-75 minutes"
        echo ""
        read -p "This will take a while. Continue? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            python eval_workspace_reuse_and_wtn.py --cleanup
        else
            echo "Cancelled."
            exit 0
        fi
        ;;

    dry-run)
        echo "Mode: DRY RUN (no execution, just plan)"
        echo ""
        python eval_workspace_reuse_and_wtn.py --dry-run
        ;;

    *)
        echo "Usage: $0 [quick|fast|full|dry-run]"
        echo ""
        echo "Modes:"
        echo "  quick    - Test 1 project (calculator) - ~10 min"
        echo "  fast     - Test 2 projects - ~20 min"
        echo "  full     - Test all 5 projects - ~45-75 min"
        echo "  dry-run  - Show plan without running"
        echo ""
        echo "Default: quick"
        exit 1
        ;;
esac
