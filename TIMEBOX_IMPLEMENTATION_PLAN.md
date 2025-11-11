# TimeBox Implementation Plan

## Goal
Add automatic temporal awareness to agents through factual time nudges and agent-scheduled reminders.

## Core Implementation

### TimeBoxBehavior (~80 lines)
**File**: `behaviors/timebox.py`

**Features**:
- Auto-inject factual nudges at 25%, 50%, 75% (configurable)
- Tool: `schedule_reminder(at_percent, message)` for agent self-programming
- Neutrally toned generic defaults
- Dual tracking: wall-clock (if budget set) OR rounds (fallback)
- NO cross-session persistence - wall clock is absolute

**Key Design**:
- Orchestrator gets 240 min budget (absolute, regardless of sub-agents)
- Creative tasks encouraged to think broader when MORE time available
- Tool description contains "Temporal Self-Management" guidance

### Configuration
- `task_executor.yaml`: 60 min budget
- `orchestrator.yaml`: 240 min budget
- Default nudges: [25, 50, 75]

## Testing Strategy
1. Simple software task (calculator, 30 min)
2. Creative task (writing, 45 min) - observe if agent thinks broader
3. Research task (planning, 30 min)
4. Orchestrator delegation (observe absolute timing)

## Success Criteria
- Factual nudges inject automatically
- Agents schedule custom reminders
- Reminders trigger at correct percentages
- Different domain responses observed
- No breaking changes

## Implementation Steps
1. Create `behaviors/timebox.py`
2. Add to agent configs (task_executor, orchestrator)
3. Test simple task
4. Test creative task
5. Test orchestrator
6. ONLY document if everything works perfectly

## Files
- **New**: `behaviors/timebox.py`
- **Modified**: `config/agents/task_executor.yaml`
- **Modified**: `config/agents/orchestrator.yaml`
- **Modified**: `BEHAVIORS_DOCUMENTATION.md` (if successful)
