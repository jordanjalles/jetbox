# Repository Organization Guide

**NOTE TO CLAUDE:** The user prefers to keep the repo clean and scannable by organizing analysis and implementation documents into subfolders.

## Folder Structure

### `/docs/analysis/`
**Purpose:** Test results, failure analysis, debugging reports, and evaluation data

**When to use:** Any document analyzing test results, failures, or performance
- Test failure analyses
- Evaluation reports
- Debugging investigations
- Root cause analyses
- Timeout/performance analysis

### `/docs/implementation/`
**Purpose:** Implementation details, design proposals, feature summaries, and fix documentation

**When to use:** Documents describing HOW something was implemented
- Feature implementation summaries
- Fix proposals and implementations
- Design documents
- "Before/After" documentation
- Version-specific changes

### `/docs/architecture/`
**Purpose:** System architecture, component design, and reference documentation

**When to use:** Documents describing system structure and design
- Architecture overviews
- Component documentation (STATUS_DISPLAY.md, CONFIG_SYSTEM.md, etc.)
- API documentation
- Design patterns
- System diagrams

### Root Level
**Keep at root:**
- `README.md` - Main project documentation
- `CLAUDE.md` - Instructions for AI assistants
- `QUICK_START.md` - Quick reference (optional)
- `LICENSE`, `CONTRIBUTING.md`, etc.

**DO NOT keep at root:**
- Analysis reports
- Implementation details
- Failure investigations
- Evaluation results
- Version-specific documentation

## Guidelines

1. **When creating new analysis/reports:** Immediately place in appropriate `/docs/` subfolder
2. **File naming:** Use descriptive names (e.g., `TIMEOUT_FIX_ANALYSIS.md` not `analysis.md`)
3. **Keep root clean:** Only essential user-facing docs at root level
4. **Commit organization:** Group file moves in single "Organize documentation" commit

## Example Organization

```
✅ GOOD:
/README.md
/CLAUDE.md
/docs/analysis/FAILURE_ANALYSIS.md
/docs/implementation/TIMEOUT_FIX_IMPLEMENTED.md

❌ BAD:
/README.md
/CLAUDE.md
/FAILURE_ANALYSIS.md
/TIMEOUT_FIX_IMPLEMENTED.md
/EVAL_RESULTS.md
```

## When in Doubt

If creating a new document and unsure where it goes:
1. Is it user-facing setup/usage? → Root level
2. Is it analyzing test results? → `/docs/analysis/`
3. Is it describing an implementation? → `/docs/implementation/`
4. Is it describing system design? → `/docs/architecture/`

**User preference:** Clean and scannable root directory with organized subfolders for deep-dive content.
