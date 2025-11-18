# Jetbox Strategic Roadmap - Elaborated & Prioritized

**Date**: 2025-11-18
**Source**: Expansion of "plan the plan.txt" themes
**Method**: Effort/Risk/Value analysis with implementation steps

---

## Executive Summary

This document analyzes 6 strategic themes for Jetbox development, breaking down implementation steps and prioritizing by:
- **Effort**: Development complexity and time (1-5, higher = more effort)
- **Risk**: Technical/architectural risk (1-5, higher = more risky)
- **Value**: Impact on users and ecosystem (1-5, higher = more valuable)
- **Priority Score**: Value / (Effort × Risk) - higher is better

---

## Theme Analysis

### 1. Enable Longer-Term Agent Execution

**Vision**: Agents that run continuously on GPU, working toward goals 24/7 with transparency, reversibility, and compact multi-dimensional summaries.

#### Current State
- Agents designed for crash-resilience (append-only logs, JSON state)
- Workspace task notes provide basic context persistence
- Manual intervention required for long-running tasks
- No built-in monitoring/oversight for multi-day runs

#### Implementation Steps

**Phase 1: Foundation (2-3 weeks)**
1. **Persistent Goal Queue System**
   - Create `goal_queue.json` with prioritized goals
   - Add daemon mode: `python agent.py --daemon --queue goal_queue.json`
   - Goal status: pending → in_progress → completed → failed
   - Auto-retry failed goals with backoff

2. **Enhanced Crash Recovery**
   - Upgrade state.json to include last checkpoint timestamp
   - Add auto-resume on crash detection
   - Implement workspace locking to prevent concurrent edits

3. **Resource Management**
   - Add VRAM monitoring (ollama API or nvidia-smi)
   - Implement graceful degradation (switch to smaller model if VRAM low)
   - Add configurable concurrency limits

**Phase 2: Observability (2-3 weeks)**
4. **Multi-Dimensional Summaries**
   - Top-down: Goal → SubGoals → Completion tree
   - Bottom-up: File changes → Feature implemented → Goal satisfied
   - Logic traces: Decision points, why agent chose X over Y
   - Timeline view: What happened when, with screenshots

5. **Live Monitoring Dashboard** (see Theme 6)
   - Real-time goal progress
   - Token usage / cost tracking
   - File diff viewer
   - Pause/Resume/Cancel controls

**Phase 3: Reversibility (1-2 weeks)**
6. **Git-Based Checkpointing**
   - Auto-commit after each successful subtask
   - Tag format: `jetbox/goal-{slug}/subtask-{id}-{timestamp}`
   - Rollback command: `jetbox rollback --to <tag>`
   - Branch-per-goal isolation

7. **Approval Gates** (for production systems)
   - Configurable approval required before: file writes, git push, API calls
   - Webhook integration for external approval workflows

#### Metrics
- **Effort**: 5/5 (Complex distributed system design)
- **Risk**: 4/5 (Concurrency, data consistency, runaway resource usage)
- **Value**: 5/5 (Game-changer for productivity)
- **Priority Score**: 5 / (5 × 4) = **0.25** (Medium priority despite high value)

#### Dependencies
- Requires Theme 6 (Enhanced TUI) for effective oversight
- Benefits from Theme 3 (LLM flexibility) for cost optimization

---

### 2. Dynamic Deployment of Security-Compliant Agents

**Vision**: Deploy specialized, tightly-scoped agents (Home Assistant, email scanner, etc.) to cloud containers with security validation.

#### Current State
- Rule of Two security system implemented (workspace-centric)
- Agent configs in YAML, behaviors composable
- No deployment tooling or packaging system
- Security validation manual, not automated

#### Implementation Steps

**Phase 1: Agent Packaging (1 week)**
1. **Agent Manifest Format**
   - Create `agent_manifest.yaml` schema:
     - `name`, `description`, `version`
     - `required_behaviors`, `config_overrides`
     - `security_profile`: untrusted_files, sensitive_files, network_access
     - `resource_limits`: max_memory, max_cpu, max_runtime
   - Validation: `jetbox validate-manifest agent_manifest.yaml`

2. **Containerization**
   - Create base Dockerfile:
     - Python + Ollama client (or vLLM)
     - Jetbox core + specified behaviors
     - Security hardening (non-root user, read-only filesystem)
   - Build command: `jetbox build --manifest agent_manifest.yaml --tag email-scanner:v1`

**Phase 2: Security Automation (1-2 weeks)**
3. **Pre-Deployment Security Scan**
   - Automated Rule of Two validation on manifest
   - If [ABC] detected, inject defense-in-depth behaviors automatically
   - Generate security report: "This agent can access network AND sensitive files - prompt injection detector enabled"

4. **Runtime Security Enforcement**
   - AppArmor/SELinux profiles for containers
   - Network policy: whitelist allowed domains
   - Filesystem mounts: only expose required paths

**Phase 3: Deployment & Orchestration (2 weeks)**
5. **Cloud Deployment Tooling**
   - `jetbox deploy --manifest email-scanner.yaml --cloud aws --region us-east-1`
   - Terraform/Pulumi code generation for:
     - ECS/Fargate task (AWS)
     - Cloud Run service (GCP)
     - Container Instance (Azure)
   - Auto-configure secrets (API keys from vault)

6. **Agent Registry**
   - Public registry: `registry.jetbox.dev/agents/<name>`
   - Private registry support (Docker Hub, ECR, GCR)
   - Search/discovery: `jetbox search email-scanner`

#### Metrics
- **Effort**: 3/5 (Existing security system helps, Docker expertise needed)
- **Risk**: 4/5 (Cloud deployments always risky, security critical)
- **Value**: 4/5 (Unlocks new use cases, ecosystem growth)
- **Priority Score**: 4 / (3 × 4) = **0.33** (Medium-high priority)

#### Dependencies
- Builds on existing Rule of Two security system
- Benefits from Theme 3 (vLLM/cloud LLM for cloud deployments)

---

### 3. LLM Provider Flexibility (Ollama → vLLM, Cloud)

**Vision**: Support multiple LLM backends beyond Ollama (vLLM, OpenAI, Anthropic, etc.) for performance and cloud deployment.

#### Current State
- Hardcoded Ollama client in `llm_utils.py`
- Model config in `llm_config.yaml` (only Ollama models)
- No abstraction layer for LLM calls

#### Implementation Steps

**Phase 1: Abstraction Layer (3-4 days)**
1. **LLM Provider Interface**
   - Create `llm_providers/base.py`:
     ```python
     class LLMProvider(ABC):
         @abstractmethod
         def chat(self, messages, tools=None, **kwargs) -> ChatResponse
         @abstractmethod
         def list_models(self) -> List[str]
     ```
   - Migrate Ollama client to `llm_providers/ollama.py`

2. **Configuration Schema**
   - Extend `llm_config.yaml`:
     ```yaml
     provider: ollama  # or vllm, openai, anthropic
     ollama:
       base_url: http://localhost:11434
       model: qwen3:14b
     vllm:
       base_url: http://localhost:8000
       model: Qwen/Qwen2.5-Coder-7B-Instruct
     openai:
       api_key_env: OPENAI_API_KEY
       model: gpt-4o-mini
     ```
   - Load provider dynamically: `provider = get_provider(config['provider'])`

**Phase 2: Provider Implementations (1 week)**
3. **vLLM Provider**
   - Implement `llm_providers/vllm.py`
   - Use OpenAI-compatible API (vLLM supports this)
   - Handle thinking token extraction (if model supports)

4. **Cloud Providers**
   - `llm_providers/openai.py` (OpenAI, Azure OpenAI)
   - `llm_providers/anthropic.py` (Claude)
   - `llm_providers/google.py` (Gemini)
   - Cost tracking: append cost per call to `usage_log.jsonl`

**Phase 3: Advanced Features (3-4 days)**
5. **Multi-Provider Routing**
   - Route by task type:
     - Architect tasks → Claude Sonnet (best reasoning)
     - Code generation → Qwen (best code quality)
     - Simple tasks → GPT-4o-mini (cheapest)
   - Config:
     ```yaml
     routing:
       architect: anthropic/claude-sonnet-4
       task_executor: vllm/qwen3-coder:7b
       default: ollama/qwen3:14b
     ```

6. **Fallback/Retry Logic**
   - Primary provider fails → fallback to secondary
   - Rate limit hit → queue request, retry with backoff
   - Cost limit exceeded → switch to cheaper model

#### Metrics
- **Effort**: 2/5 (Mostly abstraction refactoring, APIs are standard)
- **Risk**: 2/5 (Low risk, existing code still works with Ollama)
- **Value**: 4/5 (Enables cloud deployment, cost optimization)
- **Priority Score**: 4 / (2 × 2) = **1.0** (🔥 HIGHEST PRIORITY)

#### Dependencies
- None (can be done independently)
- Unlocks Theme 2 (cloud deployments)

---

### 4. Multimodal LLM Support (Screenshots & Reflection)

**Vision**: Agent takes screenshots of created software, uses vision-capable LLM to verify it looks correct.

#### Current State
- Text-only LLM interactions
- No visual feedback loop
- Server management exists (start/stop servers)
- No screenshot capability

#### Implementation Steps

**Phase 1: Screenshot Infrastructure (3-4 days)**
1. **Screenshot Tool**
   - Add `screenshot_tools.py` behavior:
     - `take_screenshot(url, path)` - uses Playwright/Selenium
     - `compare_screenshots(path1, path2)` - pixel diff or vision model
   - Whitelist for task_executor config

2. **Visual Assertion DSL**
   - Agent can request: "Take screenshot of http://localhost:3000 and verify login button is visible"
   - Store screenshots in workspace: `.screenshots/round_{N}.png`

**Phase 2: Vision Model Integration (1 week)**
3. **Vision Provider Interface**
   - Extend LLM providers to support image inputs:
     ```python
     def chat(messages, tools=None, images=None) -> ChatResponse
     ```
   - Implement for:
     - GPT-4o Vision (OpenAI)
     - Claude 3 Opus/Sonnet (Anthropic)
     - Gemini Pro Vision (Google)
     - LLaVA (local, via Ollama)

4. **Visual Verification Workflow**
   - TaskExecutor creates UI → starts server → takes screenshot
   - Sends to vision model: "Does this look like a working login page?"
   - Model responds: "Yes, but button text is cut off"
   - Agent fixes CSS → screenshot again → verify

**Phase 3: Advanced Use Cases (1 week)**
5. **Visual Regression Testing**
   - Before refactor: take baseline screenshot
   - After refactor: take new screenshot
   - Auto-compare: "UI unchanged? ✅ / ❌ Differences detected"

6. **Design Mockup → Implementation**
   - User provides mockup image: `design.png`
   - Agent generates HTML/CSS matching design
   - Uses vision model to compare screenshot vs mockup
   - Iterates until similarity > 90%

#### Metrics
- **Effort**: 3/5 (New tooling, but Playwright is well-documented)
- **Risk**: 3/5 (Browser automation can be flaky)
- **Value**: 4/5 (Significantly improves UI development quality)
- **Priority Score**: 4 / (3 × 3) = **0.44** (Medium-high priority)

#### Dependencies
- Requires Theme 3 (cloud LLMs for vision models)
- ServerManagementBehavior already exists

---

### 5. Browser Navigation

**Vision**: Agent can browse the web, interact with pages, fill forms, scrape data.

#### Current State
- No browser automation
- No web scraping capabilities
- ServerManagementBehavior only manages local dev servers

#### Implementation Steps

**Phase 1: Basic Navigation (1 week)**
1. **Browser Behavior**
   - Create `BrowserNavigationBehavior`
   - Tools:
     - `navigate_to(url)` - go to URL
     - `click(selector)` - click element
     - `fill_form(selector, value)` - fill input
     - `extract_text(selector)` - scrape content
   - Uses Playwright (headless or headed)

2. **Security Constraints**
   - Whitelist allowed domains (prevent open-ended browsing)
   - Disable javascript by default (security)
   - Sandbox: run in isolated Docker container

**Phase 2: Intelligent Interaction (1-2 weeks)**
3. **Vision-Guided Navigation** (requires Theme 4)
   - Agent doesn't need CSS selectors
   - Takes screenshot → uses vision model: "Where is the login button?"
   - Vision model returns coordinates → agent clicks

4. **Form Auto-Fill**
   - Task: "Sign up for newsletter on example.com with test email"
   - Agent navigates → detects form → fills → submits
   - Validates success (checks for confirmation message)

**Phase 3: Advanced Use Cases (2 weeks)**
5. **Web Research Agent**
   - Task: "Find Python async best practices from Stack Overflow"
   - Agent searches Google → opens top 3 results → extracts content → summarizes

6. **Integration Testing**
   - Deploy web app to staging
   - Agent navigates through user flows
   - Validates: signup → login → create post → logout
   - Reports bugs if flows fail

#### Metrics
- **Effort**: 3/5 (Playwright integration straightforward)
- **Risk**: 4/5 (Security risk if agent browses untrusted sites)
- **Value**: 3/5 (Useful, but niche compared to code generation)
- **Priority Score**: 3 / (3 × 4) = **0.25** (Medium priority)

#### Dependencies
- Benefits from Theme 4 (vision for intelligent interaction)
- Requires security hardening (sandboxing)

---

### 6. Enhanced TUI (Dashboard, Live Interaction)

**Vision**: Rich terminal UI for monitoring agent execution, viewing logs, pausing/resuming, inspecting context.

#### Current State
- Basic text output to console
- StatusDisplayBehavior deprecated (removed)
- No interactive controls
- Context inspection saves JSON files (not visualized)

#### Implementation Steps

**Phase 1: Rich Console UI (1 week)**
1. **Framework Selection**
   - Use Textual (Python TUI framework)
   - Or Rich (simpler, less interactive)
   - Or both: Rich for static logs, Textual for dashboard

2. **Core Dashboard Layout**
   - Header: Current goal, agent name, model, elapsed time
   - Left panel: Goal tree (expandable subtasks)
   - Center panel: Live log output (scrollable)
   - Right panel: Workspace file tree
   - Footer: Token usage, cost, controls (Pause/Resume/Cancel)

**Phase 2: Interactive Features (1 week)**
3. **Keyboard Controls**
   - `p` - Pause agent (finishes current round, then waits)
   - `r` - Resume agent
   - `q` - Quit (graceful shutdown)
   - `c` - Open context inspector (view last prompt)
   - `f` - Open file in editor (read-only preview)

4. **Context Inspector View**
   - Display full prompt sent to LLM
   - Syntax highlighting (markdown, JSON)
   - Token count per section
   - Pagination for long contexts

**Phase 3: Advanced Visualization (1-2 weeks)**
5. **Performance Charts**
   - Token usage over time (line chart)
   - Rounds per subtask (bar chart)
   - Time spent per behavior (pie chart)
   - Uses Plotext (TUI plotting library)

6. **Multi-Agent Dashboard** (for Theme 1)
   - Split screen: multiple agents running
   - Switch between agents with Tab
   - Shared resource monitor (total VRAM, CPU)

#### Metrics
- **Effort**: 2/5 (Textual has good docs, mostly UI work)
- **Risk**: 1/5 (Low risk, doesn't affect core logic)
- **Value**: 3/5 (Nice UX improvement, not critical)
- **Priority Score**: 3 / (2 × 1) = **1.5** (🔥 SECOND HIGHEST PRIORITY)

#### Dependencies
- None (can be done independently)
- Complements Theme 1 (long-running agents need oversight)

---

## Prioritized Roadmap

| Rank | Theme | Priority Score | Effort | Risk | Value | Rationale |
|------|-------|----------------|--------|------|-------|-----------|
| **1** | **LLM Provider Flexibility** | **1.5** | 2 | 2 | 4 | Low effort, low risk, high value. Unlocks cloud deployment and cost optimization. |
| **2** | **Enhanced TUI** | **1.0** | 2 | 1 | 3 | Low effort, minimal risk, improves UX significantly. Foundational for other themes. |
| **3** | **Multimodal LLM** | **0.44** | 3 | 3 | 4 | Moderate effort, enables high-quality UI development. Requires vision models (Theme 3). |
| **4** | **Dynamic Agent Deployment** | **0.33** | 3 | 4 | 4 | Unlocks ecosystem growth. Security is critical but already addressed. |
| **5** | **Longer-Term Execution** | **0.25** | 5 | 4 | 5 | Highest value but most complex. Needs Theme 6 for oversight. Defer until foundations solid. |
| **6** | **Browser Navigation** | **0.25** | 3 | 4 | 3 | Useful but niche. Security risks. Nice-to-have, not critical. |

---

## Recommended Implementation Order

### Phase 1: Foundations (3-4 weeks)
1. **Theme 3: LLM Provider Flexibility** (1 week)
   - Abstracts LLM interface
   - Enables cloud models
   - Quick win, high impact

2. **Theme 6: Enhanced TUI** (1 week)
   - Improves development experience
   - Foundation for monitoring long-running agents
   - Low risk, immediate value

3. **Theme 2: Dynamic Agent Deployment** (2 weeks)
   - Build packaging and deployment tooling
   - Validate security automation
   - Unlocks ecosystem

### Phase 2: Advanced Features (4-6 weeks)
4. **Theme 4: Multimodal LLM** (2 weeks)
   - Integrate vision models
   - Implement screenshot verification
   - Improves UI development

5. **Theme 1: Longer-Term Execution** (3-4 weeks)
   - Goal queue system
   - Crash recovery
   - Observability & reversibility
   - **Defer until Phases 1 complete** (needs TUI for oversight)

### Phase 3: Nice-to-Haves (2-3 weeks)
6. **Theme 5: Browser Navigation** (2-3 weeks)
   - Implement if there's clear demand
   - Security sandboxing required
   - Consider community contribution

---

## Risk Mitigation Strategies

### For Theme 1 (Longer-Term Execution)
- **Risk**: Runaway resource usage, infinite loops
- **Mitigation**:
  - Hard resource limits (max VRAM, max runtime per goal)
  - Watchdog process: kills agent if idle for >30min
  - User approval gates for destructive operations

### For Theme 2 (Agent Deployment)
- **Risk**: Security vulnerabilities in deployed agents
- **Mitigation**:
  - Automated Rule of Two validation
  - Container hardening (AppArmor profiles)
  - Network policies (whitelist domains)
  - Regular security audits

### For Theme 3 (LLM Providers)
- **Risk**: API key leakage, cost overruns
- **Mitigation**:
  - Never log API keys
  - Cost limits in config (max $X per day)
  - Alert on spike in API usage

### For Theme 4 (Multimodal)
- **Risk**: Vision model hallucinations (false positives)
- **Mitigation**:
  - Human verification for critical UI changes
  - Fallback to pixel diff if vision model uncertain
  - Log all visual assertions for audit

---

## Success Metrics

### Short-Term (3 months)
- ✅ 3+ LLM providers supported
- ✅ TUI dashboard live and in use
- ✅ 5+ agents deployed to cloud (email scanner, etc.)
- ✅ 80%+ success rate on L5-L7 evaluation tasks

### Medium-Term (6 months)
- ✅ 10+ community-contributed agents in registry
- ✅ Multimodal verification on 50+ UI projects
- ✅ 24/7 agent execution stable for 1 week+ runs
- ✅ Cost per task <$0.10 (optimized routing)

### Long-Term (12 months)
- ✅ 100+ agents in public registry
- ✅ Jetbox used in production by 10+ companies
- ✅ Browser navigation enables web scraping use cases
- ✅ Agent autonomy: 90%+ tasks complete without human intervention

---

## Conclusion

The recommended strategy is:
1. **Start with Theme 3 (LLM flexibility)** - low effort, high value, unlocks cloud
2. **Follow with Theme 6 (TUI)** - improves DX, foundation for monitoring
3. **Then Theme 2 (deployment)** - ecosystem growth, leverage existing security
4. **Add Theme 4 (multimodal)** after LLM abstraction in place
5. **Tackle Theme 1 (long-term execution)** only after TUI and observability solid
6. **Consider Theme 5 (browser)** based on community demand

This order balances quick wins, risk mitigation, and foundational work before tackling the most complex challenges.

**Estimated timeline to complete all 6 themes: 9-13 weeks**

---

*Document created by Claude Code analysis on 2025-11-18*
