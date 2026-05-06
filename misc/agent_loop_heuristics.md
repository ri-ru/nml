# Agent Self-Improvement Loop — Observed Failure Modes & Heuristics

Derived from running a real ML training loop (NBA trajectory prediction, nml-2026 Kaggle)
where Claude autonomously designed experiments, launched training runs, monitored logs,
recorded results, and queued the next experiment.

---

## Failure Mode 1: Architecture Drift

**What happened:** Given freedom to "design the next experiment," the agent gradually drifted
from the established architecture — wrong loss function, wrong feature dimensions, wrong
decoding strategy. Each step was locally plausible but globally wrong.

**Root cause:** No explicit anchor to the current best model. The agent reasoned from
general ML knowledge rather than from the specific codebase state.

**Heuristic:** Always pass the agent the *diff* it's allowed to make, not just the goal.
"Modify X, keep everything else identical to file Y" is far more reliable than "design a
better version." Constrain the search space explicitly in the prompt.

---

## Failure Mode 2: Output Path Confusion

**What happened:** Logs were saved to `models/` instead of `models/logs/`. Weight files
were overwritten across runs (35ep and 100ep shared the same `.pth`). Results were
appended to a deleted file.

**Root cause:** The agent infers paths from context rather than verifying them. When the
context window compresses, it loses track of where things actually are.

**Heuristic:** Every write operation needs an explicit absolute path in the instruction,
AND a verification step (does the file exist at that path?). Never let the agent infer
paths from filenames or conventions — state them.

---

## Failure Mode 3: Process Blindness

**What happened:** The agent launched duplicate inference scripts, didn't know which PIDs
corresponded to which training runs, and had no way to verify a process was still alive
vs. finished silently.

**Root cause:** There's no intrinsic awareness of system state. The agent can run
`Get-Process` but has no persistent map of "PID X = job Y started at time T."

**Heuristic:** Maintain a process registry as a file (e.g., `running.json`: pid, command,
start time, log file). The agent writes to it on launch and reads it on every check.
Never let process identity live only in the LLM's working memory.

---

## Failure Mode 4: Context Compaction Amnesia

**What happened:** After context compression, the agent forgot which experiments were
running, what scores had been achieved, and what the current best model was. It had to
re-read logs from scratch and sometimes got confused about what was already recorded.

**Root cause:** LLM context is ephemeral. When compacted, summaries lose precision —
exact scores become approximate, running states become ambiguous.

**Heuristic:** All agent state must live in files, not in the conversation. A `state.json`
with `{running: [...], completed: [...], best_score: X, queue: [...]}` that the agent
re-reads at the start of every loop iteration is mandatory. The LLM is stateless; treat
it that way.

---

## Failure Mode 5: No Meta-Orchestration

**What happened:** The agent followed the queue mechanically. It couldn't synthesize
"given that v8 failed because of X and v12 succeeded because of Y, the optimal next
experiment is Z." It needed the human to design the experiments.

**Root cause:** Experiment design requires holding the full result history in working
memory and reasoning about causality across runs. This is exactly what compaction
destroys.

**Heuristic:** Separate the "design" step from the "execute" step. A dedicated meta-agent
with access to the full results table designs the next experiment as a structured spec
(architecture delta, expected hypothesis, success criterion). The execution agent just
runs it. The meta-agent fires less frequently and with richer context.

---

## Failure Mode 6: Instruction Degradation

**What happened:** Loop instructions that were clear at the start became ambiguous after
several iterations. Edge cases (what to do if a run scores exactly the threshold, what
to do if logs are missing) weren't handled, causing the agent to improvise inconsistently.

**Root cause:** Natural language instructions are underspecified. The agent fills gaps
with plausible behavior that may not match intent.

**Heuristic:** Encode the loop as a structured decision tree, not prose. Every branch
should be explicit: IF score < threshold THEN ... ELSE IF score == running THEN ...
Machine-readable loop specs (YAML or JSON) are more robust than Markdown instructions.

---

## Failure Mode 7: Score Parsing Fragility

**What happened:** Results depended on parsing `AGENT_SCORE:X.XXXX` from log files.
If a run crashed before writing that line, the result was silently lost. If the format
changed slightly, parsing failed.

**Root cause:** Structured output was embedded in free-form log text.

**Heuristic:** Write results to a structured sidecar file (e.g., `result.json` next to
each run's log) immediately when available. Never parse scores from logs — parse them
from structured output only.

---

## Meta-Observation: What "Optimizing the Config" Actually Means

The agent's behavior is fully determined by its instruction set (the "config" passed in
the loop prompt). Every failure above was a config failure, not a model failure. The model
did exactly what the instructions implied — the instructions just didn't anticipate the
edge case.

A system that can optimize agent configs needs to:
1. **Observe** failure modes from execution traces (not just from the agent's self-report)
2. **Classify** which instruction was underspecified or missing
3. **Propose** a targeted patch to the instruction (not a rewrite)
4. **Validate** the patch doesn't break other branches

The loop itself can be the training signal — every time a human corrects the agent,
that correction is a labeled example of an instruction gap.
