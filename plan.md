# Plan: Mode switching (Plan → Implement → Review)

## Core types

```rust
enum Mode { Plan, Implement, Review }

enum ChatOutcome {
    Answer(String),
    Terminated { tool: String, arguments: String },
}

enum Gate { PlanApproval, ReviewVerdict }

struct Task {
    request: String,              // the user's original task
    mode: Mode,
    awaiting: Option<Gate>,
    plan: Option<String>,
    review: Option<String>,
    escalations: u32,
    touched: Vec<String>,        // write/edit paths from the current Implement round
}
```

Each stage runs a fresh `Agent` with a fresh `Context` — the handoff payload
(plan text, findings, partial state) is the bridge between stages. No transcript
carries across stages; contexts stay small. The existing pruning/re-read machinery
works unchanged inside each stage.

## Per-mode configuration

|          | Plan                          | Implement                  | Review               |
| -------- | ----------------------------- | -------------------------- | -------------------- |
| Tools    | `read_file`, `bash`, `submit_plan` | all four + `escalate` | `read_file`, `bash` |
| Terminator | `submit_plan`               | `escalate`                 | — (answer = findings) |
| Exit     | `Terminated(submit_plan)` → gate | `Answer` → Review; `Terminated(escalate)` → Plan | `Answer` → verdict gate |

`tool_definitions()` becomes `tool_definitions(mode)` — the `LazyLock` cache
becomes a 3-entry array. `submit_plan(plan: String)` and
`escalate(findings: String)` get `Args` structs + schema entries like the
others, but are **never executed** — the harness intercepts them.

## Terminator interception

In the `chat()` loop, before `run_tools`: if any call in the round is the mode's
terminator, return `ChatOutcome::Terminated` with that call's raw arguments —
**other calls in the same round are dropped**. Prompt line: *"Call `escalate`
alone, without other tools."* A terminator is only in the schema for its own
mode, so a stray call can't happen silently.

**Lenient fallback for Plan:** if the model just *answers* without calling
`submit_plan`, the answer is treated as the plan payload (local models drop
tools). Implement's `Answer` is stage-complete by definition; Review's `Answer`
is the findings.

## State machine

```
Plan ──submit_plan──▶ PlanApproval gate ──approve──▶ Implement ──answer──▶ Review ──answer──▶ Verdict gate
  ▲                        │                                            ▲                      │
  │                        └──reject(feedback)──▶ Plan                  │        done │ fix ───┘
  │                                                                   └──── replan ──▶ Plan
  └──────────────────escalate(findings)──────────────────────────────────┘
```

Routing is a **pure function** — `(Task, StageEvent) → (Task, Option<NextAction>)`
— unit-tested without any network. The `Agent`/client stays concrete; only the
routing table is tested in isolation, end-to-end verified live.

## Re-entry context assembly (pure functions, unit-tested)

- **Re-plan (after escalate):** re-plan system prompt + user message: *"The plan
  hit a blocker: {findings}. Already implemented: {touched}. Original plan:
  {plan}. Revise the plan, keeping what's done, and call submit_plan."*
  `escalations += 1`, shown at the gate.
- **Re-plan (after reject):** same, with your feedback in place of findings.
- **Implement (after approval):** implement prompt + *"Execute this plan: {plan}"*.
- **Implement (after review fix):** implement prompt + *"Fix these review
  findings: {review}"*.
- **Review:** review prompt + the task + plan; its answer is the findings.

**Partial state:** `touched` is collected from `ToolStarted` headers
(`write_file: <path>`, `edit_file: <path>`) as events stream by — the header
format is ours and stable. No rollback; the revised plan builds on what's on
disk.

## Event stream (the refactor that enables everything)

`chat()`'s callback becomes an `mpsc::UnboundedSender<AgentEvent>`; the harness
wraps it:

```rust
enum HarnessEvent {
    StageStarted(Mode),
    Agent(AgentEvent),
    PlanSubmitted(String),
    GateRequired(Gate),
    StageCompleted(Mode),
}
```

The Renderer consumes `HarnessEvent` — `Agent` variants pass through to the
existing `Renderer` untouched; stage transitions render a status line
(`[stage] plan → implement`). `CompletionStarted`/gate resets work as they do
today.

## REPL gates

- **PlanApproval:** print the plan, prompt `approve / reject <feedback>`.
- **Verdict:** print the findings, prompt `done / fix / replan`.
- Escalation count in the status line — a ping-pong shows up as the same gate
  recurring; you break it by hand. That's the loop guard (plus the count).

## Build order

1. **Event stream** — `chat()` sends to a channel; REPL consumes; Renderer
   unchanged. *Test: events arrive in order through the channel.*
2. **`Mode` + per-mode config** — `tool_definitions(mode)`, the two terminator
   `Args` structs/schemas. *Tests: schema subsets per mode.*
3. **Terminator interception + `ChatOutcome`** — *Tests: terminator ends the loop
   and captures the payload; sibling calls dropped; lenient Plan fallback.*
4. **Routing table (pure) + `Task`** — *Tests: the full graph, including
   escalate and reject loops.*
5. **Re-entry context assembly** — *Tests: each of the five message shapes.*
6. **REPL gates + status lines** — live verification of the full loop.
7. **Live end-to-end** against vLLM: task → plan → approve → implement → review
   → verdict, plus a forced escalation.

**Deliberately out of scope** (later phases): parallel research fan-out in Plan,
parallel implementation agents, the ratatui face, transcript persistence. The
event stream + `HarnessEvent` are the seams those plug into.

## Open decision

Reject at the PlanApproval gate — free-text feedback only, or also a "restart
plan from scratch" option (drops the old plan from context)? Leaning
feedback-only.
