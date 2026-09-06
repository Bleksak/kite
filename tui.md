# Plan: ratatui TUI (current architecture)

A TUI face for the current single-agent REPL. No `Mode`, no `HarnessEvent`,
no gates — the TUI consumes `AgentEvent` exactly as the `Renderer` does today.
The existing line REPL stays as the pipe face; the TUI is additive.

## UI selection

- stdin is a TTY → TUI (alternate screen)
- stdin piped → the current plain tagged output, unchanged (the
  machine-readable contract stays intact)

`--ui plain|tui` overrides for testing.

## Layout

```
┌──────────────────────────────────────────────────────────┐
│ qwen3.8-27b  ·  4637 prompt / 408 completion tok         │  status bar
├──────────────────────────────────────────────────────────┤
│ (dim) let me check the files                             │
│ ⚙ read_file: src/main.rs                                │
│     fn main() { ... }                                   │
│ (dim) done                                               │
│ implementation finished, 3 files changed                 │
├──────────────────────────────────────────────────────────┤
│ > █                                                       │  input line
└──────────────────────────────────────────────────────────┘
```

- **Status bar:** model, prompt/completion token totals (from
  `Context::record_usage`, shown after each completion).
- **Main pane:** the agent stream — thinking (dim), answer (normal), tool
  blocks (header bold + body dim/indented). Tags and escape codes are
  replaced by ratatui `Style`; the `AnswerGate` live/hidden/buffered logic is
  reused as-is.
- **Input line:** the user's task; Enter starts `chat()`. While a completion
  runs the line shows a busy state; input is accepted between turns, as in the
  current REPL.

No artifact pane — the current architecture has no artifacts. The pane appears
with the mode-switching phase.

## Key bindings

| Key | Action |
| --- | --- |
| typing + `Enter` | submit the task (between turns) |
| `PageUp` / `PageDown` / wheel | scroll the main pane |
| `q` / `Ctrl+C` | quit (restore the terminal) |

No gate keys — there are no gates in the current architecture.

## Concurrency

```
tokio runtime
├── agent task:  agent.chat(task, &mut |e| tx.send(e))  → mpsc<AgentEvent>
├── key thread:  crossterm::event::poll (blocking) → mpsc<KeyCommand>
└── TUI loop:    select! { agent event, key, resize } → update state → draw
```

The current `chat()` takes a synchronous `FnMut(AgentEvent)` callback — the TUI
adapter is that callback, sending each event into an unbounded channel (non
blocking). **No changes to `Agent` are needed.** The key poll runs on a
dedicated OS thread (crossterm polling is blocking; no `spawn_blocking`).
Event-driven redraw — no timer; if per-chunk redraws prove heavy, coalesce on
a 16 ms tick (ratatui diffs the buffer, so unchanged cells are cheap).

## Terminal lifecycle

- `EnterAlternateScreen` on start, `LeaveAlternateScreen` on exit.
- **Panic hook** that restores the terminal before printing the panic — a panic
  must never leave the user in a broken alternate-screen state.
- `Resize` events re-query `terminal::size()`; the layout is flex-based
  (ratatui `Layout`), so resizes are free.

## TUI renderer

A new consumer alongside the existing `Renderer` (which stays for the pipe
face). Same events, different sink:

```rust
struct TuiRenderer {
    gate: AnswerGate,             // reused as-is — it's a pure state machine
    scrollback: Vec<Line<'static>>,
    status: Status,
}
```

- `CompletionStarted` resets the gate and closes open scopes, as in the
  current `Renderer`.
- `Tokens` → `AnswerGate` decides live thinking (dim lines) vs buffered/answer
  (normal lines); the gate-reset and orphaned-buffer-flush behavior is ported
  1:1.
- `ToolStarted` / `ToolResult` → header bold, body dim/indented; the
  parallel-result header matching from the current `Renderer` is ported 1:1.

## State

```rust
struct TuiState {
    renderer: TuiRenderer,
    scroll: usize,
    input: String,
    running: bool,
}
```

Key handling is a **pure function**
`(TuiState, Key) → (TuiState, Option<Action>)` — unit-tested without a
terminal. `Action` is `Submit(String)` or `Quit`.

## Testing

- **TUI renderer:** golden tests — scripted `AgentEvent` sequences in,
  `scrollback`/`status` lines out. Same discipline as the current `Renderer`
  golden tests (thinking live vs hidden, buffered flush at the completion
  boundary, tool block alignment, per-stream styling).
- **Key handling:** the pure function — submit, scroll, quit.
- **Frames:** `ratatui::backend::TestBackend` — feed events, draw, assert the
  rendered cell grid (status bar, main pane, input line).
- **Live:** the current REPL's end-to-end scenario, watched in the TUI.

## Build order

1. `ratatui` + `crossterm` deps; `--ui` flag; TTY→tui / pipe→plain selection.
2. Terminal lifecycle: alternate screen, panic hook, resize.
3. Key thread + the `select!` event loop (agent events, keys, resize).
4. `TuiRenderer` (`AgentEvent` → styled scrollback; `AnswerGate` reused).
5. Layout (status / main / input) + `TestBackend` frame goldens.
6. Scroll + input line + busy state.
7. Live verification.

## Out of scope

- Gates, `Mode`, `HarnessEvent`, the artifact pane — the mode-switching phase.
- Mouse selection/copy — the alternate screen breaks terminal copy;
  `Ctrl+Shift+C` passthrough is a possible later addition.
- Simultaneous multi-agent panes.
- Themes, persistent scrollback across tasks.

## Risks

- **Redraw cost under fast streaming** — mitigated by buffer diffing + the
  16 ms coalescing tick if needed.
- **Panic in the event loop** — the panic hook is load-bearing; test it
  (deliberate panic in a test harness, assert the terminal is restored).
- **No mid-turn interruption** — `chat()` is awaited to completion; killing a
  run requires stream cancellation, which is a follow-up, not part of this
  plan.
