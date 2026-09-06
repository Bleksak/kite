# AGENTS.md

## Code style

- Never write any comments in code: no `//`, no `///`, no `//!`, no `/* */`,
  no `#[doc]`. Code should be self-explanatory.

## Testing

- Don't test obvious things: trivial getters, label/description tables, or
  anything that just mirrors the definition. Focus on behavior, edge cases,
  and negative/error paths.

## Keybindings (TUI)

- C-s: open/close the session picker
- C-q: open/close the background tasks overlay
- C-t: cycle the thinking level (off → low → medium → high → xhigh → off; shown in
  the status bar)
- Tab: switch the mode (yolo ↔ plan; shown in the prompt box header, bold green
  in plan). Implement mode is not reachable via Tab — it only starts when a
  plan is approved
- Plan gate: when the plan agent calls submit_plan, the prompt box header shows
  "plan ready" — Enter approves and starts the implement stage (a fresh agent
  in implement mode executes the plan); typing feedback rejects and re-plans.
  If the implement stage escalates, it re-plans automatically and the gate
  opens again
- C-c: quit
- Picker: C-j/↓ down, C-k/↑ up, enter select, esc cancel, C-n new session,
  C-x close the session under the cursor, C-r rename it, typing searches
  (backspace clears)
- Tasks: j/↓ down, k/↑ up, x kill the task under the cursor, enter show its
  output (jk/PageUp/PageDown scroll, Home top, End bottom, q/esc back to the
  list, C-q close all), q/esc close
- Chat: PageUp/PageDown and mouse wheel scroll, Home top, End bottom
