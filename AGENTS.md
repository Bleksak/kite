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
- Plan gate: the plan agent calls submit_plan with a plan split into stages.
  The prompt box header shows "plan ready (N stages)" — Enter implements Step 1,
  typing feedback re-plans. After each stage is implemented, a review gate opens
  ("Step N implemented — review the implementation"; still in implement mode):
  Enter moves to the plan review ("review Step N+1"; plan mode), typing feedback
  re-plans from the current stage. Enter at the plan review implements the next
  stage; typing feedback re-plans the remaining stages. Escalating re-plans the
  remaining stages automatically and the gate opens again
- C-p: open/close the plan popup — shows the viewed stage's title and tasks
  (like the background job details); ←/→ switch the viewed step, jk/PgUp/PgDn/
  wheel scroll, p/q/esc close it
- C-c: quit
- Picker: C-j/↓ down, C-k/↑ up, enter select, esc cancel, C-n new session,
  C-x close the session under the cursor, C-r rename it, typing searches
  (backspace clears)
- Tasks: j/↓ down, k/↑ up, x kill the task under the cursor, enter show its
  output (jk/PageUp/PageDown scroll, Home top, End bottom, q/esc back to the
  list, C-q close all), q/esc close
- Chat: PageUp/PageDown and mouse wheel scroll, Home top, End bottom
