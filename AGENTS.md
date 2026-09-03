# AGENTS.md

## Code style

- Never write any comments in code: no `//`, no `///`, no `//!`, no `/* */`,
  no `#[doc]`. Code should be self-explanatory.

## Testing

- Don't test obvious things: trivial getters, label/description tables, or
  anything that just mirrors the definition. Focus on behavior, edge cases,
  and negative/error paths.
