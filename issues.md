# issues

Findings from a full read of `src/` (2,275 lines), a build on both toolchains, and
probes against the stream parser. Ranked by impact.

## Literally broken

### 1. Does not compile on stable Rust — `src/tool.rs:117`

```
error[E0658]: use of unstable library feature `string_from_utf8_lossy_owned`
```

`String::from_utf8_lossy_owned` is nightly-only. Replace with
`String::from_utf8_lossy(&output.stdout).into_owned()`.

### 2. Unvalidated `i32 -> usize` cast sizes a Vec — `src/agent.rs:169-172`

```rust
let index = call.index as usize;
while self.tool_slots.len() <= index { self.tool_slots.push(ToolSlot::default()); }
```

`DeltaToolCall.index` is `i32`. A negative index becomes ~1.8e19 and the loop pushes
until the process dies. A bogus large positive index allocates blindly. Reject
`index < 0` and cap against a sane maximum.

### 3. The stream parser rejects valid chunks

Confirmed by probe:

```
PROBE: missing delta FAILS: missing field `delta`
PROBE: missing index FAILS: missing field `index`
PROBE: reasoning_content yields []
```

- `StreamDelta` is required (`src/agent.rs:104`). A server sending
  `{"choices":[{"index":0}]}` kills the whole turn with a deserialization error.
- `index` on tool-call deltas is required, so providers that omit it (some
  llama.cpp / Mistral shapes) break every tool call.
- Only `reasoning` is read (`src/agent.rs:112`). vLLM, DeepSeek, llama.cpp and
  Ollama emit `reasoning_content`, so thinking is silently dropped on all of them.

Fix: `#[serde(default)]` on `delta`, `#[serde(alias = "reasoning_content")]` on
`reasoning`, and tolerate a missing `index`.

### 4. Tool calls can be emitted with an empty id — `src/agent.rs:211`

The filter keeps a slot if *any* field is non-empty, so a provider that omits `id`
produces `"tool_call_id": ""` in the follow-up message. Synthesize an id when the
slot's id is empty.

## Correctness

### 5. bash discards stdout on failure — `src/tool.rs:109-115`

`NonZeroExit` carries only stderr. `cargo test`, `pytest`, `tsc` and `grep`
(exit 1 = no match) put the useful part on stdout, so the model receives
`"bash exited with 1"` and nothing else. Likely the single biggest quality drag on
the agent. Include stdout in the error.

### 6. bash has no timeout, no output cap, and inherits stdin — `src/tool.rs:101-107`

`npm run dev` hangs the agent permanently; `cat` of a large file blows the context
window in one call; an interactive child steals the REPL's stdin. Add
`.stdin(Stdio::null())`, a `tokio::time::timeout`, and output truncation.

### 7. Pruning also deletes the assistant's prose — `src/context.rs:54-59`

The guard is `Assistant { tool_calls } if !tool_calls.is_empty()`, so a message
carrying *both* narration and a tool call is dropped whole. The existing test
asserts this: `"let me check"` (`src/context.rs:171`) never appears in the expected
output at `src/context.rs:204`. Keep `content`, drop only `tool_calls`.

### 8. Combined with empty write results, past work becomes invisible

`write_file` and `edit_file` return `Ok(String::new())` (`src/tool.rs:130`,
`src/tool.rs:146`). Once a turn completes, the tool exchange is pruned and the
narration is pruned with it, so nothing in history records that a file was ever
written. The model re-reads and redoes work across turns. Return something like
`"wrote 412 bytes to src/foo.rs"`.

### 9. Thinking is suppressed for the rest of a turn after the first text token

`AnswerGate` lives in the `Renderer`, created once per user input
(`src/main.rs:51`), but `chat()` runs many completions in the tool loop. Once
`answer_started` flips (`src/agent.rs:49-53`), every later completion's reasoning is
`Hidden`. Gate state should reset per completion, not per user turn.

### 10. Error path leaves the terminal dim and eats the answer — `src/main.rs:52-55`

`renderer.finish()` runs only on `Ok`. On error, `\x1b[2m` (`src/render.rs:98-105`)
is never reset, `<thinking>` / `<output>` are never closed, and text buffered in the
gate (`src/agent.rs:71`) is discarded. Call `finish()` in both arms.

### 11. No loop bound and a no-op compactor

`chat()` (`src/agent.rs:271`) has no max-iteration guard, so a model that keeps
calling tools spins forever. `compact()` is `{}` (`src/context.rs:81`), so
`needs_compaction()` latches true once tripped and nothing prevents overflow.

### 12. `tty` is sampled from stdout but applied to stderr

`src/main.rs:35` feeds `src/render.rs:79-95`. Piping stderr to a log while stdout is
a terminal writes escape codes into the log file. Check each stream independently.

### 13. read_file normalizes line endings, edit_file matches literally

`lines().join("\n")` (`src/tool.rs:121-126`) strips `\r` and the trailing newline. On
a CRLF file the model builds `old_content` from LF text, and `edit_file`'s exact
match then finds 0 occurrences — a confusing failure on a correct-looking edit.

### 14. Blocking I/O in async fns — `src/tool.rs:120, 129, 133, 145`

`std::fs` inside `async fn invoke`. Use `tokio::fs`.

## Performance

### 15. The whole transcript is deep-cloned twice per request — `src/context.rs:26-33`

`pruned_messages()` clones every `Message`, then `to_request()` clones every string
again. This runs on *each* tool-loop iteration, not once per turn. Return
`Vec<&Message>` and borrow in `to_request`.

### 16. Tool calls run sequentially — `src/agent.rs:308`

Parallel tool calls are supported (there is a test named for it), but two
independent `read_file`s serialize. `join_all` for read-only tools is a straight
win; keep writes ordered.

### 17. Prefix-cache thrash

Pruning rewrites history at the last user message every turn, invalidating the
server's KV prefix cache from that point onward and paying full prefill for the
discarded tool work. Fixing #7 also makes the prefix more stable.

### 18. Two flush syscalls per streamed token — `src/render.rs:60-61`

Both stdout and stderr are flushed on every chunk. Flush only the stream written to.

### 19. Smaller wins

- `tool_definitions()` rebuilds four JSON schemas per request (`src/agent.rs:350`) —
  wrap in a `OnceLock`.
- `matches().count()` then `find()` scans the file twice (`src/tool.rs:134,142`).
- `serde_json::to_value(request)` builds a whole tree to patch three keys
  (`src/agent.rs:359`).
- `Tool::output()` clones the entire `write_file` content purely to display it
  (`src/tool.rs:83`).

## Nits

- `ToolSlot`'s manual `Default` (`src/agent.rs:123`) should be derived.
- `Agent::history` is dead code outside tests (build warning).
- `KITE_THINKING=false` *enables* thinking (`src/main.rs:29` — only `"0"` disables).
- `kite.txt` and `poem.txt` are untracked test debris.
