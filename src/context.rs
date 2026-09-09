use openai_oxide::types::chat::ChatCompletionMessageParam;
use serde::{Deserialize, Serialize};

use crate::message::Message;

/// Compact once the last reported prompt size passes this share of the window,
/// so the expensive request happens before the wall rather than at it.
const COMPACT_AT_PERCENT: u64 = 70;
/// How much of the window the post-compaction recent window may occupy. The gap
/// between this and `COMPACT_AT_PERCENT` is the headroom each compaction buys.
const KEEP_RECENT_PERCENT: u64 = 30;
/// Stands in for a tool call whose turn ended before a result was recorded.
const UNANSWERED_TOOL_CALL: &str = "[no result — the turn ended with this call]";

#[derive(Clone, Serialize, Deserialize)]
pub struct Context {
    pub system_prompt: String,
    pub messages: Vec<Message>,
    pub prompt_tokens: Option<u64>,
    pub total_prompt_tokens: u64,
    pub total_completion_tokens: u64,
    pub max_tokens: u64,
}

impl Context {
    pub fn new(system_prompt: impl Into<String>, max_tokens: u64) -> Context {
        Context {
            system_prompt: system_prompt.into(),
            messages: Vec::new(),
            prompt_tokens: None,
            total_prompt_tokens: 0,
            total_completion_tokens: 0,
            max_tokens,
        }
    }

    pub fn build_messages(&self) -> Vec<ChatCompletionMessageParam> {
        std::iter::once(Message::System {
            content: self.system_prompt.clone(),
        })
        .chain(self.messages.iter().cloned())
        .map(|message| message.to_request())
        .collect()
    }

    pub fn record_usage(&mut self, prompt: Option<u64>, completion: Option<u64>) {
        if let Some(prompt) = prompt {
            self.prompt_tokens = Some(prompt);
            self.total_prompt_tokens += prompt;
        }
        if let Some(completion) = completion {
            self.total_completion_tokens += completion;
        }
    }

    pub fn compaction_threshold(&self) -> u64 {
        self.max_tokens * COMPACT_AT_PERCENT / 100
    }

    pub fn keep_recent_tokens(&self) -> u64 {
        self.max_tokens * KEEP_RECENT_PERCENT / 100
    }

    pub fn needs_compaction(&self) -> bool {
        self.prompt_tokens
            .is_some_and(|tokens| tokens > self.compaction_threshold())
    }

    /// Append a stub result for any tool call left unanswered — the terminator
    /// path and a mid-round cancellation both end a turn without one, and an
    /// assistant `tool_calls` message with no matching result is rejected by the
    /// API now that history is sent verbatim. Stubs are inserted directly after
    /// the run of results that follows their call, so ordering stays valid
    /// whenever this runs.
    pub fn seal_dangling_tool_calls(&mut self) -> usize {
        let mut sealed = 0;
        let mut index = 0;
        while index < self.messages.len() {
            let ids: Vec<String> = match &self.messages[index] {
                Message::Assistant { tool_calls, .. } if !tool_calls.is_empty() => {
                    tool_calls.iter().map(|call| call.id.clone()).collect()
                }
                _ => {
                    index += 1;
                    continue;
                }
            };
            let mut end = index + 1;
            let mut answered: Vec<&str> = Vec::new();
            while let Some(Message::Tool { tool_call_id, .. }) = self.messages.get(end) {
                answered.push(tool_call_id);
                end += 1;
            }
            let missing: Vec<String> = ids
                .into_iter()
                .filter(|id| !answered.contains(&id.as_str()))
                .collect();
            for (offset, id) in missing.iter().enumerate() {
                self.messages.insert(
                    end + offset,
                    Message::Tool {
                        tool_call_id: id.clone(),
                        content: UNANSWERED_TOOL_CALL.to_string(),
                    },
                );
            }
            sealed += missing.len();
            index = end + missing.len();
        }
        sealed
    }

    pub fn is_in_progress(&self) -> bool {
        match self.messages.last() {
            Some(Message::Assistant { tool_calls, .. }) => !tool_calls.is_empty(),
            Some(Message::Tool { .. }) => true,
            _ => false,
        }
    }

    fn estimate_tokens(message: &Message) -> u64 {
        let chars = match message {
            Message::System { content } | Message::User { content } | Message::Tool { content, .. } => {
                content.len()
            }
            Message::Assistant { content, tool_calls } => {
                let mut chars = content.as_ref().map(|c| c.len()).unwrap_or(0);
                for call in tool_calls {
                    chars += call.function.name.len() + call.function.arguments.len();
                }
                chars
            }
        };
        (chars as u64) / 4
    }

    pub fn compact_point(&self, keep_recent_tokens: u64) -> Option<usize> {
        if self.messages.len() < 4 {
            return None;
        }
        let mut accumulated = 0u64;
        let mut cut_index = 0usize;
        let mut reached = false;
        for (index, message) in self.messages.iter().enumerate().rev() {
            accumulated += Self::estimate_tokens(message);
            if accumulated >= keep_recent_tokens {
                cut_index = index;
                reached = true;
                break;
            }
        }
        if !reached {
            return None;
        }
        for candidate in cut_index..self.messages.len() {
            if matches!(self.messages[candidate], Message::Tool { .. }) {
                continue;
            }
            // Cutting at 0 would summarize nothing and prepend the summary to a
            // context that is already too big — a compaction loop, one API call
            // per round. Better to leave it alone and let the round proceed.
            return (candidate > 0).then_some(candidate);
        }
        None
    }

    pub fn apply_compaction(&mut self, summary: String, cut_point: usize) {
        let kept = self.messages[cut_point..].to_vec();
        self.messages = vec![Message::User {
            content: format!("[summary of earlier context]\n{summary}"),
        }];
        self.messages.extend(kept);
        // `prompt_tokens` still holds the pre-compaction size reported by the
        // last completion. Leaving it would re-trigger compaction on the very
        // next round and summarize the summary. Estimate until real usage lands.
        self.prompt_tokens = Some(self.estimated_prompt_tokens());
    }

    fn estimated_prompt_tokens(&self) -> u64 {
        (self.system_prompt.len() as u64) / 4
            + self
                .messages
                .iter()
                .map(Self::estimate_tokens)
                .sum::<u64>()
    }

    pub fn file_operations(&self) -> (Vec<String>, Vec<String>) {
        let mut read = Vec::new();
        let mut modified = Vec::new();
        for message in &self.messages {
            if let Message::Assistant { tool_calls, .. } = message {
                for call in tool_calls {
                    let Some(path) = Self::tool_path(&call.function.name, &call.function.arguments)
                    else {
                        continue;
                    };
                    match call.function.name.as_str() {
                        "read_file" => {
                            if !read.contains(&path) {
                                read.push(path);
                            }
                        }
                        "write_file" | "edit_file" => {
                            if !modified.contains(&path) {
                                modified.push(path);
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
        (read, modified)
    }

    fn tool_path(name: &str, arguments: &str) -> Option<String> {
        if !matches!(name, "read_file" | "write_file" | "edit_file") {
            return None;
        }
        let value: serde_json::Value = serde_json::from_str(arguments).ok()?;
        value.get("path")?.as_str().map(String::from)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use openai_oxide::types::chat::{FunctionCall, ToolCall};

    fn context() -> Context {
        Context::new("be concise", 100)
    }

    fn assistant_call(id: &str, name: &str, content: Option<&str>) -> Message {
        Message::Assistant {
            content: content.map(String::from),
            tool_calls: vec![ToolCall {
                id: id.into(),
                type_: "function".into(),
                function: FunctionCall {
                    name: name.into(),
                    arguments: "{}".into(),
                },
            }],
        }
    }

    #[test]
    fn build_messages_puts_system_prompt_first() {
        let mut context = context();
        context.messages.push(Message::User {
            content: "hi".into(),
        });

        let json: Vec<String> = context
            .build_messages()
            .iter()
            .map(|m| serde_json::to_string(m).unwrap())
            .collect();

        assert_eq!(
            json,
            vec![
                r#"{"role":"system","content":"be concise"}"#,
                r#"{"role":"user","content":"hi"}"#
            ]
        );
    }

    #[test]
    fn record_usage_tracks_latest_and_totals() {
        let mut context = context();

        context.record_usage(Some(50), Some(10));
        context.record_usage(None, None);
        context.record_usage(Some(150), Some(20));

        assert!(context.needs_compaction());
        assert_eq!(context.total_prompt_tokens, 200);
        assert_eq!(context.total_completion_tokens, 30);
    }

    #[test]
    fn needs_compaction_once_past_the_threshold_not_the_whole_window() {
        let mut context = context();

        assert!(!context.needs_compaction());
        context.record_usage(Some(70), None);
        assert!(
            !context.needs_compaction(),
            "70% of the window is the threshold, not past it"
        );
        context.record_usage(Some(71), None);
        assert!(
            context.needs_compaction(),
            "should compact well before the 100-token window is full"
        );
    }

    #[test]
    fn compact_point_keeps_the_recent_window_and_never_splits_a_tool_pair() {
        let mut context = context();
        context.messages.push(Message::User {
            content: "a".repeat(20_000).into(),
        });
        context.messages.push(assistant_call("c1", "read_file", None));
        context.messages.push(Message::Tool {
            tool_call_id: "c1".into(),
            content: "b".repeat(20_000).into(),
        });
        context.messages.push(assistant_call("c2", "read_file", None));
        context.messages.push(Message::Tool {
            tool_call_id: "c2".into(),
            content: "c".repeat(20_000).into(),
        });
        context.messages.push(Message::Assistant {
            content: Some("done".into()),
            tool_calls: vec![],
        });

        let point = context.compact_point(10_000);
        assert!(point.is_some(), "should find a cut point");
        let point = point.unwrap();
        assert!(
            !matches!(context.messages[point], Message::Tool { .. }),
            "cut point must not be a tool result"
        );
        assert!(point >= 2, "should cut before the recent window");
    }

    #[test]
    fn apply_compaction_replaces_old_messages_with_the_summary() {
        let mut context = context();
        context.messages.push(Message::User {
            content: "old".into(),
        });
        context.messages.push(assistant_call("c1", "read_file", None));
        context.messages.push(Message::Tool {
            tool_call_id: "c1".into(),
            content: "old result".into(),
        });
        context.messages.push(Message::User {
            content: "recent".into(),
        });

        context.apply_compaction("the summary".into(), 3);

        assert_eq!(context.messages.len(), 2);
        assert!(
            matches!(&context.messages[0], Message::User { content } if content.contains("the summary")),
            "first message should be the summary"
        );
        assert!(
            matches!(&context.messages[1], Message::User { content } if content == "recent"),
            "recent message should be kept"
        );
    }

    #[test]
    fn completed_turns_keep_their_tool_exchanges() {
        let mut context = context();
        context.messages.push(Message::User {
            content: "make it blue".into(),
        });
        context
            .messages
            .push(assistant_call("c1", "read_file", Some("let me check")));
        context.messages.push(Message::Tool {
            tool_call_id: "c1".into(),
            content: "file contents".into(),
        });
        context.messages.push(Message::Assistant {
            content: Some("made it blue".into()),
            tool_calls: vec![],
        });
        context.messages.push(Message::User {
            content: "now green".into(),
        });
        context.messages.push(assistant_call("c2", "bash", None));
        context.messages.push(Message::Tool {
            tool_call_id: "c2".into(),
            content: "ok".into(),
        });
        context.messages.push(Message::Assistant {
            content: Some("made it green".into()),
            tool_calls: vec![],
        });

        let json: Vec<String> = context
            .build_messages()
            .iter()
            .map(|m| serde_json::to_string(m).unwrap())
            .collect();

        assert_eq!(
            json,
            vec![
                r#"{"role":"system","content":"be concise"}"#,
                r#"{"role":"user","content":"make it blue"}"#,
                r#"{"role":"assistant","content":"let me check","tool_calls":[{"id":"c1","type":"function","function":{"name":"read_file","arguments":"{}"}}]}"#,
                r#"{"role":"tool","content":"file contents","tool_call_id":"c1"}"#,
                r#"{"role":"assistant","content":"made it blue"}"#,
                r#"{"role":"user","content":"now green"}"#,
                r#"{"role":"assistant","tool_calls":[{"id":"c2","type":"function","function":{"name":"bash","arguments":"{}"}}]}"#,
                r#"{"role":"tool","content":"ok","tool_call_id":"c2"}"#,
                r#"{"role":"assistant","content":"made it green"}"#,
            ]
        );
    }

    #[test]
    fn current_turn_keeps_its_tool_exchanges() {
        let mut context = context();
        context.messages.push(Message::User {
            content: "t1".into(),
        });
        context.messages.push(assistant_call("c1", "bash", None));
        context.messages.push(Message::Tool {
            tool_call_id: "c1".into(),
            content: "out".into(),
        });
        context.messages.push(Message::Assistant {
            content: Some("done".into()),
            tool_calls: vec![],
        });
        context.messages.push(Message::User {
            content: "t2".into(),
        });
        context
            .messages
            .push(assistant_call("c2", "read_file", None));
        context.messages.push(Message::Tool {
            tool_call_id: "c2".into(),
            content: "content".into(),
        });

        let json: Vec<String> = context
            .build_messages()
            .iter()
            .map(|m| serde_json::to_string(m).unwrap())
            .collect();

        assert_eq!(
            json,
            vec![
                r#"{"role":"system","content":"be concise"}"#,
                r#"{"role":"user","content":"t1"}"#,
                r#"{"role":"assistant","tool_calls":[{"id":"c1","type":"function","function":{"name":"bash","arguments":"{}"}}]}"#,
                r#"{"role":"tool","content":"out","tool_call_id":"c1"}"#,
                r#"{"role":"assistant","content":"done"}"#,
                r#"{"role":"user","content":"t2"}"#,
                r#"{"role":"assistant","tool_calls":[{"id":"c2","type":"function","function":{"name":"read_file","arguments":"{}"}}]}"#,
                r#"{"role":"tool","content":"content","tool_call_id":"c2"}"#,
            ]
        );

        assert_eq!(context.messages.len(), 7);
    }

    #[test]
    fn seal_fills_in_only_the_unanswered_calls_and_is_idempotent() {
        let mut context = context();
        context.messages.push(Message::User {
            content: "go".into(),
        });
        context.messages.push(Message::Assistant {
            content: None,
            tool_calls: vec![
                ToolCall {
                    id: "c1".into(),
                    type_: "function".into(),
                    function: FunctionCall {
                        name: "read_file".into(),
                        arguments: "{}".into(),
                    },
                },
                ToolCall {
                    id: "c2".into(),
                    type_: "function".into(),
                    function: FunctionCall {
                        name: "submit_plan".into(),
                        arguments: "{}".into(),
                    },
                },
            ],
        });
        context.messages.push(Message::Tool {
            tool_call_id: "c1".into(),
            content: "answered".into(),
        });

        assert_eq!(context.seal_dangling_tool_calls(), 1);
        assert_eq!(context.messages.len(), 4);
        assert!(
            matches!(&context.messages[2], Message::Tool { tool_call_id, content }
                if tool_call_id == "c1" && content == "answered"),
            "the real result keeps its place"
        );
        assert!(
            matches!(&context.messages[3], Message::Tool { tool_call_id, .. } if tool_call_id == "c2"),
            "the stub lands directly after it, still inside the same run of results"
        );

        assert_eq!(
            context.seal_dangling_tool_calls(),
            0,
            "a sealed context must not grow on every turn"
        );
        assert_eq!(context.messages.len(), 4);
    }

    #[test]
    fn seal_leaves_a_fully_answered_history_untouched() {
        let mut context = context();
        context.messages.push(Message::User {
            content: "go".into(),
        });
        context.messages.push(assistant_call("c1", "bash", None));
        context.messages.push(Message::Tool {
            tool_call_id: "c1".into(),
            content: "out".into(),
        });
        context.messages.push(Message::Assistant {
            content: Some("done".into()),
            tool_calls: vec![],
        });
        let before = context.messages.clone();

        assert_eq!(context.seal_dangling_tool_calls(), 0);
        assert_eq!(context.messages, before);
    }

    #[test]
    fn compact_point_declines_when_the_cut_would_summarize_nothing() {
        let mut context = context();
        // Only the very first message is big enough to fill the recent window, so
        // the cut lands at 0 and there is nothing ahead of it to summarize.
        context.messages.push(Message::User {
            content: "x".repeat(40_000).into(),
        });
        for _ in 0..3 {
            context.messages.push(Message::User {
                content: "y".into(),
            });
        }

        assert_eq!(
            context.compact_point(10_000),
            None,
            "cutting at 0 would summarize an empty prefix and prepend it to an \
             already-too-big context — one summarization call per round, forever"
        );
    }

    #[test]
    fn apply_compaction_reestimates_prompt_tokens_so_it_does_not_retrigger() {
        let mut context = context();
        context.messages.push(Message::User {
            content: "old".repeat(10_000).into(),
        });
        context.messages.push(Message::User {
            content: "recent".into(),
        });
        context.record_usage(Some(9_000), None);
        assert!(context.needs_compaction());

        context.apply_compaction("the summary".into(), 1);

        assert!(
            !context.needs_compaction(),
            "prompt_tokens still reported the pre-compaction size, so the next round \
             would summarize the summary: {:?}",
            context.prompt_tokens
        );
    }

    #[test]
    fn every_request_prefix_is_a_prefix_of_the_next_one() {
        let mut context = context();
        let mut snapshots: Vec<Vec<String>> = Vec::new();
        let mut snapshot = |context: &Context, snapshots: &mut Vec<Vec<String>>| {
            snapshots.push(
                context
                    .build_messages()
                    .iter()
                    .map(|m| serde_json::to_string(m).unwrap())
                    .collect(),
            );
        };

        context.messages.push(Message::User {
            content: "t1".into(),
        });
        snapshot(&context, &mut snapshots);
        context.messages.push(assistant_call("c1", "read_file", None));
        context.messages.push(Message::Tool {
            tool_call_id: "c1".into(),
            content: "file contents".into(),
        });
        snapshot(&context, &mut snapshots);
        context.messages.push(Message::Assistant {
            content: Some("done".into()),
            tool_calls: vec![],
        });
        context.messages.push(Message::User {
            content: "t2".into(),
        });
        snapshot(&context, &mut snapshots);
        context.messages.push(assistant_call("c2", "bash", None));
        context.messages.push(Message::Tool {
            tool_call_id: "c2".into(),
            content: "out".into(),
        });
        snapshot(&context, &mut snapshots);

        for pair in snapshots.windows(2) {
            let (earlier, later) = (&pair[0], &pair[1]);
            assert!(
                later.len() >= earlier.len() && later[..earlier.len()] == earlier[..],
                "a request rewrote an earlier message, invalidating the cached prefix\nearlier: {earlier:#?}\nlater: {later:#?}"
            );
        }
    }

    #[test]
    fn dedup_only_applies_to_read_file_not_bash() {
        let mut context = context();
        context.messages.push(Message::User {
            content: "plan it".into(),
        });
        context.messages.push(bash_call("c1", "cmd1"));
        context.messages.push(Message::Tool {
            tool_call_id: "c1".into(),
            content: "OUT1\n".repeat(10).into(),
        });
        context.messages.push(bash_call("c2", "cmd2"));
        context.messages.push(Message::Tool {
            tool_call_id: "c2".into(),
            content: "OUT2\n".repeat(10).into(),
        });

        let json: Vec<String> = context
            .build_messages()
            .iter()
            .map(|m| serde_json::to_string(m).unwrap())
            .collect();
        assert!(json.iter().any(|j| j.contains("OUT1")), "first bash result should be kept: {json:?}");
        assert!(json.iter().any(|j| j.contains("OUT2")), "second bash result should be kept: {json:?}");
    }

    #[test]
    fn dedup_keeps_different_ranges_of_the_same_file() {
        let mut context = context();
        context.messages.push(Message::User {
            content: "plan it".into(),
        });
        context.messages.push(read_range_call("c1", "a.rs", Some(1), Some(250)));
        context.messages.push(Message::Tool {
            tool_call_id: "c1".into(),
            content: "RANGE1\n".repeat(10).into(),
        });
        context.messages.push(read_range_call("c2", "a.rs", Some(251), Some(500)));
        context.messages.push(Message::Tool {
            tool_call_id: "c2".into(),
            content: "RANGE2\n".repeat(10).into(),
        });

        let json: Vec<String> = context
            .build_messages()
            .iter()
            .map(|m| serde_json::to_string(m).unwrap())
            .collect();
        assert!(json.iter().any(|j| j.contains("RANGE1")), "first range should be kept: {json:?}");
        assert!(json.iter().any(|j| j.contains("RANGE2")), "second range should be kept: {json:?}");
    }

    fn read_range_call(
        id: &str,
        path: &str,
        start: Option<usize>,
        end: Option<usize>,
    ) -> Message {
        let mut args = format!("{{\"path\":\"{path}\"");
        if let Some(start) = start {
            args.push_str(&format!(",\"start\":{start}"));
        }
        if let Some(end) = end {
            args.push_str(&format!(",\"end\":{end}"));
        }
        args.push('}');
        Message::Assistant {
            content: None,
            tool_calls: vec![ToolCall {
                id: id.into(),
                type_: "function".into(),
                function: FunctionCall {
                    name: "read_file".into(),
                    arguments: args,
                },
            }],
        }
    }

    fn bash_call(id: &str, command: &str) -> Message {
        Message::Assistant {
            content: None,
            tool_calls: vec![ToolCall {
                id: id.into(),
                type_: "function".into(),
                function: FunctionCall {
                    name: "bash".into(),
                    arguments: format!("{{\"command\":\"{command}\"}}"),
                },
            }],
        }
    }

}
