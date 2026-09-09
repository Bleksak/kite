use std::borrow::Cow;

use openai_oxide::types::chat::{ChatCompletionMessageParam, ToolCall};
use serde::{Deserialize, Serialize};

use crate::message::Message;

const KEEP_RECENT_ROUNDS: usize = 2;

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
        std::iter::once(Cow::Owned(Message::System {
            content: self.system_prompt.clone(),
        }))
        .chain(self.pruned_messages())
        .map(|message| message.to_request())
        .collect()
    }

    fn pruned_messages(&self) -> Vec<Cow<'_, Message>> {
        let in_progress = match self.messages.last() {
            Some(Message::Assistant { tool_calls, .. }) => !tool_calls.is_empty(),
            Some(Message::Tool { .. }) => true,
            _ => false,
        };
        let keep_from = if in_progress {
            self.messages
                .iter()
                .rposition(|message| matches!(message, Message::User { .. }))
                .map(|index| index + 1)
                .unwrap_or(0)
        } else {
            self.messages.len()
        };

        let verbatim: std::collections::HashSet<usize> = self
            .verbatim_tool_results(keep_from)
            .into_iter()
            .collect();

        self.messages
            .iter()
            .enumerate()
            .filter_map(|(index, message)| match message {
                Message::Assistant {
                    content,
                    tool_calls,
                } if !tool_calls.is_empty() && index < keep_from => match content {
                    Some(text) if !text.is_empty() => Some(Cow::Owned(Message::Assistant {
                        content: Some(text.clone()),
                        tool_calls: vec![],
                    })),
                    _ => None,
                },
                Message::Tool { .. } if index < keep_from => None,
                Message::Tool {
                    tool_call_id,
                    content,
                } if in_progress && !verbatim.contains(&index) => Some(Cow::Owned(Message::Tool {
                    tool_call_id: tool_call_id.clone(),
                    content: self.tool_result_stub(tool_call_id, content),
                })),
                _ => Some(Cow::Borrowed(message)),
            })
            .collect()
    }

    fn verbatim_tool_results(&self, keep_from: usize) -> Vec<usize> {
        let round_boundaries: Vec<usize> = self
            .messages
            .iter()
            .enumerate()
            .filter(|(index, message)| {
                *index >= keep_from
                    && matches!(message, Message::Assistant { tool_calls, .. } if !tool_calls.is_empty())
            })
            .map(|(index, _)| index)
            .collect();
        let keep_rounds: std::collections::HashSet<usize> = round_boundaries
            .iter()
            .rev()
            .take(KEEP_RECENT_ROUNDS)
            .copied()
            .collect();
        let superseded = self.superseded_tool_results(keep_from);
        let mut verbatim = Vec::new();
        let mut current_boundary: Option<usize> = None;
        for (index, message) in self.messages.iter().enumerate() {
            if index < keep_from {
                continue;
            }
            match message {
                Message::Assistant { tool_calls, .. } if !tool_calls.is_empty() => {
                    current_boundary = Some(index);
                }
                Message::Tool { .. } => {
                    if let Some(boundary) = current_boundary
                        && keep_rounds.contains(&boundary)
                        && !superseded.contains(&index)
                    {
                        verbatim.push(index);
                    }
                }
                _ => {}
            }
        }
        verbatim
    }

    fn superseded_tool_results(&self, keep_from: usize) -> std::collections::HashSet<usize> {
        type ReadKey = (String, Option<usize>, Option<usize>);
        let mut reads: Vec<(usize, Option<ReadKey>)> = Vec::new();
        for (index, message) in self.messages.iter().enumerate() {
            if index < keep_from {
                continue;
            }
            if let Message::Tool { tool_call_id, .. } = message {
                let key = self
                    .tool_call_for(tool_call_id)
                    .filter(|call| call.function.name == "read_file")
                    .and_then(|call| Self::read_file_range(&call.function.arguments));
                reads.push((index, key));
            }
        }
        let mut last_seen: std::collections::HashMap<ReadKey, usize> = std::collections::HashMap::new();
        for (index, key) in &reads {
            if let Some(key) = key {
                last_seen.insert(key.clone(), *index);
            }
        }
        reads
            .into_iter()
            .filter(|(index, key)| {
                matches!(key, Some(k) if last_seen.get(k) != Some(index))
            })
            .map(|(index, _)| index)
            .collect()
    }

    fn read_file_range(arguments: &str) -> Option<(String, Option<usize>, Option<usize>)> {
        let value: serde_json::Value = serde_json::from_str(arguments).ok()?;
        let path = value.get("path")?.as_str()?.to_string();
        let start = value.get("start").and_then(|v| v.as_u64()).map(|v| v as usize);
        let end = value.get("end").and_then(|v| v.as_u64()).map(|v| v as usize);
        Some((path, start, end))
    }

    fn tool_call_for(&self, tool_call_id: &str) -> Option<&ToolCall> {
        self.messages.iter().rev().find_map(|message| match message {
            Message::Assistant { tool_calls, .. } => {
                tool_calls.iter().find(|call| call.id == tool_call_id)
            }
            _ => None,
        })
    }

    fn tool_result_stub(&self, tool_call_id: &str, content: &str) -> String {
        let lines = content.lines().count();
        match self.tool_call_for(tool_call_id) {
            Some(call) => format!(
                "[{} {} — {} lines, pruned; re-run if needed]",
                call.function.name,
                Self::truncate_chars(&call.function.arguments, 80),
                lines
            ),
            None => format!("[tool result pruned; {} lines; re-run if needed]", lines),
        }
    }

    fn truncate_chars(text: &str, max: usize) -> String {
        let mut count = 0;
        for (index, _) in text.char_indices() {
            if count == max {
                return format!("{}…", &text[..index]);
            }
            count += 1;
        }
        text.to_string()
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

    pub fn needs_compaction(&self) -> bool {
        self.prompt_tokens
            .is_some_and(|tokens| tokens > self.max_tokens)
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
            if !matches!(self.messages[candidate], Message::Tool { .. }) {
                return Some(candidate);
            }
        }
        None
    }

    pub fn apply_compaction(&mut self, summary: String, cut_point: usize) {
        let kept = self.messages[cut_point..].to_vec();
        self.messages = vec![Message::User {
            content: format!("[summary of earlier context]\n{summary}"),
        }];
        self.messages.extend(kept);
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
    fn needs_compaction_only_when_over_budget() {
        let mut context = context();

        assert!(!context.needs_compaction());
        context.record_usage(Some(100), None);
        assert!(!context.needs_compaction());
        context.record_usage(Some(101), None);
        assert!(context.needs_compaction());
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
    fn completed_turns_drop_their_tool_exchanges() {
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
                r#"{"role":"assistant","content":"let me check"}"#,
                r#"{"role":"assistant","content":"made it blue"}"#,
                r#"{"role":"user","content":"now green"}"#,
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
                r#"{"role":"assistant","content":"done"}"#,
                r#"{"role":"user","content":"t2"}"#,
                r#"{"role":"assistant","tool_calls":[{"id":"c2","type":"function","function":{"name":"read_file","arguments":"{}"}}]}"#,
                r#"{"role":"tool","content":"content","tool_call_id":"c2"}"#,
            ]
        );

        assert_eq!(context.messages.len(), 7);
    }

    #[test]
    fn current_turn_stubs_old_rounds_and_keeps_the_recent_rounds() {
        let mut context = context();
        context.messages.push(Message::User {
            content: "plan it".into(),
        });
        context.messages.push(assistant_call("c1", "read_file", Some("reading a")));
        context.messages.push(Message::Tool {
            tool_call_id: "c1".into(),
            content: "AAAA\n".repeat(100).into(),
        });
        context.messages.push(assistant_call("c2", "read_file", Some("reading b")));
        context.messages.push(Message::Tool {
            tool_call_id: "c2".into(),
            content: "BBBB\n".repeat(100).into(),
        });
        context.messages.push(assistant_call("c3", "read_file", None));
        context.messages.push(Message::Tool {
            tool_call_id: "c3".into(),
            content: "CCCC\n".repeat(100).into(),
        });

        let json: Vec<String> = context
            .build_messages()
            .iter()
            .map(|m| serde_json::to_string(m).unwrap())
            .collect();
        assert!(
            json.iter().any(|j| j.contains("\"tool_call_id\":\"c1\"") && j.contains("pruned")),
            "round 1 should be stubbed: {json:?}"
        );
        assert!(!json.iter().any(|j| j.contains("AAAA")), "round 1 content should be gone: {json:?}");
        assert!(json.iter().any(|j| j.contains("BBBB")), "round 2 should be verbatim: {json:?}");
        assert!(json.iter().any(|j| j.contains("CCCC")), "round 3 should be verbatim: {json:?}");
    }

    #[test]
    fn a_round_with_many_tool_calls_is_never_partially_stubbed() {
        let mut context = context();
        context.messages.push(Message::User {
            content: "plan it".into(),
        });
        context.messages.push(assistant_call("c1", "read_file", Some("old")));
        context.messages.push(Message::Tool {
            tool_call_id: "c1".into(),
            content: "OLD\n".repeat(50).into(),
        });
        context.messages.push(assistant_call("c2", "read_file", Some("mid")));
        context.messages.push(Message::Tool {
            tool_call_id: "c2".into(),
            content: "MID\n".repeat(50).into(),
        });
        let tool_calls = (0..5)
            .map(|i| ToolCall {
                id: format!("r{i}"),
                type_: "function".into(),
                function: FunctionCall {
                    name: "read_file".into(),
                    arguments: format!("{{\"path\":\"f{i}.rs\"}}"),
                },
            })
            .collect();
        context.messages.push(Message::Assistant {
            content: None,
            tool_calls,
        });
        for i in 0..5 {
            context.messages.push(Message::Tool {
                tool_call_id: format!("r{i}"),
                content: format!("FILE{i}\n").repeat(50),
            });
        }

        let json: Vec<String> = context
            .build_messages()
            .iter()
            .map(|m| serde_json::to_string(m).unwrap())
            .collect();
        assert!(!json.iter().any(|j| j.contains("OLD")), "old round should be stubbed: {json:?}");
        for i in 0..5 {
            assert!(
                json.iter().any(|j| j.contains(&format!("FILE{i}"))),
                "round result {i} should be verbatim: {json:?}"
            );
        }
    }

    #[test]
    fn within_turn_pruning_bounds_the_total_payload() {
        let mut context = context();
        context.messages.push(Message::User {
            content: "plan it".into(),
        });
        let file = "X\n".repeat(5000);
        let unique = file.len() * 9;
        let mut total_sent = 0usize;
        for round in 0..9 {
            let request = context.build_messages();
            total_sent += request
                .iter()
                .map(|m| serde_json::to_string(m).unwrap().len())
                .sum::<usize>();
            context.messages.push(assistant_call(
                &format!("c{round}"),
                "read_file",
                None,
            ));
            context.messages.push(Message::Tool {
                tool_call_id: format!("c{round}"),
                content: file.clone(),
            });
        }
        let amplification = total_sent as f64 / unique as f64;
        assert!(
            amplification < 3.0,
            "amplification {amplification:.2} should be bounded (total {total_sent} vs unique {unique})"
        );
    }

    #[test]
    fn dedup_stubs_the_earlier_read_of_the_same_path_even_in_the_recent_rounds() {
        let mut context = context();
        context.messages.push(Message::User {
            content: "plan it".into(),
        });
        context.messages.push(read_call("c1", "b.rs"));
        context.messages.push(Message::Tool {
            tool_call_id: "c1".into(),
            content: "BBB\n".repeat(10).into(),
        });
        context.messages.push(read_call("c2", "a.rs"));
        context.messages.push(Message::Tool {
            tool_call_id: "c2".into(),
            content: "OLD\n".repeat(10).into(),
        });
        context.messages.push(read_call("c3", "a.rs"));
        context.messages.push(Message::Tool {
            tool_call_id: "c3".into(),
            content: "NEW\n".repeat(10).into(),
        });

        let json: Vec<String> = context
            .build_messages()
            .iter()
            .map(|m| serde_json::to_string(m).unwrap())
            .collect();
        assert!(!json.iter().any(|j| j.contains("BBB")), "b.rs (round 1) should be stubbed: {json:?}");
        assert!(
            !json.iter().any(|j| j.contains("OLD")),
            "earlier a.rs read should be stubbed by dedup: {json:?}"
        );
        assert!(json.iter().any(|j| j.contains("NEW")), "most recent a.rs read should be verbatim: {json:?}");
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

    fn read_call(id: &str, path: &str) -> Message {
        Message::Assistant {
            content: None,
            tool_calls: vec![ToolCall {
                id: id.into(),
                type_: "function".into(),
                function: FunctionCall {
                    name: "read_file".into(),
                    arguments: format!("{{\"path\":\"{path}\"}}"),
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

    #[test]
    fn the_stub_is_descriptive_and_keeps_the_tool_call_id() {
        let mut context = context();
        context.messages.push(Message::User {
            content: "plan it".into(),
        });
        context.messages.push(assistant_call("c1", "read_file", Some("a")));
        context.messages.push(Message::Tool {
            tool_call_id: "c1".into(),
            content: "x\n".repeat(3).into(),
        });
        context.messages.push(assistant_call("c2", "read_file", Some("b")));
        context.messages.push(Message::Tool {
            tool_call_id: "c2".into(),
            content: "y\n".repeat(3).into(),
        });
        context.messages.push(assistant_call("c3", "read_file", None));
        context.messages.push(Message::Tool {
            tool_call_id: "c3".into(),
            content: "z\n".repeat(3).into(),
        });

        let json: Vec<String> = context
            .build_messages()
            .iter()
            .map(|m| serde_json::to_string(m).unwrap())
            .collect();
        let stub = json
            .iter()
            .find(|j| j.contains("\"tool_call_id\":\"c1\""))
            .expect("stubbed tool result present");
        assert!(stub.contains("read_file"), "stub names the tool: {stub}");
        assert!(stub.contains("3 lines"), "stub states the line count: {stub}");
        assert!(stub.contains("pruned"), "stub says it was pruned: {stub}");
    }
}
