use openai_oxide::types::chat::ChatCompletionMessageParam;
use serde::{Deserialize, Serialize};

use crate::message::Message;

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
