use std::borrow::Cow;

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

        self.messages
            .iter()
            .enumerate()
            .filter_map(|(index, message)| match message {
                Message::Assistant { content, tool_calls }
                    if !tool_calls.is_empty() && index < keep_from =>
                {
                    match content {
                        Some(text) if !text.is_empty() => Some(Cow::Owned(
                            Message::Assistant {
                                content: Some(text.clone()),
                                tool_calls: vec![],
                            },
                        )),
                        _ => None,
                    }
                }
                Message::Tool { .. } if index < keep_from => None,
                _ => Some(Cow::Borrowed(message)),
            })
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

    pub fn compact(&mut self) {}
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
    fn compact_is_a_no_op_for_now() {
        let mut context = context();
        context.messages.push(Message::User {
            content: "hi".into(),
        });

        context.compact();

        assert_eq!(context.messages.len(), 1);
    }

    #[test]
    fn completed_turns_drop_their_tool_exchanges() {
        let mut context = context();
        context.messages.push(Message::User {
            content: "make it blue".into(),
        });
        context.messages.push(assistant_call("c1", "read_file", Some("let me check")));
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
        context.messages.push(assistant_call("c2", "read_file", None));
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
}
