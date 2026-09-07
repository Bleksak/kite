use openai_oxide::types::chat::{DeltaToolCall, FunctionCall, ToolCall};
use serde::Deserialize;

use crate::message::Message;
use crate::tool::Question;

#[derive(Debug, PartialEq, Eq, Clone, Default)]
pub struct ChunkTokens {
    pub thinking: Option<String>,
    pub text: Option<String>,
}

#[derive(Debug, Clone)]
pub enum AgentEvent {
    CompletionStarted,
    Tokens(ChunkTokens),
    ToolStarted {
        header: String,
        body: Option<String>,
    },
    ToolResult {
        header: String,
        body: String,
    },
    AskUser {
        questions: Vec<Question>,
        reply: tokio::sync::watch::Sender<Option<Vec<String>>>,
    },
    BgTaskDone {
        id: String,
        command: String,
        code: Option<i32>,
    },
}

impl PartialEq for AgentEvent {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::CompletionStarted, Self::CompletionStarted) => true,
            (Self::Tokens(a), Self::Tokens(b)) => a == b,
            (Self::ToolStarted { header: h1, body: b1 }, Self::ToolStarted { header: h2, body: b2 }) => {
                h1 == h2 && b1 == b2
            }
            (Self::ToolResult { header: h1, body: b1 }, Self::ToolResult { header: h2, body: b2 }) => {
                h1 == h2 && b1 == b2
            }
            (Self::AskUser { questions: q1, .. }, Self::AskUser { questions: q2, .. }) => q1 == q2,
            (
                Self::BgTaskDone { id: i1, command: c1, code: k1 },
                Self::BgTaskDone { id: i2, command: c2, code: k2 },
            ) => i1 == i2 && c1 == c2 && k1 == k2,
            _ => false,
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum ThinkingMode {
    Live,
    Hidden,
}

pub struct AnswerGate {
    answer_started: bool,
}

impl AnswerGate {
    pub fn new() -> AnswerGate {
        AnswerGate {
            answer_started: false,
        }
    }

    pub fn on_chunk(&mut self, chunk: &ChunkTokens) -> (ThinkingMode, Option<String>) {
        let thinking = &chunk.thinking;
        let text = &chunk.text;

        let thinking_mode = if thinking.is_some() && !self.answer_started {
            ThinkingMode::Live
        } else {
            ThinkingMode::Hidden
        };

        let mut out = None;
        if let Some(t) = text {
            self.answer_started = true;
            out = Some(t.clone());
        }

        (thinking_mode, out)
    }
}

impl Default for AnswerGate {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Deserialize)]
pub struct StreamChunk {
    #[serde(default)]
    choices: Vec<StreamChoice>,
    #[serde(default)]
    usage: Option<StreamUsage>,
}

#[derive(Deserialize)]
struct StreamUsage {
    #[serde(default)]
    prompt_tokens: Option<u64>,
    #[serde(default)]
    completion_tokens: Option<u64>,
}

#[derive(Deserialize)]
struct StreamChoice {
    #[serde(default)]
    delta: Option<StreamDelta>,
}

#[derive(Deserialize)]
struct StreamDelta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default, alias = "reasoning_content")]
    reasoning: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<DeltaToolCall>>,
}

#[derive(Default)]
pub struct ToolSlot {
    id: String,
    name: String,
    arguments: String,
}

pub struct StreamAccumulator {
    content: String,
    tool_slots: Vec<ToolSlot>,
    usage: (Option<u64>, Option<u64>),
}

impl StreamAccumulator {
    pub fn new() -> StreamAccumulator {
        StreamAccumulator {
            content: String::new(),
            tool_slots: Vec::new(),
            usage: (None, None),
        }
    }

    pub fn feed(&mut self, chunk: StreamChunk) -> Vec<ChunkTokens> {
        let mut out = Vec::new();

        if let Some(usage) = chunk.usage {
            self.usage = (usage.prompt_tokens, usage.completion_tokens);
        }

        for choice in chunk.choices {
            let Some(delta) = choice.delta else {
                continue;
            };
            let mut tokens = ChunkTokens::default();

            if let Some(text) = delta.content.filter(|t| !t.is_empty()) {
                self.content.push_str(&text);
                tokens.text = Some(text);
            }

            if let Some(reasoning) = delta.reasoning.filter(|r| !r.is_empty()) {
                tokens.thinking = Some(reasoning);
            }

            if let Some(tool_calls) = delta.tool_calls {
                for call in tool_calls {
                    let index = call.index;
                    if index < 0 {
                        continue;
                    }
                    let index = index as usize;
                    while self.tool_slots.len() <= index {
                        self.tool_slots.push(ToolSlot::default());
                    }

                    let slot = &mut self.tool_slots[index];
                    if let Some(id) = call.id {
                        slot.id = id;
                    }
                    if let Some(function) = call.function {
                        if let Some(name) = function.name {
                            slot.name = name;
                        }
                        if let Some(arguments) = function.arguments {
                            slot.arguments.push_str(&arguments);
                        }
                    }
                }
            }

            if tokens.thinking.is_some() || tokens.text.is_some() {
                out.push(tokens);
            }
        }

        out
    }

    pub fn usage(&self) -> (Option<u64>, Option<u64>) {
        self.usage
    }

    pub fn into_message(self) -> Message {
        Message::Assistant {
            content: if self.content.is_empty() {
                None
            } else {
                Some(self.content)
            },
            tool_calls: self
                .tool_slots
                .into_iter()
                .enumerate()
                .filter(|(_, slot)| {
                    !slot.id.is_empty() || !slot.name.is_empty() || !slot.arguments.is_empty()
                })
                .map(|(position, slot)| ToolCall {
                    id: if slot.id.is_empty() {
                        format!("call_{position}")
                    } else {
                        slot.id
                    },
                    type_: "function".into(),
                    function: FunctionCall {
                        name: slot.name,
                        arguments: slot.arguments,
                    },
                })
                .collect(),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn feed_json(acc: &mut StreamAccumulator, json: &str) -> Vec<ChunkTokens> {
        let chunk: StreamChunk = serde_json::from_str(json).unwrap();
        acc.feed(chunk)
    }

    #[test]
    fn text_deltas_accumulate_in_order() {
        let mut acc = StreamAccumulator::new();

        let t1 = feed_json(&mut acc, r#"{"choices":[{"delta":{"content":"Hello "}}]}"#);
        let t2 = feed_json(&mut acc, r#"{"choices":[{"delta":{"content":"world"}}]}"#);

        assert_eq!(
            t1,
            vec![ChunkTokens {
                thinking: None,
                text: Some("Hello ".into())
            }]
        );
        assert_eq!(
            t2,
            vec![ChunkTokens {
                thinking: None,
                text: Some("world".into())
            }]
        );
        assert_eq!(
            acc.into_message(),
            Message::Assistant {
                content: Some("Hello world".into()),
                tool_calls: vec![],
            }
        );
    }

    #[test]
    fn empty_content_delta_is_ignored() {
        let mut acc = StreamAccumulator::new();

        let tokens = feed_json(
            &mut acc,
            r#"{"choices":[{"delta":{"role":"assistant","content":""}}]}"#,
        );

        assert!(tokens.is_empty());
    }

    #[test]
    fn reasoning_deltas_are_emitted_but_not_in_message() {
        let mut acc = StreamAccumulator::new();

        let t1 = feed_json(
            &mut acc,
            r#"{"choices":[{"delta":{"reasoning":"Let me think. "}}]}"#,
        );
        let t2 = feed_json(&mut acc, r#"{"choices":[{"delta":{"content":"42"}}]}"#);

        assert_eq!(
            t1,
            vec![ChunkTokens {
                thinking: Some("Let me think. ".into()),
                text: None
            }]
        );
        assert_eq!(
            t2,
            vec![ChunkTokens {
                thinking: None,
                text: Some("42".into())
            }]
        );
        assert_eq!(
            acc.into_message(),
            Message::Assistant {
                content: Some("42".into()),
                tool_calls: vec![],
            }
        );
    }

    #[test]
    fn tool_call_arguments_reassemble_across_chunks() {
        let mut acc = StreamAccumulator::new();

        let t1 = feed_json(
            &mut acc,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"bash"}}]}}]}"#,
        );
        let t2 = feed_json(
            &mut acc,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"command\":"}}]}}]}"#,
        );
        let t3 = feed_json(
            &mut acc,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"echo hi\"}"}}]}}]}"#,
        );

        assert!(t1.is_empty() && t2.is_empty() && t3.is_empty());
        let Message::Assistant {
            content,
            tool_calls,
        } = acc.into_message()
        else {
            panic!("expected assistant message");
        };
        assert_eq!(content, None);
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id, "call_1");
        assert_eq!(tool_calls[0].function.name, "bash");
        assert_eq!(tool_calls[0].function.arguments, r#"{"command":"echo hi"}"#);
    }

    #[test]
    fn negative_tool_call_indices_are_dropped() {
        let mut acc = StreamAccumulator::new();

        feed_json(
            &mut acc,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":-1,"id":"call_bad","function":{"name":"bash","arguments":"{\"command\":\"x\"}"}}]}}]}"#,
        );
        feed_json(
            &mut acc,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_ok","function":{"name":"bash","arguments":"{\"command\":\"ls\"}"}}]}}]}"#,
        );

        let Message::Assistant { tool_calls, .. } = acc.into_message() else {
            panic!("expected assistant message");
        };
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id, "call_ok");
    }

    #[test]
    fn parallel_tool_calls_keep_their_indices() {
        let mut acc = StreamAccumulator::new();

        feed_json(
            &mut acc,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":1,"id":"call_b","function":{"name":"bash","arguments":"{\"command\":\"ls\"}"}}]}}]}"#,
        );
        feed_json(
            &mut acc,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_a","function":{"name":"read_file","arguments":"{\"path\":\"a\"}"}}]}}]}"#,
        );

        let Message::Assistant { tool_calls, .. } = acc.into_message() else {
            panic!("expected assistant message");
        };
        assert_eq!(
            tool_calls.iter().map(|c| c.id.clone()).collect::<Vec<_>>(),
            vec!["call_a", "call_b"]
        );
        assert_eq!(
            tool_calls
                .iter()
                .map(|c| c.function.name.clone())
                .collect::<Vec<_>>(),
            vec!["read_file", "bash"]
        );
    }

    #[test]
    fn usage_chunk_after_finish_is_captured() {
        let mut acc = StreamAccumulator::new();

        feed_json(&mut acc, r#"{"choices":[{"delta":{"content":"hi"}}]}"#);
        feed_json(
            &mut acc,
            r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#,
        );
        feed_json(
            &mut acc,
            r#"{"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":7}}"#,
        );

        assert_eq!(acc.usage(), (Some(12), Some(7)));
    }

    #[test]
    fn chunk_without_delta_is_ignored() {
        let mut acc = StreamAccumulator::new();

        let tokens = feed_json(&mut acc, r#"{"choices":[{"index":0}]}"#);

        assert!(tokens.is_empty());
        let Message::Assistant {
            content,
            tool_calls,
        } = acc.into_message()
        else {
            panic!("expected assistant message");
        };
        assert_eq!(content, None);
        assert!(tool_calls.is_empty());
    }

    #[test]
    fn reasoning_content_alias_is_surfaced_as_thinking() {
        let mut acc = StreamAccumulator::new();

        let tokens = feed_json(
            &mut acc,
            r#"{"choices":[{"delta":{"reasoning_content":"hmm"}}]}"#,
        );

        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].thinking.as_deref(), Some("hmm"));
        let Message::Assistant { content, .. } = acc.into_message() else {
            panic!("expected assistant message");
        };
        assert_eq!(content, None);
    }

    #[test]
    fn missing_tool_call_id_is_synthesized() {
        let mut acc = StreamAccumulator::new();

        feed_json(
            &mut acc,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"bash","arguments":"{\"command\":\"ls\"}"}}]}}]}"#,
        );

        let Message::Assistant { tool_calls, .. } = acc.into_message() else {
            panic!("expected assistant message");
        };
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id, "call_0");
        assert_eq!(tool_calls[0].function.name, "bash");
    }

    fn gate_chunk(thinking: Option<&str>, text: Option<&str>) -> ChunkTokens {
        ChunkTokens {
            thinking: thinking.map(String::from),
            text: text.map(String::from),
        }
    }

    #[test]
    fn gate_streams_thinking_live_until_answer_starts() {
        let mut gate = AnswerGate::new();

        let (mode, out) = gate.on_chunk(&gate_chunk(Some("thinking "), None));
        assert_eq!((mode, out), (ThinkingMode::Live, None));

        let (mode, out) = gate.on_chunk(&gate_chunk(None, Some("answer")));
        assert_eq!((mode, out), (ThinkingMode::Hidden, Some("answer".into())));

        let (mode, out) = gate.on_chunk(&gate_chunk(Some("straggler"), Some(" more")));
        assert_eq!((mode, out), (ThinkingMode::Hidden, Some(" more".into())));
    }

    #[test]
    fn gate_emits_text_glued_to_reasoning_immediately() {
        let mut gate = AnswerGate::new();

        let (mode, out) = gate.on_chunk(&gate_chunk(Some("reasoning"), Some("Hello")));
        assert_eq!((mode, out), (ThinkingMode::Live, Some("Hello".into())));

        let (mode, out) = gate.on_chunk(&gate_chunk(Some("more reasoning"), Some(" world")));
        assert_eq!((mode, out), (ThinkingMode::Hidden, Some(" world".into())));
    }
}
