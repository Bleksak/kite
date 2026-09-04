use futures_util::StreamExt;
use openai_oxide::client::OpenAI;
use openai_oxide::error::OpenAIError;
use openai_oxide::types::chat::{
    ChatCompletionRequest, DeltaToolCall, FunctionCall, ToolCall,
};
use serde::Deserialize;

use crate::message::Message;
use crate::tool::{tool_definitions, Tool};

#[derive(Debug, PartialEq, Eq, Clone, Default)]
pub struct ChunkTokens {
    pub thinking: Option<String>,
    pub text: Option<String>,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum ThinkingMode {
    Live,
    Hidden,
}

pub struct AnswerGate {
    answer_started: bool,
    buffered: String,
}

impl AnswerGate {
    pub fn new() -> AnswerGate {
        AnswerGate {
            answer_started: false,
            buffered: String::new(),
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
            if self.answer_started {
                out = Some(t.clone());
            } else if thinking.is_none() {
                self.answer_started = true;
                self.buffered.push_str(t);
                out = Some(std::mem::take(&mut self.buffered));
            } else {
                self.buffered.push_str(t);
            }
        }

        (thinking_mode, out)
    }

    pub fn finish(&mut self) -> Option<String> {
        if self.answer_started || self.buffered.is_empty() {
            None
        } else {
            Some(std::mem::take(&mut self.buffered))
        }
    }
}

impl Default for AnswerGate {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Deserialize)]
struct StreamChunk {
    #[serde(default)]
    choices: Vec<StreamChoice>,
}

#[derive(Deserialize)]
struct StreamChoice {
    delta: StreamDelta,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct StreamDelta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    reasoning: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<DeltaToolCall>>,
}

struct ToolSlot {
    id: String,
    name: String,
    arguments: String,
}

impl Default for ToolSlot {
    fn default() -> Self {
        ToolSlot {
            id: String::new(),
            name: String::new(),
            arguments: String::new(),
        }
    }
}

struct StreamAccumulator {
    content: String,
    tool_slots: Vec<ToolSlot>,
    finished: bool,
}

impl StreamAccumulator {
    fn new() -> StreamAccumulator {
        StreamAccumulator {
            content: String::new(),
            tool_slots: Vec::new(),
            finished: false,
        }
    }

    fn feed(&mut self, chunk: StreamChunk) -> Vec<ChunkTokens> {
        let mut out = Vec::new();

        for choice in chunk.choices {
            if choice.finish_reason.is_some() {
                self.finished = true;
            }

            let mut tokens = ChunkTokens::default();

            if let Some(text) = choice.delta.content.filter(|t| !t.is_empty()) {
                self.content.push_str(&text);
                tokens.text = Some(text);
            }

            if let Some(reasoning) = choice.delta.reasoning.filter(|r| !r.is_empty()) {
                tokens.thinking = Some(reasoning);
            }

            if let Some(tool_calls) = choice.delta.tool_calls {
                for call in tool_calls {
                    let index = call.index as usize;
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

    fn is_finished(&self) -> bool {
        self.finished
    }

    fn into_message(self) -> Message {
        Message::Assistant {
            content: if self.content.is_empty() {
                None
            } else {
                Some(self.content)
            },
            tool_calls: self
                .tool_slots
                .into_iter()
                .filter(|slot| !slot.id.is_empty() || !slot.name.is_empty() || !slot.arguments.is_empty())
                .map(|slot| ToolCall {
                    id: slot.id,
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

#[derive(Debug, PartialEq)]
enum Step {
    Continue,
    Done(String),
}

pub struct Agent {
    client: OpenAI,
    model: String,
    system_prompt: String,
    messages: Vec<Message>,
    extra_body: Option<serde_json::Value>,
}

impl Agent {
    pub fn new(client: OpenAI, model: impl Into<String>, system_prompt: impl Into<String>) -> Agent {
        Agent {
            client,
            model: model.into(),
            system_prompt: system_prompt.into(),
            messages: Vec::new(),
            extra_body: None,
        }
    }

    pub fn with_extra_body(mut self, extra: serde_json::Value) -> Agent {
        self.extra_body = Some(extra);
        self
    }

    pub fn history(&self) -> &[Message] {
        &self.messages
    }

    pub async fn chat(
        &mut self,
        user_message: &str,
        on_token: &mut impl FnMut(ChunkTokens),
    ) -> Result<String, OpenAIError> {
        self.messages.push(Message::User {
            content: user_message.to_string(),
        });

        loop {
            let message = self.stream_completion(self.build_request(), on_token).await?;
            if let Step::Done(text) = self.handle_response(message).await {
                return Ok(text);
            }
        }
    }

    async fn handle_response(&mut self, message: Message) -> Step {
        let Message::Assistant { content, tool_calls } = message else {
            return Step::Done(String::new());
        };

        if tool_calls.is_empty() {
            self.messages.push(Message::Assistant {
                content: content.clone(),
                tool_calls,
            });
            return Step::Done(content.unwrap_or_default());
        }

        self.messages.push(Message::Assistant {
            content,
            tool_calls: tool_calls.clone(),
        });

        for call in tool_calls {
            let tool_call_id = call.id.clone();
            let content = match Tool::try_from(call) {
                Ok(tool) => match tool.invoke().await {
                    Ok(output) => output,
                    Err(error) => error.to_string(),
                },
                Err(error) => error.to_string(),
            };
            self.messages.push(Message::Tool {
                tool_call_id,
                content,
            });
        }

        Step::Continue
    }

    fn build_request(&self) -> ChatCompletionRequest {
        let mut request = ChatCompletionRequest::new(
            self.model.clone(),
            std::iter::once(Message::System {
                content: self.system_prompt.clone(),
            })
            .chain(self.messages.iter().cloned())
            .map(|message| message.to_request())
            .collect(),
        );
        request.tools = Some(tool_definitions());
        request
    }

    async fn stream_completion(
        &self,
        request: ChatCompletionRequest,
        on_token: &mut impl FnMut(ChunkTokens),
    ) -> Result<Message, OpenAIError> {
        let mut body = serde_json::to_value(request)?;
        body["stream"] = serde_json::Value::Bool(true);
        if let Some(extra) = &self.extra_body
            && let (Some(map), Some(extra_map)) = (body.as_object_mut(), extra.as_object())
        {
            for (key, value) in extra_map {
                map.insert(key.clone(), value.clone());
            }
        }

        let mut stream = self.client.chat().completions().create_stream_raw(&body).await?;
        let mut accumulator = StreamAccumulator::new();

        while let Some(value) = stream.next().await {
            let value = value?;
            let chunk: StreamChunk = serde_json::from_value(value)?;
            for tokens in accumulator.feed(chunk) {
                on_token(tokens);
            }
            if accumulator.is_finished() {
                break;
            }
        }

        Ok(accumulator.into_message())
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use openai_oxide::types::chat::{FunctionCall, ToolCall};

    fn tool_call(id: &str, name: &str, arguments: &str) -> ToolCall {
        ToolCall {
            id: id.into(),
            type_: "function".into(),
            function: FunctionCall {
                name: name.into(),
                arguments: arguments.into(),
            },
        }
    }

    fn agent() -> Agent {
        Agent::new(OpenAI::new("test-key"), "test-model", "be concise")
    }

    async fn run_tool_call(agent: &mut Agent, id: &str, name: &str, arguments: &str) -> Step {
        agent.handle_response(Message::Assistant {
            content: None,
            tool_calls: vec![tool_call(id, name, arguments)],
        })
        .await
    }

    async fn finish(agent: &mut Agent, text: &str) -> Step {
        agent.handle_response(Message::Assistant {
            content: Some(text.into()),
            tool_calls: vec![],
        })
        .await
    }

    #[tokio::test]
    async fn simple_answer_returns_text() {
        let mut agent = agent();

        let step = finish(&mut agent, "hello").await;

        assert_eq!(step, Step::Done("hello".into()));
        assert_eq!(
            agent.history()[0],
            Message::Assistant {
                content: Some("hello".into()),
                tool_calls: vec![],
            }
        );
    }

    #[tokio::test]
    async fn tool_call_executes_and_continues() {
        let mut agent = agent();

        let step = run_tool_call(&mut agent, "call_1", "bash", r#"{"command":"echo out"}"#).await;

        assert_eq!(step, Step::Continue);
        assert_eq!(agent.history().len(), 2);
        assert_eq!(
            agent.history()[1],
            Message::Tool {
                tool_call_id: "call_1".into(),
                content: "out\n".into(),
            }
        );
    }

    #[tokio::test]
    async fn tool_error_is_returned_to_model() {
        let mut agent = agent();

        run_tool_call(&mut agent, "call_1", "bash", r#"{"command":"exit 3"}"#).await;

        assert_eq!(
            agent.history()[1],
            Message::Tool {
                tool_call_id: "call_1".into(),
                content: "bash exited with 3\n".into(),
            }
        );
    }

    #[tokio::test]
    async fn invalid_arguments_are_returned_to_model() {
        let mut agent = agent();

        run_tool_call(&mut agent, "call_1", "bash", r#"{"command":"ls""#).await;

        let Message::Tool { content, .. } = &agent.history()[1] else {
            panic!("expected tool message");
        };
        assert!(content.contains("invalid arguments for bash"));
        assert!(content.contains(r#"{"command":"ls""#));
    }

    #[tokio::test]
    async fn unknown_tool_is_returned_to_model() {
        let mut agent = agent();

        run_tool_call(&mut agent, "call_1", "nuke", "{}").await;

        assert_eq!(
            agent.history()[1],
            Message::Tool {
                tool_call_id: "call_1".into(),
                content: "unknown tool nuke".into(),
            }
        );
    }

    #[tokio::test]
    async fn parallel_tool_calls_all_get_results() {
        let mut agent = agent();
        let step = agent
            .handle_response(Message::Assistant {
                content: None,
                tool_calls: vec![
                    tool_call("call_1", "bash", r#"{"command":"echo one"}"#),
                    tool_call("call_2", "bash", r#"{"command":"echo two"}"#),
                ],
            })
            .await;
        assert_eq!(step, Step::Continue);

        assert_eq!(
            agent.history()[1],
            Message::Tool {
                tool_call_id: "call_1".into(),
                content: "one\n".into(),
            }
        );
        assert_eq!(
            agent.history()[2],
            Message::Tool {
                tool_call_id: "call_2".into(),
                content: "two\n".into(),
            }
        );
    }

    #[test]
    fn request_contains_system_prompt_and_tools() {
        let mut agent = agent();
        agent.messages.push(Message::User {
            content: "hello".into(),
        });

        let request = agent.build_request();

        assert_eq!(request.model, "test-model");
        assert_eq!(request.messages.len(), 2);

        let openai_oxide::types::chat::ChatCompletionMessageParam::System { content, .. } =
            &request.messages[0]
        else {
            panic!("expected system message first");
        };
        assert_eq!(content, "be concise");

        let openai_oxide::types::chat::ChatCompletionMessageParam::User { content, .. } =
            &request.messages[1]
        else {
            panic!("expected user message second");
        };
        let openai_oxide::types::chat::UserContent::Text(text) = content else {
            panic!("expected text content");
        };
        assert_eq!(text, "hello");

        let tools = request.tools.as_ref().unwrap();
        assert_eq!(tools.len(), 4);
        assert_eq!(
            tools.iter().map(|t| t.function.name.as_str()).collect::<Vec<_>>(),
            vec!["bash", "read_file", "write_file", "edit_file"]
        );
    }

    fn feed_json(acc: &mut StreamAccumulator, json: &str) -> Vec<ChunkTokens> {
        let chunk: StreamChunk = serde_json::from_str(json).unwrap();
        acc.feed(chunk)
    }

    #[test]
    fn text_deltas_accumulate_in_order() {
        let mut acc = StreamAccumulator::new();

        let t1 = feed_json(&mut acc, r#"{"choices":[{"delta":{"content":"Hello "}}]}"#);
        let t2 = feed_json(&mut acc, r#"{"choices":[{"delta":{"content":"world"}}]}"#);

        assert_eq!(t1, vec![ChunkTokens { thinking: None, text: Some("Hello ".into()) }]);
        assert_eq!(t2, vec![ChunkTokens { thinking: None, text: Some("world".into()) }]);
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

        let tokens = feed_json(&mut acc, r#"{"choices":[{"delta":{"role":"assistant","content":""}}]}"#);

        assert!(tokens.is_empty());
    }

    #[test]
    fn reasoning_deltas_are_emitted_but_not_in_message() {
        let mut acc = StreamAccumulator::new();

        let t1 = feed_json(&mut acc, r#"{"choices":[{"delta":{"reasoning":"Let me think. "}}]}"#);
        let t2 = feed_json(&mut acc, r#"{"choices":[{"delta":{"content":"42"}}]}"#);

        assert_eq!(t1, vec![ChunkTokens { thinking: Some("Let me think. ".into()), text: None }]);
        assert_eq!(t2, vec![ChunkTokens { thinking: None, text: Some("42".into()) }]);
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
        let Message::Assistant { content, tool_calls } = acc.into_message() else {
            panic!("expected assistant message");
        };
        assert_eq!(content, None);
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id, "call_1");
        assert_eq!(tool_calls[0].function.name, "bash");
        assert_eq!(tool_calls[0].function.arguments, r#"{"command":"echo hi"}"#);
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
            tool_calls
                .iter()
                .map(|c| c.id.clone())
                .collect::<Vec<_>>(),
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
    fn finish_reason_marks_the_stream_finished() {
        let mut acc = StreamAccumulator::new();

        assert!(!acc.is_finished());
        feed_json(&mut acc, r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#);
        assert!(acc.is_finished());
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
    fn gate_buffers_text_glued_to_reasoning() {
        let mut gate = AnswerGate::new();

        let (mode, out) = gate.on_chunk(&gate_chunk(Some("reasoning"), Some("Hello")));
        assert_eq!((mode, out), (ThinkingMode::Live, None));

        let (mode, out) = gate.on_chunk(&gate_chunk(None, Some(" world")));
        assert_eq!((mode, out), (ThinkingMode::Hidden, Some("Hello world".into())));
    }

    #[test]
    fn gate_flushes_buffered_text_when_no_pure_chunk_arrives() {
        let mut gate = AnswerGate::new();

        gate.on_chunk(&gate_chunk(Some("reasoning"), Some("all")));
        gate.on_chunk(&gate_chunk(Some("still reasoning"), Some("glued")));

        assert_eq!(gate.finish(), Some("allglued".into()));
    }

    #[test]
    fn gate_finish_is_empty_when_answer_started() {
        let mut gate = AnswerGate::new();

        gate.on_chunk(&gate_chunk(None, Some("hi")));

        assert_eq!(gate.finish(), None);
    }
}
