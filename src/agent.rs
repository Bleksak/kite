use std::time::Duration;

use futures_util::future::join_all;
use futures_util::StreamExt;
use openai_oxide::client::OpenAI;
use openai_oxide::error::OpenAIError;
use openai_oxide::types::chat::{
    ChatCompletionRequest, DeltaToolCall, FunctionCall, StreamOptions, ToolCall,
};
use serde::Deserialize;

use crate::context::Context;
use crate::message::Message;
use crate::tool::{TOOL_DEFINITIONS, Tool, ToolOutput};

#[derive(Debug, PartialEq, Eq, Clone, Default)]
pub struct ChunkTokens {
    pub thinking: Option<String>,
    pub text: Option<String>,
}

#[derive(Debug, PartialEq, Clone)]
pub enum AgentEvent {
    CompletionStarted,
    Tokens(ChunkTokens),
    ToolStarted { header: String, body: Option<String> },
    ToolResult { header: String, body: String },
    BgTaskDone { id: String, command: String, code: Option<i32> },
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
struct StreamChunk {
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
struct ToolSlot {
    id: String,
    name: String,
    arguments: String,
}

struct StreamAccumulator {
    content: String,
    tool_slots: Vec<ToolSlot>,
    usage: (Option<u64>, Option<u64>),
}

impl StreamAccumulator {
    fn new() -> StreamAccumulator {
        StreamAccumulator {
            content: String::new(),
            tool_slots: Vec::new(),
            usage: (None, None),
        }
    }

    fn feed(&mut self, chunk: StreamChunk) -> Vec<ChunkTokens> {
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

    fn usage(&self) -> (Option<u64>, Option<u64>) {
        self.usage
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

#[derive(Debug, PartialEq)]
enum Step {
    Continue,
    Done(String),
}

pub struct Agent {
    pub client: OpenAI,
    pub model: String,
    pub context: Context,
    pub extra_body: Option<serde_json::Value>,
    pub bash_timeout: Duration,
    bg_seen: std::collections::HashSet<String>,
    bg_mine: std::collections::HashSet<String>,
}

const MAX_TOOL_ROUNDS: usize = 25;

impl Agent {
    pub fn new(
        client: OpenAI,
        model: impl Into<String>,
        system_prompt: impl Into<String>,
        max_tokens: u64,
        bash_timeout: Duration,
    ) -> Agent {
        Agent {
            client,
            model: model.into(),
            context: Context::new(system_prompt, max_tokens),
            extra_body: None,
            bash_timeout,
            bg_seen: std::collections::HashSet::new(),
            bg_mine: std::collections::HashSet::new(),
        }
    }

    pub fn with_extra_body(mut self, extra: serde_json::Value) -> Agent {
        self.extra_body = Some(extra);
        self
    }

    #[cfg(test)]
    pub fn history(&self) -> &[Message] {
        &self.context.messages
    }

    pub async fn chat(
        &mut self,
        user_message: &str,
        on_event: &mut impl FnMut(AgentEvent),
    ) -> Result<String, Box<dyn std::error::Error>> {
        self.context.messages.push(Message::User {
            content: user_message.to_string(),
        });
        self.run_turns(on_event).await
    }

    pub async fn bg_turn(
        &mut self,
        on_event: &mut impl FnMut(AgentEvent),
    ) -> Result<String, Box<dyn std::error::Error>> {
        self.run_turns(on_event).await
    }

    pub fn owns_and_unseen(&self, id: &str) -> bool {
        self.bg_mine.contains(id) && !self.bg_seen.contains(id)
    }

    async fn run_turns(
        &mut self,
        on_event: &mut impl FnMut(AgentEvent),
    ) -> Result<String, Box<dyn std::error::Error>> {
        let mut round = 0;
        loop {
            round += 1;
            if round > MAX_TOOL_ROUNDS {
                return Err(format!("tool loop exceeded {MAX_TOOL_ROUNDS} rounds").into());
            }
            if self.context.needs_compaction() {
                self.context.compact();
            }
            self.report_bg_tasks(on_event);
            on_event(AgentEvent::CompletionStarted);

            let (message, usage) = self
                .stream_completion(
                    self.build_request(),
                    &mut |tokens| on_event(AgentEvent::Tokens(tokens)),
                )
                .await?;
            self.context.record_usage(usage.0, usage.1);

            if let Step::Done(text) = self.handle_response(message, on_event).await {
                return Ok(text);
            }
        }
    }

    fn report_bg_tasks(&mut self, on_event: &mut impl FnMut(AgentEvent)) {
        let reports = crate::bg::REGISTRY.due_reports(&mut self.bg_seen);
        for report in reports {
            self.context.messages.push(Message::User {
                content: format!(
                    "Background task {} finished ({}).\nCommand: {}\nOutput: {}\nOutput file: {}",
                    report.id,
                    report.status_line(),
                    report.command,
                    report.tail(4096),
                    report.output_path.display(),
                ),
            });
            on_event(AgentEvent::BgTaskDone {
                id: report.id,
                command: report.command,
                code: report.code,
            });
        }
    }

    async fn handle_response(&mut self, message: Message, on_event: &mut impl FnMut(AgentEvent)) -> Step {
        let Message::Assistant { content, tool_calls } = message else {
            return Step::Done(String::new());
        };

        if tool_calls.is_empty() {
            self.context.messages.push(Message::Assistant {
                content: content.clone(),
                tool_calls,
            });
            return Step::Done(content.unwrap_or_default());
        }

        self.context.messages.push(Message::Assistant {
            content,
            tool_calls: tool_calls.clone(),
        });

        let mut index = 0;
        while index < tool_calls.len() {
            let mut end = index;
            while end < tool_calls.len()
                && tool_calls[end].function.name == "read_file"
            {
                end += 1;
            }
            if end == index {
                end = index + 1;
            }
            self.run_tools(&tool_calls[index..end], on_event).await;
            index = end;
        }

        Step::Continue
    }

    async fn run_tools(
        &mut self,
        calls: &[ToolCall],
        on_event: &mut impl FnMut(AgentEvent),
    ) {
        let mut slots: Vec<Result<(Tool, String), String>> = Vec::new();
        for call in calls {
            match Tool::try_from(call.clone()) {
                Ok(tool) => {
                    let header = tool.header();
                    let output = tool.output();
                    let body = match &output {
                        ToolOutput::Before(body) => Some(body.to_string()),
                        ToolOutput::After => None,
                    };
                    on_event(AgentEvent::ToolStarted {
                        header: header.clone(),
                        body,
                    });
                    slots.push(Ok((tool, header)));
                }
                Err(error) => slots.push(Err(error.to_string())),
            }
        }

        let futures: Vec<_> = slots
            .iter()
            .filter_map(|slot| slot.as_ref().ok().map(|(tool, _)| tool.invoke(self.bash_timeout)))
            .collect();
        let outputs = join_all(futures).await;

        let mut outputs = outputs.into_iter();
        for (call, slot) in calls.iter().zip(slots) {
            let content = match slot {
                Ok((tool, header)) => {
                    let content = match outputs.next().unwrap() {
                        Ok(output) => output,
                        Err(error) => error.to_string(),
                    };
                    if matches!(tool, Tool::BgRun(_))
                        && let Some(rest) = content.strip_prefix("task ")
                        && let Some(id) = rest.split_whitespace().next()
                    {
                        self.bg_mine.insert(id.to_string());
                    }
                    on_event(AgentEvent::ToolResult {
                        header,
                        body: content.clone(),
                    });
                    content
                }
                Err(error) => error,
            };
            self.context.messages.push(Message::Tool {
                tool_call_id: call.id.clone(),
                content,
            });
        }
    }

    fn build_request(&self) -> ChatCompletionRequest {
        let mut request = ChatCompletionRequest::new(
            self.model.clone(),
            self.context.build_messages(),
        );
        request.tools = Some((*TOOL_DEFINITIONS).clone());
        request
    }

    async fn stream_completion(
        &self,
        request: ChatCompletionRequest,
        on_token: &mut impl FnMut(ChunkTokens),
    ) -> Result<(Message, (Option<u64>, Option<u64>)), OpenAIError> {
        let mut request = request;
        request.stream = Some(true);
        request.stream_options = Some(StreamOptions {
            include_usage: Some(true),
        });

        let mut stream = if let Some(extra) = &self.extra_body {
            let mut body = serde_json::to_value(&request)?;
            if let (Some(map), Some(extra_map)) = (body.as_object_mut(), extra.as_object()) {
                for (key, value) in extra_map {
                    map.insert(key.clone(), value.clone());
                }
            }
            self.client.chat().completions().create_stream_raw(&body).await?
        } else {
            self.client.chat().completions().create_stream_raw(&request).await?
        };
        let mut accumulator = StreamAccumulator::new();

        while let Some(value) = stream.next().await {
            let value = value?;
            let chunk: StreamChunk = serde_json::from_value(value)?;
            for tokens in accumulator.feed(chunk) {
                on_token(tokens);
            }
        }

        let usage = accumulator.usage();
        Ok((accumulator.into_message(), usage))
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use openai_oxide::types::chat::{FunctionCall, ToolCall};
    use test_files::TestFiles;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

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
        Agent::new(OpenAI::new("test-key"), "test-model", "be concise", 10000, Duration::from_secs(30))
    }

    async fn run_tool_call(
        agent: &mut Agent,
        id: &str,
        name: &str,
        arguments: &str,
        events: &mut Vec<AgentEvent>,
    ) -> Step {
        agent
            .handle_response(
                Message::Assistant {
                    content: None,
                    tool_calls: vec![tool_call(id, name, arguments)],
                },
                &mut |event| events.push(event),
            )
            .await
    }

    async fn run_tool_calls(
        agent: &mut Agent,
        calls: Vec<ToolCall>,
        events: &mut Vec<AgentEvent>,
    ) -> Step {
        agent
            .handle_response(
                Message::Assistant {
                    content: None,
                    tool_calls: calls,
                },
                &mut |event| events.push(event),
            )
            .await
    }

    async fn finish(agent: &mut Agent, text: &str, events: &mut Vec<AgentEvent>) -> Step {
        agent
            .handle_response(
                Message::Assistant {
                    content: Some(text.into()),
                    tool_calls: vec![],
                },
                &mut |event| events.push(event),
            )
            .await
    }

    #[tokio::test]
    async fn bg_turn_delivers_finished_task_to_the_model() {
        fn complete_request(data: &[u8]) -> Option<String> {
            let header_end = data.windows(4).position(|w| w == b"\r\n\r\n")?;
            let headers = std::str::from_utf8(&data[..header_end]).unwrap();
            let length = headers
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    if name.trim().eq_ignore_ascii_case("content-length") {
                        value.trim().parse::<usize>().ok()
                    } else {
                        None
                    }
                })?;
            let body_start = header_end + 4;
            if data.len() < body_start + length {
                return None;
            }
            Some(String::from_utf8_lossy(&data[body_start..body_start + length]).into_owned())
        }

        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            while let Ok((mut socket, _)) = listener.accept().await {
                let tx = tx.clone();
                tokio::spawn(async move {
                    let mut data = Vec::new();
                    let mut buf = [0u8; 8192];
                    loop {
                        match socket.read(&mut buf).await {
                            Ok(0) | Err(_) => break,
                            Ok(n) => {
                                data.extend_from_slice(&buf[..n]);
                                if let Some(body) = complete_request(&data) {
                                    let _ = tx.send(body);
                                    break;
                                }
                            }
                        }
                    }
                    let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"tests reported\"}}]}\n\ndata: [DONE]\n\n";
                    let response = format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n\r\n{sse}"
                    );
                    let _ = socket.write_all(response.as_bytes()).await;
                });
            }
        });

        let client = OpenAI::with_config(
            openai_oxide::ClientConfig::new("local").base_url(format!("http://{addr}")),
        );
        let mut agent = Agent::new(client, "test-model", "be concise", 10000, Duration::from_secs(30));

        let mut events = vec![];
        run_tool_call(
            &mut agent,
            "t1",
            "bg_run",
            r#"{"command":"echo bg-e2e-marker"}"#,
            &mut events,
        )
        .await;

        for _ in 0..200 {
            if crate::bg::REGISTRY.list().iter().any(|t| {
                t.command == "echo bg-e2e-marker"
                    && matches!(t.status, crate::bg::BgStatus::Finished(_))
            }) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }

        let id = crate::bg::REGISTRY
            .list()
            .into_iter()
            .find(|t| t.command == "echo bg-e2e-marker")
            .unwrap()
            .id;
        assert!(agent.owns_and_unseen(&id));

        let answer = agent.bg_turn(&mut |event| events.push(event)).await.unwrap();
        assert_eq!(answer, "tests reported");

        let request = rx.recv().await.unwrap();
        assert!(request.contains("Background task"));
        assert!(request.contains("bg-e2e-marker"));
    }

    #[tokio::test]
    async fn simple_answer_returns_text() {
        let mut agent = agent();

        let step = finish(&mut agent, "hello", &mut Vec::new()).await;

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

        let step = run_tool_call(&mut agent, "call_1", "bash", r#"{"command":"echo out"}"#, &mut Vec::new()).await;

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

        run_tool_call(&mut agent, "call_1", "bash", r#"{"command":"exit 3"}"#, &mut Vec::new()).await;

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

        run_tool_call(&mut agent, "call_1", "bash", r#"{"command":"ls""#, &mut Vec::new()).await;

        let Message::Tool { content, .. } = &agent.history()[1] else {
            panic!("expected tool message");
        };
        assert!(content.contains("invalid arguments for bash"));
        assert!(content.contains(r#"{"command":"ls""#));
    }

    #[tokio::test]
    async fn mixed_round_runs_the_read_batch_in_order_with_the_write() {
        let mut agent = agent();
        let temp_dir = TestFiles::new();
        temp_dir.file("a.txt", "one\n");
        temp_dir.file("b.txt", "two\n");
        let file_a = temp_dir.path().join("a.txt");
        let file_b = temp_dir.path().join("b.txt");
        let file_c = temp_dir.path().join("c.txt");

        let step = run_tool_calls(
            &mut agent,
            vec![
                tool_call("c1", "read_file", &format!("{{\"path\":\"{}\"}}", file_a.to_string_lossy())),
                tool_call("c2", "read_file", &format!("{{\"path\":\"{}\"}}", file_b.to_string_lossy())),
                tool_call("c3", "write_file", &format!("{{\"path\":\"{}\",\"content\":\"three\"}}", file_c.to_string_lossy())),
            ],
            &mut Vec::new(),
        )
        .await;

        assert_eq!(step, Step::Continue);
        assert_eq!(agent.history()[1], Message::Tool { tool_call_id: "c1".into(), content: "one".into() });
        assert_eq!(agent.history()[2], Message::Tool { tool_call_id: "c2".into(), content: "two".into() });
        assert_eq!(agent.history()[3], Message::Tool { tool_call_id: "c3".into(), content: format!("wrote 5 bytes to {}", file_c.to_string_lossy()) });
    }

    #[tokio::test]
    async fn invalid_arguments_in_a_read_batch_keep_their_position() {
        let mut agent = agent();
        let temp_dir = TestFiles::new();
        temp_dir.file("a.txt", "one\n");
        let file = temp_dir.path().join("a.txt");

        run_tool_calls(
            &mut agent,
            vec![
                tool_call("c1", "read_file", r#"{"path":"a.txt","start":"not-a-number"}"#),
                tool_call("c2", "read_file", &format!("{{\"path\":\"{}\"}}", file.to_string_lossy())),
            ],
            &mut Vec::new(),
        )
        .await;

        let Message::Tool { tool_call_id, content } = &agent.history()[1] else {
            panic!("expected tool message");
        };
        assert_eq!(tool_call_id, "c1");
        assert!(content.contains("invalid arguments"));
        assert_eq!(
            agent.history()[2],
            Message::Tool {
                tool_call_id: "c2".into(),
                content: "one".into(),
            }
        );
    }

    #[tokio::test]
    async fn unknown_tool_is_returned_to_model() {
        let mut agent = agent();

        run_tool_call(&mut agent, "call_1", "nuke", "{}", &mut Vec::new()).await;

        assert_eq!(
            agent.history()[1],
            Message::Tool {
                tool_call_id: "call_1".into(),
                content: "unknown tool nuke".into(),
            }
        );
    }

    #[tokio::test]
    async fn bash_tool_call_emits_command_as_started_body() {
        let mut agent = agent();
        let mut events = Vec::new();

        run_tool_call(
            &mut agent,
            "call_1",
            "bash",
            r#"{"command":"echo out"}"#,
            &mut events,
        )
        .await;

        assert_eq!(
            events,
            vec![
                AgentEvent::ToolStarted {
                    header: "bash".into(),
                    body: Some("echo out".into()),
                },
                AgentEvent::ToolResult {
                    header: "bash".into(),
                    body: "out\n".into(),
                },
            ]
        );
    }

    #[tokio::test]
    async fn read_file_tool_call_emits_read_content_as_result() {
        let files = TestFiles::new();
        files.file("a.txt", "line one\nline two");
        let path = files.path().join("a.txt").to_string_lossy().into_owned();
        let mut agent = agent();
        let mut events = Vec::new();

        run_tool_call(
            &mut agent,
            "call_1",
            "read_file",
            &format!("{{\"path\":\"{path}\"}}"),
            &mut events,
        )
        .await;

        assert_eq!(
            events,
            vec![
                AgentEvent::ToolStarted {
                    header: format!("read_file: {path}"),
                    body: None,
                },
                AgentEvent::ToolResult {
                    header: format!("read_file: {path}"),
                    body: "line one\nline two".into(),
                },
            ]
        );
    }

    #[tokio::test]
    async fn parallel_tool_calls_all_get_results() {
        let mut agent = agent();
        let step = agent
            .handle_response(
                Message::Assistant {
                    content: None,
                    tool_calls: vec![
                        tool_call("call_1", "bash", r#"{"command":"echo one"}"#),
                        tool_call("call_2", "bash", r#"{"command":"echo two"}"#),
                    ],
                },
                &mut |event| {
                    let _ = event;
                },
            )
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
        agent.context.messages.push(Message::User {
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
        assert_eq!(tools.len(), 7);
        assert_eq!(
            tools.iter().map(|t| t.function.name.as_str()).collect::<Vec<_>>(),
            vec!["bash", "readonly_bash", "read_file", "write_file", "edit_file", "webfetch", "bg_run"]
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
    fn usage_chunk_after_finish_is_captured() {
        let mut acc = StreamAccumulator::new();

        feed_json(&mut acc, r#"{"choices":[{"delta":{"content":"hi"}}]}"#);
        feed_json(&mut acc, r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#);
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
        let Message::Assistant { content, tool_calls } = acc.into_message() else {
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
