use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures_util::StreamExt;
use futures_util::future::join_all;
use openai_oxide::client::OpenAI;
use openai_oxide::types::chat::{
    ChatCompletionMessageParam, ChatCompletionRequest, StreamOptions, ToolCall, UserContent,
};

use crate::context::Context;
use crate::message::Message;
use crate::mode::Mode;
use crate::stream::{AgentEvent, ChunkTokens, StreamAccumulator, StreamChunk};
use crate::thinking::ThinkingLevel;

use crate::tool::{Tool, ToolError, ToolOutput, tool_definitions};

#[derive(Debug, PartialEq)]
pub enum ChatOutcome {
    Answer(String),
    Terminated { tool: crate::tool::Tool },
    Cancelled,
}

#[derive(Debug)]
pub struct Cancelled;

impl std::fmt::Display for Cancelled {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("cancelled")
    }
}

impl std::error::Error for Cancelled {}

#[derive(Debug, PartialEq)]
enum Step {
    Continue,
    Done(String),
    Cancelled,
}

#[derive(Clone)]
pub struct Steering {
    queue: std::sync::Arc<std::sync::Mutex<std::collections::VecDeque<String>>>,
}

impl Steering {
    pub fn new() -> Steering {
        Steering {
            queue: std::sync::Arc::new(std::sync::Mutex::new(std::collections::VecDeque::new())),
        }
    }

    pub fn push(&self, text: String) {
        self.queue.lock().unwrap().push_back(text);
    }

    pub fn drain(&self) -> Vec<String> {
        self.queue.lock().unwrap().drain(..).collect()
    }
}

impl Default for Steering {
    fn default() -> Steering {
        Steering::new()
    }
}

pub struct Agent {
    pub client: OpenAI,
    pub model: String,
    pub context: Context,
    pub mode: Arc<Mutex<Mode>>,
    pub thinking: Option<(Arc<Mutex<ThinkingLevel>>, Option<bool>)>,
    pub bash_timeout: Duration,
    cancel: Option<std::sync::Arc<tokio::sync::watch::Receiver<u64>>>,
    steering: Steering,
    bg_seen: std::collections::HashSet<String>,
    bg_mine: std::collections::HashSet<String>,
}

const MAX_TOOL_ROUNDS: usize = 500;

const SUMMARIZATION_PROMPT: &str = "The messages above are a conversation to summarize. Create a structured context checkpoint summary that another LLM will use to continue the work.\n\nUse this EXACT format:\n\n## Goal\n[What is the user trying to accomplish? Can be multiple items.]\n\n## Constraints & Preferences\n- [Any constraints, preferences, or requirements mentioned by user]\n- [Or \"(none)\" if none were mentioned]\n\n## Progress\n### Done\n- [x] [Completed tasks/changes]\n\n### In Progress\n- [ ] [Current work]\n\n### Blocked\n- [Issues preventing progress, if any]\n\n## Key Decisions\n- **[Decision]**: [Brief rationale]\n\n## Next Steps\n1. [Ordered list of what should happen next]\n\n## Critical Context\n- [Any data, examples, or references needed to continue]\n- [Or \"(none)\" if not applicable]\n\nKeep each section concise. Preserve exact file paths, function names, and error messages.";

impl Agent {
    pub fn new(
        client: OpenAI,
        model: impl Into<String>,
        mode: Arc<Mutex<Mode>>,
        max_tokens: u64,
        bash_timeout: Duration,
    ) -> Agent {
        let system_prompt = mode.lock().unwrap().system_prompt_with_cwd();
        Agent {
            client,
            model: model.into(),
            context: Context::new(system_prompt, max_tokens),
            mode,
            thinking: None,
            bash_timeout,
            cancel: None,
            steering: Steering::default(),
            bg_seen: std::collections::HashSet::new(),
            bg_mine: std::collections::HashSet::new(),
        }
    }

    pub fn with_thinking(mut self, cell: Arc<Mutex<ThinkingLevel>>, base: Option<bool>) -> Agent {
        self.thinking = Some((cell, base));
        self
    }

    pub fn with_cancel(
        mut self,
        rx: std::sync::Arc<tokio::sync::watch::Receiver<u64>>,
    ) -> Agent {
        self.cancel = Some(rx);
        self
    }

    pub fn with_steering(mut self, steering: Steering) -> Agent {
        self.steering = steering;
        self
    }

    pub fn with_pinned_mode(mut self, mode: Mode) -> Agent {
        self.mode = Arc::new(Mutex::new(mode));
        self.context.system_prompt = mode.system_prompt_with_cwd();
        self
    }

    pub fn stage_mode(&self) -> Option<Mode> {
        let guard = self.mode.lock().unwrap();
        if Arc::strong_count(&self.mode) == 1 {
            Some(*guard)
        } else {
            None
        }
    }

    pub async fn chat(
        &mut self,
        user_message: &str,
        on_event: &mut impl FnMut(AgentEvent),
    ) -> Result<ChatOutcome, Box<dyn std::error::Error>> {
        self.context.messages.push(Message::User {
            content: user_message.to_string(),
        });
        self.run_turns(on_event).await
    }

    pub async fn bg_turn(
        &mut self,
        on_event: &mut impl FnMut(AgentEvent),
    ) -> Result<ChatOutcome, Box<dyn std::error::Error>> {
        self.run_turns(on_event).await
    }

    pub fn owns_and_unseen(&self, id: &str) -> bool {
        self.bg_mine.contains(id) && !self.bg_seen.contains(id)
    }

    async fn run_turns(
        &mut self,
        on_event: &mut impl FnMut(AgentEvent),
    ) -> Result<ChatOutcome, Box<dyn std::error::Error>> {
        self.context.system_prompt = self.mode.lock().unwrap().system_prompt_with_cwd();
        self.context.seal_dangling_tool_calls();
        let base = self.cancel.as_ref().map(|rx| *rx.borrow());
        let mut round = 0;
        loop {
            round += 1;
            if round > MAX_TOOL_ROUNDS {
                return Err(format!("tool loop exceeded {MAX_TOOL_ROUNDS} rounds").into());
            }
            if self.is_cancelled(base) {
                return Ok(ChatOutcome::Cancelled);
            }
            for text in self.drain_steering() {
                self.context.messages.push(Message::User { content: text });
            }
            if self.context.needs_compaction() {
                self.compact_context().await;
            }
            self.report_bg_tasks(on_event);
            on_event(AgentEvent::CompletionStarted);

            let (message, usage) = match self
                .stream_completion(self.build_request(), base, &mut |tokens| {
                    on_event(AgentEvent::Tokens(tokens))
                })
                .await
            {
                Ok(value) => value,
                Err(error) if error.downcast_ref::<Cancelled>().is_some() => {
                    return Ok(ChatOutcome::Cancelled)
                }
                Err(error) => return Err(error),
            };
            self.context.record_usage(usage.0, usage.1);
            on_event(AgentEvent::Usage {
                prompt: self.context.prompt_tokens,
            });

            let terminator = self.mode.lock().unwrap().terminator();
            if let Some(terminator) = terminator
                && let Message::Assistant { content, tool_calls, .. } = &message
                && let Some(call) = tool_calls.iter().find(|c| c.function.name == terminator)
            {
                match Tool::try_from(call.clone()) {
                    Ok(tool) => {
                        self.context.messages.push(message.clone());
                        let body = match &tool {
                            Tool::SubmitPlan(stages) => Some(crate::tool::plan_text(stages)),
                            _ => None,
                        };
                        on_event(AgentEvent::ToolStarted {
                            header: terminator.to_string(),
                            body,
                        });
                        return Ok(ChatOutcome::Terminated { tool });
                    }
                    Err(error) => {
                        self.context.messages.push(Message::Assistant {
                            content: content.clone(),
                            tool_calls: vec![call.clone()],
                        });
                        on_event(AgentEvent::ToolStarted {
                            header: terminator.to_string(),
                            body: None,
                        });
                        let feedback =
                            format!("{error} Call {terminator} again with corrected arguments.");
                        on_event(AgentEvent::ToolResult {
                            header: terminator.to_string(),
                            body: feedback.clone(),
                        });
                        self.context.messages.push(Message::Tool {
                            tool_call_id: call.id.clone(),
                            content: feedback,
                        });
                    }
                }
            }

            match self.handle_response(message, on_event).await {
                Step::Done(text) => {
                    let queued = self.drain_steering();
                    if queued.is_empty() {
                        return Ok(ChatOutcome::Answer(text));
                    }
                    for item in queued {
                        self.context.messages.push(Message::User { content: item });
                    }
                }
                Step::Cancelled => return Ok(ChatOutcome::Cancelled),
                Step::Continue => {}
            }
        }
    }

    async fn compact_context(&mut self) {
        let keep_recent = self.context.keep_recent_tokens();
        let Some(cut_point) = self.context.compact_point(keep_recent) else {
            return;
        };
        let to_summarize = self.context.messages[..cut_point].to_vec();
        let (read, modified) = self.context.file_operations();
        let Ok(summary) = self.summarize(&to_summarize).await else {
            return;
        };
        if summary.trim().is_empty() {
            return;
        }
        let mut full = summary;
        if !read.is_empty() || !modified.is_empty() {
            full.push_str("\n\n## Files");
            if !read.is_empty() {
                full.push_str(&format!("\nRead: {}", read.join(", ")));
            }
            if !modified.is_empty() {
                full.push_str(&format!("\nModified: {}", modified.join(", ")));
            }
        }
        self.context.apply_compaction(full, cut_point);
    }

    async fn summarize(
        &self,
        messages: &[Message],
    ) -> Result<String, Box<dyn std::error::Error>> {
        let conversation = Self::serialize_messages(messages);
        let request = ChatCompletionRequest::new(
            self.model.clone(),
            vec![
                ChatCompletionMessageParam::System {
                    content: SUMMARIZATION_PROMPT.to_string(),
                    name: None,
                },
                ChatCompletionMessageParam::User {
                    content: UserContent::Text(format!(
                        "<conversation>\n{conversation}\n</conversation>"
                    )),
                    name: None,
                },
            ],
        );
        let response = self.client.chat().completions().create_raw(&request).await?;
        Ok(response["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string())
    }

    fn serialize_messages(messages: &[Message]) -> String {
        messages
            .iter()
            .map(|message| match message {
                Message::System { content } => format!("[system] {content}"),
                Message::User { content } => format!("[user] {content}"),
                Message::Assistant { content, tool_calls } => {
                    let calls = tool_calls
                        .iter()
                        .map(|call| {
                            format!(
                                "[tool_call {} {}]",
                                call.function.name, call.function.arguments
                            )
                        })
                        .collect::<Vec<_>>()
                        .join(" ");
                    let content = content.as_deref().unwrap_or("");
                    format!("[assistant] {content} {calls}")
                }
                Message::Tool { content, .. } => format!("[tool] {content}"),
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn is_cancelled(&self, base: Option<u64>) -> bool {
        match (base, self.cancel.as_ref()) {
            (Some(base), Some(rx)) => *rx.borrow() != base,
            _ => false,
        }
    }

    fn drain_steering(&self) -> Vec<String> {
        self.steering.drain()
    }

    async fn cancel_wait(&self, base: Option<u64>) {
        match (base, self.cancel.as_ref()) {
            (Some(base), Some(rx)) => {
                let mut rx = (**rx).clone();
                let _ = rx.wait_for(|value| *value != base).await;
            }
            _ => std::future::pending::<()>().await,
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

    async fn handle_response(
        &mut self,
        message: Message,
        on_event: &mut impl FnMut(AgentEvent),
    ) -> Step {
        let Message::Assistant {
            content,
            tool_calls,
        } = message
        else {
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

        let base = self.cancel.as_ref().map(|rx| *rx.borrow());

        let mut index = 0;
        while index < tool_calls.len() {
            let mut end = index;
            while end < tool_calls.len() && tool_calls[end].function.name == "read_file" {
                end += 1;
            }
            if end == index {
                end = index + 1;
            }
            if self.run_tools(&tool_calls[index..end], on_event, base).await {
                return Step::Cancelled;
            }
            index = end;
        }

        Step::Continue
    }

    async fn run_tools(
        &mut self,
        calls: &[ToolCall],
        on_event: &mut impl FnMut(AgentEvent),
        base: Option<u64>,
    ) -> bool {
        let mut slots: Vec<
            Result<(Tool, String, Option<tokio::sync::watch::Receiver<Option<Vec<String>>>>), String>,
        > = Vec::new();
        for call in calls {
            match Tool::try_from(call.clone()) {
                Ok(tool) => {
                    if !self.mode.lock().unwrap().allows(&tool) {
                        slots.push(Err(format!(
                            "{} is not allowed in mode {}",
                            tool.label(),
                            self.mode.lock().unwrap().label()
                        )));
                        continue;
                    }
                    let header = tool.header();
                    let output = tool.output();
                    let body = match &output {
                        ToolOutput::Before(body) => Some(body.clone()),
                        ToolOutput::After => None,
                    };
                    on_event(AgentEvent::ToolStarted {
                        header: header.clone(),
                        body,
                    });
                    let reply_rx = if let Tool::AskUser(questions) = &tool {
                        let (reply_tx, reply_rx) = tokio::sync::watch::channel(None);
                        on_event(AgentEvent::AskUser {
                            questions: questions.clone(),
                            reply: reply_tx,
                        });
                        Some(reply_rx)
                    } else {
                        None
                    };
                    slots.push(Ok((tool, header, reply_rx)));
                }
                Err(error) => slots.push(Err(error.to_string())),
            }
        }

        let futures: Vec<_> = slots
            .iter()
            .filter_map(|slot| {
                slot.as_ref()
                    .ok()
                    .map(|(tool, _, rx)| self.invoke_tool(tool, rx.as_ref()))
            })
            .collect();
        let join = join_all(futures);
        let (outputs, cancelled) = match base {
            Some(base) => {
                let mut join = Some(join);
                let result = tokio::select! {
                    result = async { join.as_mut().unwrap().await } => Some(result),
                    _ = self.cancel_wait(Some(base)) => None,
                };
                match result {
                    Some(outputs) => (outputs, false),
                    None => (Vec::new(), true),
                }
            }
            None => (join.await, false),
        };
        if cancelled {
            for (call, slot) in calls.iter().zip(slots) {
                let header = match &slot {
                    Ok((_, header, _)) => header.clone(),
                    Err(_) => String::new(),
                };
                on_event(AgentEvent::ToolResult {
                    header,
                    body: "cancelled".into(),
                });
                self.context.messages.push(Message::Tool {
                    tool_call_id: call.id.clone(),
                    content: "cancelled".into(),
                });
            }
            return true;
        }

        let mut outputs = outputs.into_iter();
        for (call, slot) in calls.iter().zip(slots) {
            let content = match slot {
                Ok((tool, header, _)) => {
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
        false
    }

    fn invoke_tool<'a>(
        &'a self,
        tool: &'a Tool,
        reply_rx: Option<&'a tokio::sync::watch::Receiver<Option<Vec<String>>>>,
    ) -> futures_util::future::BoxFuture<'a, Result<String, ToolError>> {
        match (tool, reply_rx) {
            (Tool::AskUser(questions), Some(rx)) => {
                let questions = questions.clone();
                let mut rx = rx.clone();
                Box::pin(async move {
                    match rx.changed().await {
                        Ok(_) => {
                            let answers = rx.borrow().clone().unwrap_or_default();
                            Ok(format_question_result(&questions, answers))
                        }
                        Err(_) => Err(ToolError::InteractiveNotExecutable),
                    }
                })
            }
            (Tool::AskUser(_), None) => {
                Box::pin(async { Err(ToolError::InteractiveNotExecutable) })
            }
            (_, _) => Box::pin(tool.invoke(self.bash_timeout)),
        }
    }

    fn build_request(&self) -> ChatCompletionRequest {
        let mut request =
            ChatCompletionRequest::new(self.model.clone(), self.context.build_messages());
        request.tools = Some(tool_definitions(&self.mode.lock().unwrap().base_tools()));
        request
    }

    async fn stream_completion(
        &self,
        request: ChatCompletionRequest,
        base: Option<u64>,
        on_token: &mut impl FnMut(ChunkTokens),
    ) -> Result<
        (Message, (Option<u64>, Option<u64>)),
        Box<dyn std::error::Error>,
    > {
        let mut request = request;
        request.stream = Some(true);
        request.stream_options = Some(StreamOptions {
            include_usage: Some(true),
        });

        let extra = self
            .thinking
            .as_ref()
            .and_then(|(cell, base)| cell.lock().unwrap().body(*base));
        let mut stream = if let Some(extra) = extra {
            let mut body = serde_json::to_value(&request)?;
            if let (Some(map), Some(extra_map)) = (body.as_object_mut(), extra.as_object()) {
                for (key, value) in extra_map {
                    map.insert(key.clone(), value.clone());
                }
            }
            self.client
                .chat()
                .completions()
                .create_stream_raw(&body)
                .await?
        } else {
            self.client
                .chat()
                .completions()
                .create_stream_raw(&request)
                .await?
        };
        let mut accumulator = StreamAccumulator::new();

        loop {
            let next = tokio::select! {
                next = stream.next() => next,
                _ = self.cancel_wait(base) => return Err(Cancelled.into()),
            };
            match next {
                Some(Ok(value)) => {
                    let chunk: StreamChunk = serde_json::from_value(value)?;
                    for tokens in accumulator.feed(chunk) {
                        on_token(tokens);
                    }
                }
                Some(Err(error)) => return Err(error.into()),
                None => break,
            }
        }

        let usage = accumulator.usage();
        Ok((accumulator.into_message(), usage))
    }
}

fn format_question_result(questions: &[crate::tool::Question], answers: Vec<String>) -> String {
    let mut out = String::new();
    for (i, (q, answer)) in questions.iter().zip(answers.iter()).enumerate() {
        out.push_str(&format!("{}. {}\n   → {answer}\n", i + 1, q.prompt));
    }
    out.trim_end().to_string()
}

#[cfg(test)]
mod test {
    use super::*;
    use openai_oxide::types::chat::{FunctionCall, ToolCall};
    use test_files::TestFiles;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    fn complete_request(data: &[u8]) -> Option<String> {
        let header_end = data.windows(4).position(|w| w == b"\r\n\r\n")?;
        let headers = std::str::from_utf8(&data[..header_end]).unwrap();
        let length = headers.lines().find_map(|line| {
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

    async fn mock_server(
        responses: Vec<String>,
    ) -> (String, tokio::sync::mpsc::UnboundedReceiver<String>) {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<String>();
        let responses = Arc::new(Mutex::new(responses.into_iter()));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            while let Ok((mut socket, _)) = listener.accept().await {
                let tx = tx.clone();
                let responses = responses.clone();
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
                    let sse = responses
                        .lock()
                        .unwrap()
                        .next()
                        .unwrap_or_else(|| "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\ndata: [DONE]\n\n".to_string());
                    let response =
                        format!("HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n\r\n{sse}");
                    let _ = socket.write_all(response.as_bytes()).await;
                });
            }
        });
        (format!("http://{addr}"), rx)
    }

    fn plan_agent(base_url: String) -> Agent {
        let client =
            OpenAI::with_config(openai_oxide::ClientConfig::new("local").base_url(base_url));
        Agent::new(
            client,
            "test-model",
            Arc::new(Mutex::new(crate::mode::Mode::Plan)),
            10000,
            Duration::from_secs(30),
        )
    }

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
        Agent::new(
            OpenAI::new("test-key"),
            "test-model",
            Arc::new(Mutex::new(Mode::Yolo)),
            10000,
            Duration::from_secs(30),
        )
    }

    async fn json_mock_server(response: &str) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let response = response.to_string();
        tokio::spawn(async move {
            while let Ok((mut socket, _)) = listener.accept().await {
                let response = response.clone();
                tokio::spawn(async move {
                    let mut data = Vec::new();
                    let mut buf = [0u8; 8192];
                    loop {
                        match socket.read(&mut buf).await {
                            Ok(0) | Err(_) => break,
                            Ok(n) => {
                                data.extend_from_slice(&buf[..n]);
                                if complete_request(&data).is_some() {
                                    break;
                                }
                            }
                        }
                    }
                    let response = format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n\r\n{response}"
                    );
                    let _ = socket.write_all(response.as_bytes()).await;
                });
            }
        });
        format!("http://{addr}")
    }

    /// Streams SSE for completions and returns JSON for the non-streaming
    /// summarization call, so a single agent run can do both.
    async fn hybrid_mock_server(
        sse: Vec<String>,
        json: String,
    ) -> (String, tokio::sync::mpsc::UnboundedReceiver<String>) {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<String>();
        let sse = Arc::new(Mutex::new(sse.into_iter()));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            while let Ok((mut socket, _)) = listener.accept().await {
                let tx = tx.clone();
                let sse = sse.clone();
                let json = json.clone();
                tokio::spawn(async move {
                    let mut data = Vec::new();
                    let mut buf = [0u8; 8192];
                    let mut body = String::new();
                    loop {
                        match socket.read(&mut buf).await {
                            Ok(0) | Err(_) => break,
                            Ok(n) => {
                                data.extend_from_slice(&buf[..n]);
                                if let Some(complete) = complete_request(&data) {
                                    body = complete.clone();
                                    let _ = tx.send(complete);
                                    break;
                                }
                            }
                        }
                    }
                    let response = if body.contains("\"stream\":true") {
                        let sse = sse.lock().unwrap().next().unwrap_or_else(|| {
                            "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\ndata: [DONE]\n\n"
                                .to_string()
                        });
                        format!("HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n\r\n{sse}")
                    } else {
                        format!("HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n\r\n{json}")
                    };
                    let _ = socket.write_all(response.as_bytes()).await;
                });
            }
        });
        (format!("http://{addr}"), rx)
    }

    #[tokio::test]
    async fn compaction_runs_while_a_turn_is_still_in_progress() {
        let (base_url, mut requests) = hybrid_mock_server(
            vec!["data: {\"choices\":[{\"delta\":{\"content\":\"done\"}}]}\n\ndata: [DONE]\n\n".to_string()],
            "{\"choices\":[{\"message\":{\"content\":\"Test goal summary\"}}]}".to_string(),
        )
        .await;
        let client =
            OpenAI::with_config(openai_oxide::ClientConfig::new("local").base_url(base_url));
        let mut agent = Agent::new(
            client,
            "test-model",
            Arc::new(Mutex::new(Mode::Yolo)),
            1000,
            Duration::from_secs(30),
        );
        agent.context.messages.push(Message::User {
            content: "old work ".repeat(1000).into(),
        });
        agent.context.messages.push(Message::Assistant {
            content: None,
            tool_calls: vec![tool_call("c1", "read_file", r#"{"path":"a.rs"}"#)],
        });
        agent.context.messages.push(Message::Tool {
            tool_call_id: "c1".into(),
            content: "old result ".repeat(1000).into(),
        });
        agent.context.messages.push(Message::Assistant {
            content: None,
            tool_calls: vec![tool_call("c2", "read_file", r#"{"path":"b.rs"}"#)],
        });
        agent.context.messages.push(Message::Tool {
            tool_call_id: "c2".into(),
            content: "recent result".into(),
        });
        agent.context.prompt_tokens = Some(5000);
        assert!(
            agent.context.is_in_progress(),
            "the turn must be mid-tool-loop for this test to mean anything"
        );

        agent.bg_turn(&mut |_| {}).await.unwrap();

        assert!(
            matches!(&agent.context.messages[0], Message::User { content } if content.contains("Test goal summary")),
            "compaction should have run mid-turn, messages: {:?}",
            agent.context.messages.len()
        );

        let summarize = requests.try_recv().expect("a summarization request");
        assert!(!summarize.contains("\"stream\":true"));
        let completion = requests.try_recv().expect("a completion request");
        assert!(
            completion.contains("Test goal summary"),
            "the compacted context should be what got sent"
        );
        assert!(
            !completion.contains("old work"),
            "the summarized prefix should be gone from the request"
        );
    }

    #[tokio::test]
    async fn compact_context_summarizes_old_messages_and_keeps_the_recent_window() {
        let base_url = json_mock_server(
            "{\"choices\":[{\"message\":{\"content\":\"Test goal summary: done item\"}}]}",
        )
        .await;
        let client =
            OpenAI::with_config(openai_oxide::ClientConfig::new("local").base_url(base_url));
        let mut agent = Agent::new(
            client,
            "test-model",
            Arc::new(Mutex::new(Mode::Yolo)),
            1000,
            Duration::from_secs(30),
        );
        agent
            .context
            .messages
            .push(Message::User { content: "old work ".repeat(1000).into() });
        agent
            .context
            .messages
            .push(Message::Assistant {
                content: None,
                tool_calls: vec![tool_call("c1", "read_file", r#"{"path":"a.rs"}"#)],
            });
        agent
            .context
            .messages
            .push(Message::Tool {
                tool_call_id: "c1".into(),
                content: "old result ".repeat(1000).into(),
            });
        agent
            .context
            .messages
            .push(Message::User { content: "recent".into() });
        agent.context.prompt_tokens = Some(5000);
        assert!(agent.context.needs_compaction());
        assert!(!agent.context.is_in_progress());

        agent.compact_context().await;

        assert!(
            agent.context.messages.len() < 4,
            "context should be compacted, was {}",
            agent.context.messages.len()
        );
        assert!(
            matches!(&agent.context.messages[0], Message::User { content } if content.contains("Test goal")),
            "first message should be the summary"
        );
        assert!(
            agent
                .context
                .messages
                .iter()
                .any(|m| matches!(m, Message::User { content } if content == "recent")),
            "recent message should be kept"
        );
        assert!(
            !agent
                .context
                .messages
                .iter()
                .any(|m| matches!(m, Message::Tool { .. })),
            "old tool result should be summarized away"
        );
    }

    #[test]
    fn pinned_mode_is_unaffected_by_the_shared_cell() {
        let shared = Arc::new(Mutex::new(Mode::Yolo));
        let mut agent = Agent::new(
            OpenAI::new("test-key"),
            "test-model",
            shared.clone(),
            10000,
            Duration::from_secs(30),
        );
        agent = agent.with_pinned_mode(Mode::Implement);
        *shared.lock().unwrap() = Mode::Plan;
        assert_eq!(*agent.mode.lock().unwrap(), Mode::Implement);
        assert!(agent.context.system_prompt.contains(Mode::Implement.system_prompt()));
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
            let length = headers.lines().find_map(|line| {
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
                    let response =
                        format!("HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n\r\n{sse}");
                    let _ = socket.write_all(response.as_bytes()).await;
                });
            }
        });

        let client = OpenAI::with_config(
            openai_oxide::ClientConfig::new("local").base_url(format!("http://{addr}")),
        );
        let mut agent = Agent::new(
            client,
            "test-model",
            Arc::new(Mutex::new(Mode::Yolo)),
            10000,
            Duration::from_secs(30),
        );

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

        let answer = agent
            .bg_turn(&mut |event| events.push(event))
            .await
            .unwrap();
        assert_eq!(answer, ChatOutcome::Answer("tests reported".into()));

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
            agent.context.messages.as_slice()[0],
            Message::Assistant {
                content: Some("hello".into()),
                tool_calls: vec![],
            }
        );
    }

    #[tokio::test]
    async fn tool_call_executes_and_continues() {
        let mut agent = agent();

        let step = run_tool_call(
            &mut agent,
            "call_1",
            "bash",
            r#"{"command":"echo out"}"#,
            &mut Vec::new(),
        )
        .await;

        assert_eq!(step, Step::Continue);
        assert_eq!(agent.context.messages.as_slice().len(), 2);
        assert_eq!(
            agent.context.messages.as_slice()[1],
            Message::Tool {
                tool_call_id: "call_1".into(),
                content: "out\n".into(),
            }
        );
    }

    #[tokio::test]
    async fn tool_error_is_returned_to_model() {
        let mut agent = agent();

        run_tool_call(
            &mut agent,
            "call_1",
            "bash",
            r#"{"command":"exit 3"}"#,
            &mut Vec::new(),
        )
        .await;

        assert_eq!(
            agent.context.messages.as_slice()[1],
            Message::Tool {
                tool_call_id: "call_1".into(),
                content: "bash exited with 3\n".into(),
            }
        );
    }

    #[tokio::test]
    async fn invalid_arguments_are_returned_to_model() {
        let mut agent = agent();

        run_tool_call(
            &mut agent,
            "call_1",
            "bash",
            r#"{"command":"ls""#,
            &mut Vec::new(),
        )
        .await;

        let Message::Tool { content, .. } = &agent.context.messages.as_slice()[1] else {
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
                tool_call(
                    "c1",
                    "read_file",
                    &format!("{{\"path\":\"{}\"}}", file_a.to_string_lossy()),
                ),
                tool_call(
                    "c2",
                    "read_file",
                    &format!("{{\"path\":\"{}\"}}", file_b.to_string_lossy()),
                ),
                tool_call(
                    "c3",
                    "write_file",
                    &format!(
                        "{{\"path\":\"{}\",\"content\":\"three\"}}",
                        file_c.to_string_lossy()
                    ),
                ),
            ],
            &mut Vec::new(),
        )
        .await;

        assert_eq!(step, Step::Continue);
        assert_eq!(
            agent.context.messages.as_slice()[1],
            Message::Tool {
                tool_call_id: "c1".into(),
                content: "one".into()
            }
        );
        assert_eq!(
            agent.context.messages.as_slice()[2],
            Message::Tool {
                tool_call_id: "c2".into(),
                content: "two".into()
            }
        );
        assert_eq!(
            agent.context.messages.as_slice()[3],
            Message::Tool {
                tool_call_id: "c3".into(),
                content: format!("wrote 5 bytes to {}", file_c.to_string_lossy())
            }
        );
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
                tool_call(
                    "c1",
                    "read_file",
                    r#"{"path":"a.txt","start":"not-a-number"}"#,
                ),
                tool_call(
                    "c2",
                    "read_file",
                    &format!("{{\"path\":\"{}\"}}", file.to_string_lossy()),
                ),
            ],
            &mut Vec::new(),
        )
        .await;

        let Message::Tool {
            tool_call_id,
            content,
        } = &agent.context.messages.as_slice()[1]
        else {
            panic!("expected tool message");
        };
        assert_eq!(tool_call_id, "c1");
        assert!(content.contains("invalid arguments"));
        assert_eq!(
            agent.context.messages.as_slice()[2],
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
            agent.context.messages.as_slice()[1],
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
    async fn ask_user_waits_for_the_answer_and_formats_it() {
        let agent = agent();
        let events = Arc::new(std::sync::Mutex::new(Vec::new()));
        let events_for_task = events.clone();
        let handle = tokio::spawn(async move {
            let mut agent = agent;
            let call = tool_call(
                "call_1",
                "ask_user",
                r#"{"questions":[{"prompt":"which db?","type":"single_choice","options":["pg","mysql"]},{"prompt":"name?","type":"free"}]}"#,
            );
            agent
                .handle_response(
                    Message::Assistant {
                        content: None,
                        tool_calls: vec![call],
                    },
                    &mut |event| events_for_task.lock().unwrap().push(event),
                )
                .await
        });
        let reply = loop {
            let found = events
                .lock()
                .unwrap()
                .iter()
                .find_map(|event| match event {
                    AgentEvent::AskUser { reply, .. } => Some(reply.clone()),
                    _ => None,
                });
            if let Some(reply) = found {
                break reply;
            }
            tokio::task::yield_now().await;
        };
        reply
            .send(Some(vec!["pg".to_string(), "postgres".to_string()]))
            .unwrap();
        handle.await.unwrap();
        let events = events.lock().unwrap();
        let result = events
            .iter()
            .find_map(|event| match event {
                AgentEvent::ToolResult { body, .. } => Some(body.clone()),
                _ => None,
            })
            .unwrap();
        assert_eq!(
            result,
            "1. which db?\n   → pg\n2. name?\n   → postgres"
        );
    }

    #[tokio::test]
    async fn ask_user_with_a_dropped_reply_is_an_error() {
        let agent = agent();
        let events = Arc::new(std::sync::Mutex::new(Vec::new()));
        let events_for_task = events.clone();
        let handle = tokio::spawn(async move {
            let mut agent = agent;
            let call = tool_call(
                "call_1",
                "ask_user",
                r#"{"questions":[{"prompt":"q?","type":"free"}]}"#,
            );
            agent
                .handle_response(
                    Message::Assistant {
                        content: None,
                        tool_calls: vec![call],
                    },
                    &mut |event| events_for_task.lock().unwrap().push(event),
                )
                .await
        });
        let reply = loop {
            let mut evs = events.lock().unwrap();
            if let Some(pos) = evs.iter().position(|event| matches!(event, AgentEvent::AskUser { .. }))
            {
                let removed = evs.remove(pos);
                if let AgentEvent::AskUser { reply, .. } = removed {
                    break reply;
                }
            }
            drop(evs);
            tokio::task::yield_now().await;
        };
        drop(reply);
        handle.await.unwrap();
        let events = events.lock().unwrap();
        let result = events
            .iter()
            .find_map(|event| match event {
                AgentEvent::ToolResult { body, .. } => Some(body.clone()),
                _ => None,
            })
            .unwrap();
        assert!(result.contains("ask_user"));
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
            agent.context.messages.as_slice()[1],
            Message::Tool {
                tool_call_id: "call_1".into(),
                content: "one\n".into(),
            }
        );
        assert_eq!(
            agent.context.messages.as_slice()[2],
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
        assert!(content.contains(Mode::Yolo.system_prompt()));

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
        assert_eq!(tools.len(), 8);
        assert_eq!(
            tools
                .iter()
                .map(|t| t.function.name.as_str())
                .collect::<Vec<_>>(),
            vec![
                "bash",
                "readonly_bash",
                "read_file",
                "write_file",
                "edit_file",
                "webfetch",
                "bg_run",
                "ask_user"
            ]
        );
    }

    #[tokio::test]
    async fn thinking_level_changes_at_runtime_reach_the_next_request() {
        fn complete_request(data: &[u8]) -> Option<String> {
            let header_end = data.windows(4).position(|w| w == b"\r\n\r\n")?;
            let headers = std::str::from_utf8(&data[..header_end]).unwrap();
            let length = headers.lines().find_map(|line| {
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
                    let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\ndata: [DONE]\n\n";
                    let response =
                        format!("HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n\r\n{sse}");
                    let _ = socket.write_all(response.as_bytes()).await;
                });
            }
        });

        let client = OpenAI::with_config(
            openai_oxide::ClientConfig::new("local").base_url(format!("http://{addr}")),
        );
        let cell = Arc::new(Mutex::new(ThinkingLevel::Off));
        let mut agent = Agent::new(
            client,
            "test-model",
            Arc::new(Mutex::new(Mode::Yolo)),
            10000,
            Duration::from_secs(30),
        )
        .with_thinking(cell.clone(), None);

        let mut turn = async |agent: &mut Agent| {
            let answer = agent.chat("hi", &mut |_event| {}).await.unwrap();
            assert_eq!(answer, ChatOutcome::Answer("ok".into()));
            rx.recv().await.unwrap()
        };

        let body = turn(&mut agent).await;
        assert!(
            !body.contains("reasoning_effort"),
            "auto + off must send no override: {body}"
        );
        assert!(
            !body.contains("chat_template_kwargs"),
            "auto + off must send no override: {body}"
        );

        *cell.lock().unwrap() = ThinkingLevel::Low;
        let body = turn(&mut agent).await;
        assert!(body.contains("\"reasoning_effort\":\"low\""), "{body}");
        assert!(body.contains("\"enable_thinking\":true"), "{body}");

        *cell.lock().unwrap() = ThinkingLevel::XHigh;
        let body = turn(&mut agent).await;
        assert!(body.contains("\"reasoning_effort\":\"xhigh\""), "{body}");
    }

    #[tokio::test]
    async fn the_terminator_ends_the_turn_and_captures_the_payload() {
        let sse = "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"submit_plan\",\"arguments\":\"{\\\"stages\\\":[{\\\"title\\\":\\\"step one\\\",\\\"description\\\":\\\"do it\\\"}]}\"}}]}}]}\n\ndata: [DONE]\n\n";
        let (base_url, _requests) = mock_server(vec![sse.to_string()]).await;
        let mut agent = plan_agent(base_url);
        let mut events = Vec::new();

        let outcome = agent
            .chat("plan this", &mut |event| events.push(event))
            .await
            .unwrap();
        assert_eq!(
            outcome,
            ChatOutcome::Terminated {
                tool: Tool::SubmitPlan(vec![crate::tool::PlanStage {
                    title: "step one".into(),
                    description: "do it".into(),
                }]),
            }
        );
        assert!(events.iter().any(|event| {
            matches!(event, AgentEvent::ToolStarted { header, .. } if header == "submit_plan")
        }));
    }

    #[tokio::test]
    async fn a_terminated_turn_leaves_no_dangling_tool_call_in_the_next_request() {
        let sse = "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"submit_plan\",\"arguments\":\"{\\\"stages\\\":[{\\\"title\\\":\\\"step one\\\",\\\"description\\\":\\\"do it\\\"}]}\"}}]}}]}\n\ndata: [DONE]\n\n";
        let (base_url, mut requests) = mock_server(vec![sse.to_string()]).await;
        let mut agent = plan_agent(base_url);

        let outcome = agent.chat("plan this", &mut |_| {}).await.unwrap();
        assert!(matches!(outcome, ChatOutcome::Terminated { .. }));
        // The terminator path stores the assistant call and returns without a
        // result; history is sent verbatim now, so the next turn must repair it.
        agent.chat("carry on", &mut |_| {}).await.unwrap();

        let _first = requests.try_recv().expect("the first request");
        let second = requests.try_recv().expect("the second request");
        assert!(
            second.contains("call_1"),
            "the terminator call is still in history: {second}"
        );
        assert!(
            second.contains("\"role\":\"tool\""),
            "an assistant tool_calls message with no matching result would be \
             rejected by the API: {second}"
        );
    }

    #[tokio::test]
    async fn sibling_calls_are_dropped_when_the_terminator_is_called() {
        let sse = "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"read_file\",\"arguments\":\"{\\\"path\\\":\\\"a.txt\\\"}\"}}]}}]}\n\n"
            .to_string()
            + "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":1,\"id\":\"call_2\",\"type\":\"function\",\"function\":{\"name\":\"submit_plan\",\"arguments\":\"{\\\"stages\\\":[{\\\"title\\\":\\\"step one\\\",\\\"description\\\":\\\"do it\\\"}]}\"}}]}}]}\n\n"
            + "data: [DONE]\n\n";
        let (base_url, _requests) = mock_server(vec![sse]).await;
        let mut agent = plan_agent(base_url);
        let mut events = Vec::new();

        let outcome = agent
            .chat("plan this", &mut |event| events.push(event))
            .await
            .unwrap();
        assert!(matches!(outcome, ChatOutcome::Terminated { .. }));
        assert!(events.iter().any(|event| {
            matches!(event, AgentEvent::ToolStarted { header, .. } if header == "submit_plan")
        }));
        assert!(!events.iter().any(|event| {
            matches!(event, AgentEvent::ToolStarted { header, .. } if header.starts_with("read_file"))
        }));
        assert!(
            !events
                .iter()
                .any(|event| { matches!(event, AgentEvent::ToolResult { .. }) })
        );
    }

    #[tokio::test]
    async fn cancel_during_streaming_stops_the_turn() {
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel::<String>();
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
                                    let partial = "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"par\"}}]}\n\n";
                                    let _ = socket.write_all(partial.as_bytes()).await;
                                    tokio::time::sleep(Duration::from_secs(30)).await;
                                    break;
                                }
                            }
                        }
                    }
                });
            }
        });
        let client = OpenAI::with_config(
            openai_oxide::ClientConfig::new("local").base_url(format!("http://{addr}")),
        );
        let (cancel_tx, cancel_rx) = tokio::sync::watch::channel(0u64);
        let mut agent = Agent::new(
            client,
            "test-model",
            Arc::new(Mutex::new(Mode::Yolo)),
            10000,
            Duration::from_secs(30),
        )
        .with_cancel(std::sync::Arc::new(cancel_rx));
        let cancel_tx2 = cancel_tx.clone();
        let waiter = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(300)).await;
            cancel_tx2.send(1).unwrap();
        });

        let outcome = agent.chat("hi", &mut |_| {}).await.unwrap();

        assert_eq!(outcome, ChatOutcome::Cancelled);
        waiter.await.unwrap();
    }

    #[tokio::test]
    async fn cancel_during_tool_execution_cancels_the_turn() {
        let sse = "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"bash\",\"arguments\":\"{\\\"command\\\":\\\"sleep 30\\\"}\"}}]}}]}\n\ndata: [DONE]\n\n";
        let (base_url, _requests) = mock_server(vec![sse.to_string()]).await;
        let client =
            OpenAI::with_config(openai_oxide::ClientConfig::new("local").base_url(base_url));
        let (cancel_tx, cancel_rx) = tokio::sync::watch::channel(0u64);
        let mut agent = Agent::new(
            client,
            "test-model",
            Arc::new(Mutex::new(Mode::Yolo)),
            10000,
            Duration::from_secs(30),
        )
        .with_cancel(std::sync::Arc::new(cancel_rx));
        let events = Arc::new(Mutex::new(Vec::<AgentEvent>::new()));
        let events2 = events.clone();
        let cancel_tx2 = cancel_tx.clone();
        let waiter = tokio::spawn(async move {
            loop {
                if events2
                    .lock()
                    .unwrap()
                    .iter()
                    .any(|event| matches!(event, AgentEvent::ToolStarted { .. }))
                {
                    cancel_tx2.send(1).unwrap();
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        });

        let outcome = agent
            .chat("hi", &mut |event| events.lock().unwrap().push(event))
            .await
            .unwrap();

        assert_eq!(outcome, ChatOutcome::Cancelled);
        waiter.await.unwrap();
        assert_eq!(
            agent.context.messages.as_slice().last().unwrap(),
            &Message::Tool {
                tool_call_id: "call_1".into(),
                content: "cancelled".into(),
            }
        );
    }

    #[tokio::test]
    async fn steering_queued_during_a_tool_loop_is_picked_up_on_the_next_round() {
        let tool_call = "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"bash\",\"arguments\":\"{\\\"command\\\":\\\"echo hi\\\"}\"}}]}}]}\n\ndata: [DONE]\n\n";
        let answer = "data: {\"choices\":[{\"delta\":{\"content\":\"done\"}}]}\n\ndata: [DONE]\n\n";
        let (base_url, _requests) =
            mock_server(vec![tool_call.to_string(), answer.to_string()]).await;
        let client =
            OpenAI::with_config(openai_oxide::ClientConfig::new("local").base_url(base_url));
        let steering = Steering::new();
        let mut agent = Agent::new(
            client,
            "test-model",
            Arc::new(Mutex::new(Mode::Yolo)),
            10000,
            Duration::from_secs(30),
        )
        .with_steering(steering.clone());

        let outcome = agent
            .chat("go", &mut |event| {
                if matches!(event, AgentEvent::ToolResult { .. }) {
                    steering.push("steer me".into());
                }
            })
            .await
            .unwrap();

        assert_eq!(outcome, ChatOutcome::Answer("done".into()));
        let history = agent.context.messages.as_slice();
        let steer = history
            .iter()
            .position(|m| matches!(m, Message::User { content } if content == "steer me"))
            .unwrap();
        assert!(matches!(history[steer - 1], Message::Tool { .. }));
        assert!(matches!(history[steer + 1], Message::Assistant { .. }));
    }

    #[tokio::test]
    async fn steering_at_turn_end_triggers_an_extra_round() {
        let first = "data: {\"choices\":[{\"delta\":{\"content\":\"first\"}}]}\n\ndata: [DONE]\n\n";
        let second = "data: {\"choices\":[{\"delta\":{\"content\":\"second\"}}]}\n\ndata: [DONE]\n\n";
        let (base_url, mut requests) =
            mock_server(vec![first.to_string(), second.to_string()]).await;
        let client =
            OpenAI::with_config(openai_oxide::ClientConfig::new("local").base_url(base_url));
        let steering = Steering::new();
        let pushed = std::sync::atomic::AtomicBool::new(false);
        let mut agent = Agent::new(
            client,
            "test-model",
            Arc::new(Mutex::new(Mode::Yolo)),
            10000,
            Duration::from_secs(30),
        )
        .with_steering(steering.clone());

        let outcome = agent
            .chat("go", &mut |event| {
                if matches!(event, AgentEvent::CompletionStarted)
                    && !pushed.swap(true, std::sync::atomic::Ordering::Relaxed)
                {
                    steering.push("steer me".into());
                }
            })
            .await
            .unwrap();

        assert_eq!(outcome, ChatOutcome::Answer("second".into()));
        let mut count = 0;
        while requests.try_recv().is_ok() {
            count += 1;
        }
        assert_eq!(count, 2);
        let history = agent.context.messages.as_slice();
        let position_of = |predicate: fn(&Message) -> bool| {
            history.iter().position(predicate).unwrap()
        };
        let first = position_of(|m| matches!(m, Message::Assistant { content, .. } if content.as_deref() == Some("first")));
        let steer = position_of(|m| matches!(m, Message::User { content } if content == "steer me"));
        let second = position_of(|m| matches!(m, Message::Assistant { content, .. } if content.as_deref() == Some("second")));
        assert!(first < steer);
        assert!(steer < second);
    }

    #[tokio::test]
    async fn file_ref_content_survives_the_turn_so_the_prefix_stays_cacheable() {
        let answer = "data: {\"choices\":[{\"delta\":{\"content\":\"done\"}}]}\n\ndata: [DONE]\n\n";
        let (base_url, _requests) = mock_server(vec![answer.to_string()]).await;
        let client =
            OpenAI::with_config(openai_oxide::ClientConfig::new("local").base_url(base_url));
        let mut agent = Agent::new(
            client,
            "test-model",
            Arc::new(Mutex::new(Mode::Yolo)),
            10000,
            Duration::from_secs(30),
        );
        agent.context.messages.push(Message::User {
            content: "read @a.rs\n<file path=\"a.rs\">\nfn main() {}\n</file>".into(),
        });

        let outcome = agent.chat("go", &mut |_| {}).await.unwrap();
        assert!(matches!(outcome, ChatOutcome::Answer(_)));
        let user = agent
            .context
            .messages
            .iter()
            .find(|m| matches!(m, Message::User { .. }))
            .unwrap();
        if let Message::User { content } = user {
            assert!(
                content.contains("fn main() {}"),
                "the turn must not rewrite an earlier user message: {content}"
            );
        } else {
            panic!("expected a user message");
        }
    }

    #[tokio::test]
    async fn a_plain_answer_in_plan_mode_is_the_plan_payload() {
        let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"step one, step two\"}}]}\n\ndata: [DONE]\n\n";
        let (base_url, _requests) = mock_server(vec![sse.to_string()]).await;
        let mut agent = plan_agent(base_url);
        let mut events = Vec::new();

        let outcome = agent
            .chat("plan this", &mut |event| events.push(event))
            .await
            .unwrap();
        assert_eq!(outcome, ChatOutcome::Answer("step one, step two".into()));
    }

    #[tokio::test]
    async fn yolo_rejects_submit_plan_at_execution_time() {
        let tool_call_sse = "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"submit_plan\",\"arguments\":\"{\\\"stages\\\":[{\\\"title\\\":\\\"step one\\\",\\\"tasks\\\":[\\\"do it\\\"]}]}\"}}]}}]}\n\ndata: [DONE]\n\n";
        let answer_sse =
            "data: {\"choices\":[{\"delta\":{\"content\":\"done\"}}]}\n\ndata: [DONE]\n\n";
        let (base_url, _requests) =
            mock_server(vec![tool_call_sse.to_string(), answer_sse.to_string()]).await;
        let client =
            OpenAI::with_config(openai_oxide::ClientConfig::new("local").base_url(base_url));
        let mut agent = Agent::new(
            client,
            "test-model",
            Arc::new(Mutex::new(crate::mode::Mode::Yolo)),
            10000,
            Duration::from_secs(30),
        );
        let mut events = Vec::new();

        let outcome = agent
            .chat("do it", &mut |event| events.push(event))
            .await
            .unwrap();
        assert_eq!(outcome, ChatOutcome::Answer("done".into()));
        let rejection = agent.context.messages.as_slice().iter().any(|message| {
            matches!(message, Message::Tool { content, .. }
                if content.contains("submit_plan is not allowed in mode yolo"))
        });
        assert!(rejection);
    }

    fn sse_tool_call(id: &str, name: &str, arguments: &str) -> String {
        let payload = serde_json::json!({
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": id,
                        "type": "function",
                        "function": { "name": name, "arguments": arguments }
                    }]
                }
            }]
        });
        format!("data: {payload}\n\ndata: [DONE]\n\n")
    }

    #[tokio::test]
    async fn a_malformed_submit_plan_is_reported_to_the_model_and_retried() {
        let bad_inner = r#"[{"title":"step one","tasks":["do it"]}"#;
        let bad_args = format!("{{\"stages\":{}}}", serde_json::to_string(bad_inner).unwrap());
        let bad_sse = sse_tool_call("call_1", "submit_plan", &bad_args);
        let good_sse = sse_tool_call(
            "call_2",
            "submit_plan",
            r#"{"stages":[{"title":"step one","tasks":["do it"]}]}"#,
        );
        let (base_url, _requests) = mock_server(vec![bad_sse, good_sse]).await;
        let mut agent = plan_agent(base_url);
        let mut events = Vec::new();

        let outcome = agent
            .chat("plan this", &mut |event| events.push(event))
            .await
            .unwrap();
        assert!(matches!(outcome, ChatOutcome::Terminated { .. }));
        let reported = agent.context.messages.iter().any(|message| {
            matches!(message, Message::Tool { content, .. }
                if content.contains("invalid arguments for submit_plan")
                && content.contains("Call submit_plan again with corrected arguments"))
        });
        assert!(reported);
        let shown = events.iter().any(|event| {
            matches!(event, AgentEvent::ToolResult { header, .. } if header == "submit_plan")
        });
        assert!(shown);
    }

    #[tokio::test]
    async fn a_malformed_escalate_is_reported_to_the_model_and_retried() {
        let bad_sse = sse_tool_call(
            "call_1",
            "escalate",
            r#"{"findings":123}"#,
        );
        let good_sse = sse_tool_call(
            "call_2",
            "escalate",
            r#"{"findings":"the build is broken"}"#,
        );
        let (base_url, _requests) = mock_server(vec![bad_sse, good_sse]).await;
        let client =
            OpenAI::with_config(openai_oxide::ClientConfig::new("local").base_url(base_url));
        let mut agent = Agent::new(
            client,
            "test-model",
            Arc::new(Mutex::new(crate::mode::Mode::Implement)),
            10000,
            Duration::from_secs(30),
        );
        let mut events = Vec::new();

        let outcome = agent
            .chat("implement it", &mut |event| events.push(event))
            .await
            .unwrap();
        assert!(matches!(outcome, ChatOutcome::Terminated { .. }));
        let reported = agent.context.messages.iter().any(|message| {
            matches!(message, Message::Tool { content, .. }
                if content.contains("invalid arguments for escalate")
                && content.contains("Call escalate again with corrected arguments"))
        });
        assert!(reported);
    }
}
