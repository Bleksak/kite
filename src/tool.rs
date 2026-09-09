use std::process::Stdio;
use std::time::Duration;

use tokio::io::AsyncReadExt;
use tokio::process::Command;

use similar::TextDiff;

use openai_oxide::types::chat::{FunctionDef, Tool as OpenAITool, ToolCall as OpenAIToolCall};
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Debug, thiserror::Error)]
pub enum ToolError {
    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error("{tool} exited with {status}\n{output}")]
    NonZeroExit {
        tool: &'static str,
        status: i32,
        output: String,
    },

    #[error("{tool} timed out after {seconds}s")]
    TimedOut { tool: &'static str, seconds: u64 },

    #[error("terminator tools are intercepted and never executed")]
    TerminatorNotExecutable,

    #[error("ask_user is answered interactively by the TUI and never executed directly")]
    InteractiveNotExecutable,

    #[error("webfetch of {url} failed: {source}")]
    WebFetchFailed { url: String, source: reqwest::Error },

    #[error("webfetch of {url} failed with status {status}")]
    WebFetchStatus { url: String, status: u16 },

    #[error("expected exactly one occurrence of old content in {path}, found {occurrences}")]
    AmbiguousEdit { path: String, occurrences: usize },

    #[error("invalid arguments for {name}: {source}\n{raw}")]
    InvalidArguments {
        name: String,
        raw: String,
        source: serde_json::Error,
    },

    #[error("unknown tool {name}")]
    UnknownTool { name: String },
}

#[derive(Debug, Clone, PartialEq)]
pub enum QuestionKind {
    SingleChoice,
    MultiChoice,
    Free,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Question {
    pub prompt: String,
    pub kind: QuestionKind,
    pub options: Vec<String>,
}

#[derive(Debug, PartialEq)]
pub enum Tool {
    Bash(String),
    ReadOnlyBash(String),
    ReadFile(String, Option<usize>, Option<usize>),
    WriteFile(String, String),
    EditFile(String, String, String),
    WebFetch(String),
    BgRun(String),
    AskUser(Vec<Question>),
    SubmitPlan(Vec<PlanStage>),
    Escalate(String),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanStage {
    pub title: String,
    pub tasks: Vec<String>,
}

pub fn plan_text(stages: &[PlanStage]) -> String {
    stages
        .iter()
        .enumerate()
        .map(|(i, stage)| {
            let tasks = stage
                .tasks
                .iter()
                .map(|task| format!("- {task}"))
                .collect::<Vec<_>>()
                .join("\n");
            format!("Step {} of {}: {}\n{}", i + 1, stages.len(), stage.title, tasks)
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

pub fn parse_stages(arguments: &str) -> Vec<PlanStage> {
    let value: serde_json::Value =
        serde_json::from_str(arguments).unwrap_or(serde_json::Value::Null);

    let stages_value = match value.get("stages") {
        Some(serde_json::Value::String(encoded)) => serde_json::from_str::<serde_json::Value>(encoded)
            .ok()
            .or_else(|| Some(serde_json::Value::String(encoded.clone()))),
        other => other.cloned(),
    };

    if let Some(serde_json::Value::Array(items)) = stages_value {
        let parsed: Vec<PlanStage> = items
            .iter()
            .filter_map(|item| {
                if let Some(title) = item.as_str() {
                    return Some(PlanStage {
                        title: title.chars().take(40).collect(),
                        tasks: vec![title.to_string()],
                    });
                }
                let obj = item.as_object()?;
                let title = obj
                    .get("title")
                    .or_else(|| obj.get("name"))
                    .or_else(|| obj.get("step"))
                    .and_then(|v| v.as_str())
                    .map(str::to_string)
                    .unwrap_or_else(|| "Step".to_string());
                let tasks = match obj.get("tasks").or_else(|| obj.get("items")) {
                    Some(serde_json::Value::String(s)) => vec![s.clone()],
                    Some(serde_json::Value::Array(items)) => items
                        .iter()
                        .filter_map(|t| t.as_str().map(str::to_string))
                        .collect(),
                    _ => vec![obj
                        .get("description")
                        .and_then(|v| v.as_str())
                        .map(str::to_string)
                        .unwrap_or_default()],
                };
                if tasks.is_empty() {
                    return None;
                }
                Some(PlanStage {
                    title: title.chars().take(40).collect(),
                    tasks,
                })
            })
            .collect();
        if !parsed.is_empty() {
            return parsed;
        }
    }

    if let Some(plan) = value.get("plan").and_then(|p| p.as_str()) {
        return vec![PlanStage {
            title: "Plan".to_string(),
            tasks: vec![plan.to_string()],
        }];
    }

    vec![PlanStage {
        title: "Plan".to_string(),
        tasks: vec!["(the plan could not be parsed)".to_string()],
    }]
}

pub enum ToolOutput {
    Before(String),
    After,
}

const OUTPUT_LIMIT: usize = 10_000;
const READ_FILE_LINE_CAP: usize = 250;

fn cap_output(text: String) -> String {
    if text.len() <= OUTPUT_LIMIT {
        return text;
    }
    let mut start = text.len() - OUTPUT_LIMIT;
    while !text.is_char_boundary(start) {
        start += 1;
    }
    format!("[truncated {start} bytes]\n{}", &text[start..])
}

struct KillOnDrop(Option<tokio::process::Child>);

impl Drop for KillOnDrop {
    fn drop(&mut self) {
        if let Some(mut child) = self.0.take() {
            let _ = child.start_kill();
        }
    }
}

impl Tool {
    pub fn label(&self) -> &'static str {
        match &self {
            Tool::Bash(_) => "bash",
            Tool::ReadOnlyBash(_) => "readonly_bash",
            Tool::ReadFile(..) => "read_file",
            Tool::WriteFile(_, _) => "write_file",
            Tool::EditFile(_, _, _) => "edit_file",
            Tool::WebFetch(_) => "webfetch",
            Tool::BgRun(_) => "bg_run",
            Tool::AskUser(_) => "ask_user",
            Tool::SubmitPlan(_) => "submit_plan",
            Tool::Escalate(_) => "escalate",
        }
    }

    pub fn header(&self) -> String {
        match self {
            Tool::Bash(_) => "bash".to_string(),
            Tool::ReadOnlyBash(_) => "readonly_bash".to_string(),
            Tool::ReadFile(path, start, end) => {
                let range = match (start, end) {
                    (None, None) => None,
                    (Some(start), None) => Some(format!("[{start}..]")),
                    (None, Some(end)) => Some(format!("[..={end}]")),
                    (Some(start), Some(end)) => Some(format!("[{start}..={end}]")),
                };
                match range {
                    Some(range) => format!("read_file: {path} {range}"),
                    None => format!("read_file: {path}"),
                }
            }
            Tool::WriteFile(path, _) => format!("write_file: {path}"),
            Tool::EditFile(path, _, _) => format!("edit_file: {path}"),
            Tool::WebFetch(url) => format!("webfetch: {url}"),
            Tool::BgRun(_) => "bg_run".to_string(),
            Tool::AskUser(questions) => format!(
                "ask_user: {}",
                questions
                    .first()
                    .map(|q| q.prompt.chars().take(40).collect::<String>())
                    .unwrap_or_default()
            ),
            Tool::SubmitPlan(_) => "submit_plan".to_string(),
            Tool::Escalate(_) => "escalate".to_string(),
        }
    }

    pub fn output(&self) -> ToolOutput {
        match self {
            Tool::Bash(cmd) => ToolOutput::Before(cmd.clone()),
            Tool::ReadOnlyBash(cmd) => ToolOutput::Before(cmd.clone()),
            Tool::ReadFile(..) => ToolOutput::After,
            Tool::WriteFile(_, content) => ToolOutput::Before(content.clone()),
            Tool::EditFile(..) => ToolOutput::After,
            Tool::WebFetch(url) => ToolOutput::Before(url.clone()),
            Tool::BgRun(command) => ToolOutput::Before(command.clone()),
            Tool::AskUser(questions) => ToolOutput::Before(
                questions
                    .iter()
                    .enumerate()
                    .map(|(i, q)| format!("{}. {}", i + 1, q.prompt))
                    .collect::<Vec<_>>()
                    .join("\n"),
            ),
            Tool::SubmitPlan(stages) => ToolOutput::Before(plan_text(stages)),
            Tool::Escalate(findings) => ToolOutput::Before(findings.clone()),
        }
    }

    pub fn description(&self) -> &'static str {
        match &self {
            Tool::Bash(_) => "Run a bash script",
            Tool::ReadOnlyBash(_) => "Run a bash script in a read-only mode",
            Tool::ReadFile(_, _, _) => {
                "Read a file, optionally a line range (1-based start and end line, both inclusive). For large files, prefer a line range — a full read is capped to the first 250 lines"
            }
            Tool::WriteFile(_, _) => "Write a file",
            Tool::EditFile(_, _, _) => "Edit a file",
            Tool::WebFetch(_) => "Fetch a URL and return the response body",
            Tool::BgRun(_) => {
                "Run a long-running command in the background (tests, builds, dev servers). Returns a task id immediately; the task's result is reported automatically when it finishes"
            }
            Tool::AskUser(_) => {
                "Ask the user one or more questions and wait for their answers. Use it when you need a decision, preference, or information only the user can provide. Question types: single_choice (the user picks one option or types their own answer), multi_choice (the user picks any number of options and/or types their own answer), free (the user types their own answer). Provide options for the choice types. You may ask several questions in one call; the user answers them step by step. Call this alone, without other tools."
            }
            Tool::SubmitPlan(_) => {
                "Submit the final plan, split into stages. Call this alone, without other tools."
            }
            Tool::Escalate(_) => {
                "Escalate a blocker you cannot resolve. Call this alone, without other tools."
            }
        }
    }

    pub async fn invoke(&self, timeout: Duration) -> Result<String, ToolError> {
        match &self {
            Tool::Bash(script) => {
                let child = Command::new("bash")
                    .arg("-c")
                    .arg(script)
                    .stdin(Stdio::null())
                    .stdout(Stdio::piped())
                    .stderr(Stdio::piped())
                    .spawn()
                    .map_err(ToolError::Io)?;
                Self::run_captured(child, timeout, self.label()).await
            }
            Tool::ReadOnlyBash(script) => {
                let cwd = std::env::current_dir().map_err(ToolError::Io)?;
                let child = Command::new("bwrap")
                    .args(["--ro-bind", "/", "/", "--ro-bind"])
                    .arg(&cwd)
                    .arg(&cwd)
                    .args([
                        "--tmpfs", "/tmp", "--proc", "/proc", "--dev", "/dev", "--chdir",
                    ])
                    .arg(&cwd)
                    .args(["--unshare-pid", "--unshare-ipc", "--unshare-uts"])
                    .arg("bash")
                    .arg("-c")
                    .arg(script)
                    .stdin(Stdio::null())
                    .stdout(Stdio::piped())
                    .stderr(Stdio::piped())
                    .spawn()
                    .map_err(ToolError::Io)?;
                Self::run_captured(child, timeout, self.label()).await
            }
            Tool::ReadFile(file, start, end) => {
                let contents = tokio::fs::read_to_string(file)
                    .await
                    .map_err(ToolError::Io)?;
                let lines: Vec<&str> = contents.lines().collect();
                let total = lines.len();
                let start = start.unwrap_or(1).saturating_sub(1).min(total);
                let end = end
                    .map_or(total, |end| end.min(total))
                    .max(start);
                if end - start > READ_FILE_LINE_CAP {
                    let capped_end = start + READ_FILE_LINE_CAP;
                    let body = lines[start..capped_end].join("\n");
                    return Ok(format!(
                        "{body}\n[showing lines {}–{} of {}; call read_file with start/end for the rest]",
                        start + 1,
                        capped_end,
                        total
                    ));
                }
                Ok(lines[start..end].join("\n"))
            }
            Tool::WriteFile(file, content) => {
                tokio::fs::write(file, content)
                    .await
                    .map_err(ToolError::Io)?;
                Ok(format!("wrote {} bytes to {file}", content.len()))
            }
            Tool::EditFile(file, old_content, new_content) => {
                let mut contents = tokio::fs::read_to_string(file)
                    .await
                    .map_err(ToolError::Io)?;
                let original = contents.clone();
                let crlf = contents.contains("\r\n");
                let old = if crlf {
                    old_content.replace('\n', "\r\n")
                } else {
                    old_content.clone()
                };
                let new = if crlf {
                    new_content.replace('\n', "\r\n")
                } else {
                    new_content.clone()
                };
                let mut occurrences = 0;
                let mut first_index = 0;
                for (index, _) in contents.match_indices(&old) {
                    if occurrences == 0 {
                        first_index = index;
                    }
                    occurrences += 1;
                    if occurrences > 1 {
                        break;
                    }
                }
                if occurrences != 1 {
                    return Err(ToolError::AmbiguousEdit {
                        path: file.clone(),
                        occurrences,
                    });
                }

                contents.replace_range(first_index..first_index + old.len(), &new);

                let text_diff = TextDiff::from_lines(&original, &contents);
                let mut unified = text_diff.unified_diff();
                unified.header(file, file);
                let diff = unified.to_string();

                tokio::fs::write(file, contents)
                    .await
                    .map_err(ToolError::Io)?;
                Ok(diff)
            }
            Tool::WebFetch(url) => {
                let collected = tokio::time::timeout(timeout, async {
                    let response = reqwest::Client::new()
                        .get(url.as_str())
                        .send()
                        .await
                        .map_err(|source| ToolError::WebFetchFailed {
                            url: url.clone(),
                            source,
                        })?;
                    let status = response.status().as_u16();
                    let body =
                        response
                            .text()
                            .await
                            .map_err(|source| ToolError::WebFetchFailed {
                                url: url.clone(),
                                source,
                            })?;
                    Ok::<(u16, String), ToolError>((status, body))
                })
                .await;

                let (status, body) = match collected {
                    Ok(Ok(value)) => value,
                    Ok(Err(error)) => return Err(error),
                    Err(_) => {
                        return Err(ToolError::TimedOut {
                            tool: self.label(),
                            seconds: timeout.as_secs(),
                        });
                    }
                };

                if status >= 400 {
                    return Err(ToolError::WebFetchStatus {
                        url: url.clone(),
                        status,
                    });
                }

                Ok(cap_output(body))
            }
            Tool::BgRun(command) => {
                let id = crate::bg::REGISTRY
                    .run(command)
                    .map_err(|source| ToolError::Io(std::io::Error::other(source.to_string())))?;
                Ok(format!(
                    "task {id} started; its result will be reported when it finishes"
                ))
            }
            Tool::SubmitPlan(_) => Err(ToolError::TerminatorNotExecutable),
            Tool::Escalate(_) => Err(ToolError::TerminatorNotExecutable),
            Tool::AskUser(_) => Err(ToolError::InteractiveNotExecutable),
        }
    }

    async fn run_captured(
        child: tokio::process::Child,
        timeout: Duration,
        tool: &'static str,
    ) -> Result<String, ToolError> {
        let mut guard = KillOnDrop(Some(child));
        let collected = tokio::time::timeout(timeout, async {
            let child = guard.0.as_mut().unwrap();
            let mut stdout = Vec::new();
            let mut stderr = Vec::new();
            if let Some(mut out) = child.stdout.take() {
                out.read_to_end(&mut stdout).await.map_err(ToolError::Io)?;
            }
            if let Some(mut err) = child.stderr.take() {
                err.read_to_end(&mut stderr).await.map_err(ToolError::Io)?;
            }
            let status = child.wait().await.map_err(ToolError::Io)?;
            Ok::<(std::process::ExitStatus, Vec<u8>, Vec<u8>), ToolError>((status, stdout, stderr))
        })
        .await;

        let (status, stdout, stderr) = match collected {
            Ok(Ok(value)) => value,
            Ok(Err(error)) => return Err(error),
            Err(_) => {
                if let Some(mut child) = guard.0.take() {
                    let _ = child.kill().await;
                }
                return Err(ToolError::TimedOut {
                    tool,
                    seconds: timeout.as_secs(),
                });
            }
        };

        if !status.success() {
            let mut combined = String::from_utf8_lossy(&stdout).into_owned();
            let stderr = String::from_utf8_lossy(&stderr).into_owned();
            if !stderr.is_empty() {
                if !combined.is_empty() {
                    combined.push('\n');
                }
                combined.push_str(&stderr);
            }
            return Err(ToolError::NonZeroExit {
                tool,
                status: status.code().unwrap_or(-1),
                output: cap_output(combined),
            });
        }

        Ok(cap_output(String::from_utf8_lossy(&stdout).into_owned()))
    }
}

pub fn tool_definitions(tools: &[Tool]) -> Vec<OpenAITool> {
    tools
        .iter()
        .map(|tool| OpenAITool {
            type_: "function".to_string(),
            function: FunctionDef {
                name: tool.label().to_string(),
                description: Some(tool.description().to_string()),
                parameters: Some(parameters(tool)),
                strict: None,
            },
        })
        .collect()
}

#[derive(Deserialize)]
struct BashArgs {
    command: String,
}

#[derive(Deserialize)]
struct ReadFileArgs {
    path: String,
    #[serde(default)]
    start: Option<usize>,
    #[serde(default)]
    end: Option<usize>,
}

#[derive(Deserialize)]
struct WriteFileArgs {
    path: String,
    content: String,
}

#[derive(Deserialize)]
struct EditFileArgs {
    path: String,
    old_content: String,
    new_content: String,
}

#[derive(Deserialize)]
struct WebFetchArgs {
    url: String,
}

#[derive(Deserialize)]
struct EscalateArgs {
    findings: String,
}

#[derive(Deserialize)]
struct AskUserArgs {
    questions: Vec<QuestionArg>,
}

#[derive(Deserialize)]
struct QuestionArg {
    prompt: String,
    #[serde(rename = "type")]
    kind: QuestionKindWire,
    #[serde(default)]
    options: Vec<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum QuestionKindWire {
    SingleChoice,
    MultiChoice,
    Free,
}

impl QuestionKindWire {
    fn into_kind(self) -> QuestionKind {
        match self {
            QuestionKindWire::SingleChoice => QuestionKind::SingleChoice,
            QuestionKindWire::MultiChoice => QuestionKind::MultiChoice,
            QuestionKindWire::Free => QuestionKind::Free,
        }
    }
}

fn parse_args<A: serde::de::DeserializeOwned>(name: &str, arguments: &str) -> Result<A, ToolError> {
    serde_json::from_str(arguments).map_err(|source| ToolError::InvalidArguments {
        name: name.to_string(),
        raw: arguments.to_string(),
        source,
    })
}

impl TryFrom<OpenAIToolCall> for Tool {
    type Error = ToolError;

    fn try_from(call: OpenAIToolCall) -> Result<Tool, ToolError> {
        let OpenAIToolCall {
            id: _,
            type_: _,
            function,
        } = call;

        match function.name.as_str() {
            "bash" => parse_args::<BashArgs>(&function.name, &function.arguments)
                .map(|a| Tool::Bash(a.command)),
            "readonly_bash" => parse_args::<BashArgs>(&function.name, &function.arguments)
                .map(|a| Tool::ReadOnlyBash(a.command)),
            "read_file" => parse_args::<ReadFileArgs>(&function.name, &function.arguments)
                .map(|a| Tool::ReadFile(a.path, a.start, a.end)),
            "write_file" => parse_args::<WriteFileArgs>(&function.name, &function.arguments)
                .map(|a| Tool::WriteFile(a.path, a.content)),
            "edit_file" => parse_args::<EditFileArgs>(&function.name, &function.arguments)
                .map(|a| Tool::EditFile(a.path, a.old_content, a.new_content)),
            "webfetch" => parse_args::<WebFetchArgs>(&function.name, &function.arguments)
                .map(|a| Tool::WebFetch(a.url)),
            "bg_run" => parse_args::<BashArgs>(&function.name, &function.arguments)
                .map(|a| Tool::BgRun(a.command)),
            "submit_plan" => {
                let stages = parse_stages(&function.arguments);
                if stages
                    .iter()
                    .any(|s| s.title.trim().is_empty() || s.tasks.is_empty())
                {
                    Err(ToolError::InvalidArguments {
                        name: function.name.clone(),
                        raw: function.arguments.clone(),
                        source: serde::de::Error::custom(
                            "each stage needs a title and at least one task",
                        ),
                    })
                } else {
                    Ok(Tool::SubmitPlan(stages))
                }
            }
            "escalate" => parse_args::<EscalateArgs>(&function.name, &function.arguments)
                .map(|a| Tool::Escalate(a.findings)),
            "ask_user" => parse_args::<AskUserArgs>(&function.name, &function.arguments)
                .and_then(|a| {
                    let mut questions = Vec::new();
                    for q in a.questions {
                        let kind = q.kind.into_kind();
                        let choice = matches!(
                            kind,
                            QuestionKind::SingleChoice | QuestionKind::MultiChoice
                        );
                        if choice && q.options.is_empty() {
                            return Err(ToolError::InvalidArguments {
                                name: function.name.clone(),
                                raw: function.arguments.clone(),
                                source: serde::de::Error::custom(
                                    "options are required for single_choice and multi_choice",
                                ),
                            });
                        }
                        questions.push(Question {
                            prompt: q.prompt,
                            kind,
                            options: if choice { q.options } else { Vec::new() },
                        });
                    }
                    if questions.is_empty() {
                        return Err(ToolError::InvalidArguments {
                            name: function.name.clone(),
                            raw: function.arguments.clone(),
                            source: serde::de::Error::custom("questions must not be empty"),
                        });
                    }
                    Ok(Tool::AskUser(questions))
                }),
            other => Err(ToolError::UnknownTool {
                name: other.to_string(),
            }),
        }
    }
}

fn parameters(tool: &Tool) -> serde_json::Value {
    match tool {
        Tool::Bash(_) | Tool::ReadOnlyBash(_) | Tool::BgRun(_) => json!({
            "type": "object",
            "properties": {
                "command": { "type": "string", "description": "the shell command to run" }
            },
            "required": ["command"]
        }),
        Tool::ReadFile(..) => json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "path of the file to read; a full read is capped to the first 250 lines, so use start/end for large files" },
                "start": { "type": "integer", "minimum": 1, "description": "1-based start line, inclusive" },
                "end": { "type": "integer", "minimum": 1, "description": "1-based end line, inclusive" }
            },
            "required": ["path"]
        }),
        Tool::WriteFile(_, _) => json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "path of the file to write" },
                "content": { "type": "string", "description": "content to write to the file" }
            },
            "required": ["path", "content"]
        }),
        Tool::EditFile(_, _, _) => json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "path of the file to edit" },
                "old_content": { "type": "string", "description": "exact text to replace; must occur exactly once in the file" },
                "new_content": { "type": "string", "description": "replacement text; may be empty to delete" }
            },
            "required": ["path", "old_content", "new_content"]
        }),
        Tool::WebFetch(_) => json!({
            "type": "object",
            "properties": {
                "url": { "type": "string", "description": "the URL to fetch" }
            },
            "required": ["url"]
        }),
        Tool::AskUser(_) => json!({
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "prompt": { "type": "string", "description": "the question to ask" },
                            "type": {
                                "type": "string",
                                "enum": ["single_choice", "multi_choice", "free"],
                                "description": "single_choice: the user picks one option or types their own answer; multi_choice: the user picks any number of options and/or types their own answer; free: the user types their own answer"
                            },
                            "options": {
                                "type": "array",
                                "items": { "type": "string" },
                                "description": "the options, required for single_choice and multi_choice"
                            }
                        },
                        "required": ["prompt", "type"]
                    }
                }
            },
            "required": ["questions"]
        }),
        Tool::SubmitPlan(_) => json!({
            "type": "object",
            "properties": {
                "stages": {
                    "type": "array",
                    "minItems": 1,
                    "description": "the implementation plan split into stages; each stage is a self-contained set of tasks the user reviews before it is implemented",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": { "type": "string", "description": "a short stage name" },
                            "tasks": {
                                "type": "array",
                                "minItems": 1,
                                "items": { "type": "string" },
                                "description": "the concrete tasks of this stage"
                            }
                        },
                        "required": ["title", "tasks"]
                    }
                }
            },
            "required": ["stages"]
        }),
        Tool::Escalate(_) => json!({
            "type": "object",
            "properties": {
                "findings": { "type": "string", "description": "the blocker you cannot resolve, with what was tried" }
            },
            "required": ["findings"]
        }),
    }
}

#[cfg(test)]
mod test {
    use crate::tool::{
        KillOnDrop, Tool, ToolError, Question, QuestionKind, PlanStage, plan_text, parse_stages,
        tool_definitions,
    };
    use openai_oxide::types::chat::{FunctionCall, ToolCall as OpenAIToolCall};
    use std::io::{Read, Write};
    use std::process::Stdio;
    use std::time::Duration;
    use test_files::TestFiles;
    use tokio::process::Command;

    fn timeout() -> Duration {
        Duration::from_secs(30)
    }

    fn call(name: &str, arguments: &str) -> OpenAIToolCall {
        OpenAIToolCall {
            id: "call_1".into(),
            type_: "function".into(),
            function: FunctionCall {
                name: name.into(),
                arguments: arguments.into(),
            },
        }
    }

    #[test]
    fn read_file_header_formats_the_range() {
        assert_eq!(
            Tool::ReadFile("a.txt".into(), None, None).header(),
            "read_file: a.txt"
        );
        assert_eq!(
            Tool::ReadFile("a.txt".into(), Some(10), None).header(),
            "read_file: a.txt [10..]"
        );
        assert_eq!(
            Tool::ReadFile("a.txt".into(), None, Some(20)).header(),
            "read_file: a.txt [..=20]"
        );
        assert_eq!(
            Tool::ReadFile("a.txt".into(), Some(10), Some(20)).header(),
            "read_file: a.txt [10..=20]"
        );
    }

    #[tokio::test]
    async fn bash_hello_world() {
        let script = r#"
            echo "hello world"
        "#;

        let tool = Tool::Bash(script.into());

        let result = tool.invoke(timeout()).await.unwrap();

        assert_eq!(result, "hello world\n");
    }

    #[tokio::test]
    async fn readonly_bash_runs_the_script() {
        let tool = Tool::ReadOnlyBash("echo sandbox-ok".into());

        assert_eq!(tool.invoke(timeout()).await.unwrap(), "sandbox-ok\n");
    }

    #[tokio::test]
    async fn readonly_bash_writes_to_cwd_fail() {
        let cwd = std::env::current_dir().unwrap();
        let tool = Tool::ReadOnlyBash(format!("touch {}/probe", cwd.display()));

        let error = tool.invoke(timeout()).await.unwrap_err();

        assert!(matches!(error, ToolError::NonZeroExit { status: 1, .. }));
    }

    #[tokio::test]
    async fn readonly_bash_writes_to_tmp_succeed() {
        let tool = Tool::ReadOnlyBash("touch /tmp/probe && echo ok".into());

        assert_eq!(tool.invoke(timeout()).await.unwrap(), "ok\n");
    }

    #[tokio::test]
    async fn readonly_bash_reads_cwd_files() {
        let cwd = std::env::current_dir().unwrap();
        let tool = Tool::ReadOnlyBash(format!("cat {}/Cargo.toml", cwd.display()));

        let output = tool.invoke(timeout()).await.unwrap();
        assert!(output.starts_with("[package]"));
    }

    #[tokio::test]
    async fn readonly_bash_hypothesis_test_against_memory_db() {
        let tool = Tool::ReadOnlyBash(
            "python3 -c \"import sqlite3; sqlite3.connect(':memory:').execute('SELECT 1')\" && echo ok".into(),
        );

        assert_eq!(tool.invoke(timeout()).await.unwrap(), "ok\n");
    }

    #[tokio::test]
    async fn readonly_bash_timeout_kills_the_sandbox() {
        let tool = Tool::ReadOnlyBash("sleep 5".into());

        let error = tool.invoke(Duration::from_secs(1)).await.unwrap_err();

        assert!(matches!(
            error,
            ToolError::TimedOut {
                tool: "readonly_bash",
                seconds: 1
            }
        ));
    }

    #[tokio::test]
    async fn read_file() {
        let temp_dir = TestFiles::new();
        temp_dir.file("hello.txt", "hello world");
        let file = temp_dir.path().join("hello.txt");

        assert!(file.exists());

        let tool = Tool::ReadFile(file.to_string_lossy().into(), None, None);

        let result = tool.invoke(timeout()).await.unwrap();

        assert_eq!(result, "hello world");
    }

    #[tokio::test]
    async fn read_file_full_multiline_strips_trailing_newline() {
        let temp_dir = TestFiles::new();
        temp_dir.file("a.txt", "one\ntwo\nthree\n");
        let file = temp_dir.path().join("a.txt");

        let tool = Tool::ReadFile(file.to_string_lossy().into(), None, None);

        assert_eq!(tool.invoke(timeout()).await.unwrap(), "one\ntwo\nthree");
    }

    #[tokio::test]
    async fn read_file_start_line_is_one_based() {
        let temp_dir = TestFiles::new();
        temp_dir.file("a.txt", "one\ntwo\nthree\n");
        let file = temp_dir.path().join("a.txt");

        let tool = Tool::ReadFile(file.to_string_lossy().into(), Some(2), None);

        assert_eq!(tool.invoke(timeout()).await.unwrap(), "two\nthree");
    }

    #[tokio::test]
    async fn read_file_start_and_end_are_inclusive() {
        let temp_dir = TestFiles::new();
        temp_dir.file("a.txt", "one\ntwo\nthree\n");
        let file = temp_dir.path().join("a.txt");

        let tool = Tool::ReadFile(file.to_string_lossy().into(), Some(2), Some(3));

        assert_eq!(tool.invoke(timeout()).await.unwrap(), "two\nthree");
    }

    #[tokio::test]
    async fn read_file_caps_a_full_read_to_the_head() {
        let temp_dir = TestFiles::new();
        let content = (0..300)
            .map(|i| format!("line{i}"))
            .collect::<Vec<_>>()
            .join("\n");
        temp_dir.file("big.txt", &content);
        let file = temp_dir.path().join("big.txt");

        let tool = Tool::ReadFile(file.to_string_lossy().into(), None, None);
        let result = tool.invoke(timeout()).await.unwrap();

        assert!(result.starts_with("line0"), "should start at the head: {result}");
        assert!(result.contains("line249"), "should include line 250 (index 249)");
        assert!(!result.contains("line250"), "should not include line 251 (index 250)");
        assert!(
            result.contains("[showing lines 1–250 of 300"),
            "should have the cap note: {result}"
        );
    }

    #[tokio::test]
    async fn read_file_honors_an_explicit_range_under_the_cap() {
        let temp_dir = TestFiles::new();
        let content = (0..300)
            .map(|i| format!("line{i}"))
            .collect::<Vec<_>>()
            .join("\n");
        temp_dir.file("big.txt", &content);
        let file = temp_dir.path().join("big.txt");

        let tool = Tool::ReadFile(file.to_string_lossy().into(), Some(5), Some(104));
        let result = tool.invoke(timeout()).await.unwrap();

        assert_eq!(
            result,
            (4..104)
                .map(|i| format!("line{i}"))
                .collect::<Vec<_>>()
                .join("\n")
        );
        assert!(!result.contains("[showing"), "should not have the cap note: {result}");
    }

    #[tokio::test]
    async fn read_file_caps_a_range_that_exceeds_the_cap() {
        let temp_dir = TestFiles::new();
        let content = (0..400)
            .map(|i| format!("line{i}"))
            .collect::<Vec<_>>()
            .join("\n");
        temp_dir.file("big.txt", &content);
        let file = temp_dir.path().join("big.txt");

        let tool = Tool::ReadFile(file.to_string_lossy().into(), Some(100), None);
        let result = tool.invoke(timeout()).await.unwrap();

        assert!(result.starts_with("line99"), "should start at the requested start: {result}");
        assert!(result.contains("line348"), "should include 250 lines from the start");
        assert!(!result.contains("line349"), "should not exceed the cap");
        assert!(
            result.contains("[showing lines 100–349 of 400"),
            "should have the cap note: {result}"
        );
    }

    #[tokio::test]
    async fn read_file_single_line() {
        let temp_dir = TestFiles::new();
        temp_dir.file("a.txt", "one\ntwo\nthree\n");
        let file = temp_dir.path().join("a.txt");

        let tool = Tool::ReadFile(file.to_string_lossy().into(), Some(1), Some(1));

        assert_eq!(tool.invoke(timeout()).await.unwrap(), "one");
    }

    #[tokio::test]
    async fn read_file_end_beyond_eof_returns_rest() {
        let temp_dir = TestFiles::new();
        temp_dir.file("a.txt", "one\ntwo\nthree\n");
        let file = temp_dir.path().join("a.txt");

        let tool = Tool::ReadFile(file.to_string_lossy().into(), Some(2), Some(10));

        assert_eq!(tool.invoke(timeout()).await.unwrap(), "two\nthree");
    }

    #[tokio::test]
    async fn read_file_start_beyond_eof_is_empty() {
        let temp_dir = TestFiles::new();
        temp_dir.file("a.txt", "one\n");
        let file = temp_dir.path().join("a.txt");

        let tool = Tool::ReadFile(file.to_string_lossy().into(), Some(99), None);

        assert_eq!(tool.invoke(timeout()).await.unwrap(), "");
    }

    #[tokio::test]
    async fn read_file_start_after_end_is_empty() {
        let temp_dir = TestFiles::new();
        temp_dir.file("a.txt", "one\ntwo\nthree\n");
        let file = temp_dir.path().join("a.txt");

        let tool = Tool::ReadFile(file.to_string_lossy().into(), Some(5), Some(2));

        assert_eq!(tool.invoke(timeout()).await.unwrap(), "");
    }

    #[tokio::test]
    async fn write_file() {
        let temp_dir = TestFiles::new();
        let file = temp_dir.path().join("hello.txt");

        assert!(!file.exists());

        let tool = Tool::WriteFile(file.to_string_lossy().into(), "hello world".into());

        let result = tool.invoke(timeout()).await.unwrap();

        assert_eq!(
            result,
            format!("wrote 11 bytes to {}", file.to_string_lossy())
        );
        assert!(file.exists());
    }

    #[tokio::test]
    async fn edit_file() {
        let temp_dir = TestFiles::new();
        temp_dir.file(
            "hello.txt",
            r#"---
version: 3"#,
        );

        let file = temp_dir.path().join("hello.txt");

        assert!(file.exists());

        let tool = Tool::EditFile(
            file.to_string_lossy().into(),
            "---\nver".into(),
            "test".into(),
        );

        let result = tool.invoke(timeout()).await.unwrap();

        let path = file.to_string_lossy().to_string();
        assert!(result.starts_with(&format!("--- {path}\n+++ {path}\n")));
        assert!(result.contains("@@"));
        assert!(result.contains("----"));
        assert!(result.contains("+test"));
        assert!(file.exists());
        assert_eq!(std::fs::read_to_string(file).unwrap(), "testsion: 3");
    }

    #[tokio::test]
    async fn edit_file_crlf_file_keeps_crlf() {
        let temp_dir = TestFiles::new();
        temp_dir.file("a.txt", "line one\r\nline two\r\n");
        let file = temp_dir.path().join("a.txt");

        let tool = Tool::EditFile(
            file.to_string_lossy().into(),
            "line one\nline two".into(),
            "line ONE\nline two".into(),
        );

        let result = tool.invoke(timeout()).await.unwrap();

        assert!(result.contains("-line one"));
        assert!(result.contains("+line ONE"));
        assert_eq!(
            std::fs::read_to_string(file).unwrap(),
            "line ONE\r\nline two\r\n"
        );
    }

    #[tokio::test]
    async fn edit_file_crlf_single_line() {
        let temp_dir = TestFiles::new();
        temp_dir.file("a.txt", "line one\r\nline two\r\n");
        let file = temp_dir.path().join("a.txt");

        let tool = Tool::EditFile(
            file.to_string_lossy().into(),
            "line one".into(),
            "line ONE".into(),
        );

        tool.invoke(timeout()).await.unwrap();

        assert_eq!(
            std::fs::read_to_string(file).unwrap(),
            "line ONE\r\nline two\r\n"
        );
    }

    #[tokio::test]
    async fn edit_file_output_is_unified_diff() {
        let temp_dir = TestFiles::new();
        temp_dir.file("a.txt", "one\ntwo\nthree\n");
        let file = temp_dir.path().join("a.txt");

        let tool = Tool::EditFile(file.to_string_lossy().into(), "two".into(), "TWO".into());

        let result = tool.invoke(timeout()).await.unwrap();

        let path = file.to_string_lossy().to_string();
        assert!(result.starts_with(&format!("--- {path}\n+++ {path}\n")));
        assert!(result.contains("@@ -1,3 +1,3 @@"));
        assert!(result.contains(" one\n"));
        assert!(result.contains("-two\n"));
        assert!(result.contains("+TWO\n"));
        assert!(result.contains(" three\n"));
    }

    #[tokio::test]
    async fn edit_file_ambiguous_old_content_is_error() {
        let temp_dir = TestFiles::new();
        temp_dir.file("a.txt", "a b a");
        let file = temp_dir.path().join("a.txt");

        let tool = Tool::EditFile(file.to_string_lossy().into(), "a".into(), "c".into());

        let result = tool.invoke(timeout()).await.unwrap_err();

        assert!(matches!(
            result,
            ToolError::AmbiguousEdit { occurrences: 2, .. }
        ));
        assert_eq!(std::fs::read_to_string(file).unwrap(), "a b a");
    }

    #[tokio::test]
    async fn edit_file_missing_old_content_is_error() {
        let temp_dir = TestFiles::new();
        temp_dir.file("a.txt", "hello world");
        let file = temp_dir.path().join("a.txt");

        let tool = Tool::EditFile(file.to_string_lossy().into(), "nope".into(), "x".into());

        let result = tool.invoke(timeout()).await.unwrap_err();

        assert!(matches!(
            result,
            ToolError::AmbiguousEdit { occurrences: 0, .. }
        ));
        assert_eq!(std::fs::read_to_string(file).unwrap(), "hello world");
    }

    #[tokio::test]
    async fn bash_nonzero_exit_is_error() {
        let tool = Tool::Bash("exit 3".into());

        let result = tool.invoke(timeout()).await.unwrap_err();

        assert!(matches!(result, ToolError::NonZeroExit { status: 3, .. }));
    }

    #[tokio::test]
    async fn bash_stderr_is_collected_in_error() {
        let tool = Tool::Bash(r#"echo "oops" 1>&2; exit 1"#.into());

        let error = tool.invoke(timeout()).await.unwrap_err();
        let message = error.to_string();

        assert!(message.contains("bash exited with 1"));
        assert!(message.contains("oops"));
    }

    #[tokio::test]
    async fn bash_stdout_is_kept_on_failure() {
        let tool = Tool::Bash(r#"echo "failure detail"; exit 1"#.into());

        let error = tool.invoke(timeout()).await.unwrap_err();
        let message = error.to_string();

        assert!(message.contains("bash exited with 1"));
        assert!(message.contains("failure detail"));
    }

    #[tokio::test]
    async fn bash_failure_combines_stdout_and_stderr() {
        let tool = Tool::Bash(r#"echo "out-line"; echo "err-line" 1>&2; exit 1"#.into());

        let error = tool.invoke(timeout()).await.unwrap_err();
        let message = error.to_string();

        assert!(message.contains("out-line"));
        assert!(message.contains("err-line"));
    }

    #[tokio::test]
    async fn bash_timeout_kills_the_child() {
        let tool = Tool::Bash("sleep 5".into());

        let error = tool.invoke(Duration::from_secs(1)).await.unwrap_err();

        assert!(matches!(
            error,
            ToolError::TimedOut {
                tool: "bash",
                seconds: 1
            }
        ));
    }

    #[tokio::test]
    async fn dropping_the_guard_kills_the_child() {
        let child = Command::new("bash")
            .arg("-c")
            .arg("sleep 30")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let pid = child.id().unwrap();
        let guard = KillOnDrop(Some(child));
        drop(guard);
        let mut dead = false;
        for _ in 0..200 {
            match std::fs::read_to_string(format!("/proc/{pid}/stat")) {
                Ok(stat) => {
                    let state = stat
                        .rsplit(')')
                        .next()
                        .and_then(|rest| rest.split_whitespace().next())
                        .unwrap_or_default();
                    if state == "Z" || state.is_empty() {
                        dead = true;
                        break;
                    }
                }
                Err(_) => {
                    dead = true;
                    break;
                }
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        assert!(dead, "child {pid} was not killed by dropping the guard");
    }

    #[tokio::test]
    async fn bash_output_is_capped_to_the_tail() {
        let tool = Tool::Bash(r#"head -c 100000 /dev/zero | tr '\0' a"#.into());

        let result = tool.invoke(timeout()).await.unwrap();

        assert!(result.starts_with("[truncated "));
        assert!(result.contains("bytes]\n"));
        assert!(result.len() < 11_000);
        assert!(result.trim_end().ends_with('a'));
    }

    #[tokio::test]
    async fn bash_missing_command_is_error_127() {
        let error = Tool::Bash("definitely-not-a-command".into())
            .invoke(timeout())
            .await
            .unwrap_err();

        let ToolError::NonZeroExit { status, output, .. } = error else {
            panic!("expected NonZeroExit, got {error:?}");
        };

        assert_eq!(status, 127);
        assert!(output.contains("not found"));
    }

    #[tokio::test]
    async fn bash_empty_output_is_ok() {
        let tool = Tool::Bash("true".into());

        assert_eq!(tool.invoke(timeout()).await.unwrap(), "");
    }

    #[tokio::test]
    async fn bash_stderr_on_success_is_ignored() {
        let tool = Tool::Bash(r#"echo "out"; echo "warn" 1>&2"#.into());

        assert_eq!(tool.invoke(timeout()).await.unwrap(), "out\n");
    }

    #[tokio::test]
    async fn bash_stdout_non_ascii_roundtrip() {
        let tool = Tool::Bash("printf 'héllo 🚀'".into());

        assert_eq!(tool.invoke(timeout()).await.unwrap(), "héllo 🚀");
    }

    #[tokio::test]
    async fn read_file_missing_is_error() {
        let tool = Tool::ReadFile("/nonexistent/nope.txt".into(), None, None);

        let error = tool.invoke(timeout()).await.unwrap_err();

        assert!(matches!(
            error,
            ToolError::Io(e) if e.kind() == std::io::ErrorKind::NotFound
        ));
    }

    #[tokio::test]
    async fn read_file_non_utf8_is_error() {
        let temp_dir = TestFiles::new();
        let file = temp_dir.path().join("binary.bin");
        std::fs::write(&file, [0xff, 0xfe, 0x00]).unwrap();

        let tool = Tool::ReadFile(file.to_string_lossy().into(), None, None);

        let error = tool.invoke(timeout()).await.unwrap_err();

        assert!(matches!(
            error,
            ToolError::Io(e) if e.kind() == std::io::ErrorKind::InvalidData
        ));
    }

    #[tokio::test]
    async fn write_file_overwrites_existing() {
        let temp_dir = TestFiles::new();
        temp_dir.file("a.txt", "old");
        let file = temp_dir.path().join("a.txt");

        let tool = Tool::WriteFile(file.to_string_lossy().into(), "new".into());

        tool.invoke(timeout()).await.unwrap();
        assert_eq!(std::fs::read_to_string(file).unwrap(), "new");
    }

    #[tokio::test]
    async fn edit_file_delete_text() {
        let temp_dir = TestFiles::new();
        temp_dir.file("a.txt", "hello world");
        let file = temp_dir.path().join("a.txt");

        let tool = Tool::EditFile(file.to_string_lossy().into(), "hello ".into(), "".into());

        tool.invoke(timeout()).await.unwrap();
        assert_eq!(std::fs::read_to_string(file).unwrap(), "world");
    }

    #[tokio::test]
    async fn edit_file_matches_literally_not_regex() {
        let temp_dir = TestFiles::new();
        temp_dir.file("a.txt", "a.b and aXb");
        let file = temp_dir.path().join("a.txt");

        let tool = Tool::EditFile(file.to_string_lossy().into(), "a.b".into(), "a_b".into());

        tool.invoke(timeout()).await.unwrap();
        assert_eq!(std::fs::read_to_string(file).unwrap(), "a_b and aXb");
    }

    #[tokio::test]
    async fn edit_file_multiline_and_non_ascii() {
        let temp_dir = TestFiles::new();
        temp_dir.file("a.txt", "héllo line1\nline2\nline3");
        let file = temp_dir.path().join("a.txt");

        let tool = Tool::EditFile(
            file.to_string_lossy().into(),
            "héllo line1\nline2".into(),
            "hej X".into(),
        );

        tool.invoke(timeout()).await.unwrap();
        assert_eq!(std::fs::read_to_string(file).unwrap(), "hej X\nline3");
    }

    #[test]
    fn try_from_bash() {
        let tool = Tool::try_from(call("bash", r#"{"command":"ls -la"}"#)).unwrap();
        assert_eq!(tool, Tool::Bash("ls -la".into()));
    }

    #[test]
    fn try_from_readonly_bash() {
        let tool = Tool::try_from(call("readonly_bash", r#"{"command":"ls"}"#)).unwrap();
        assert_eq!(tool, Tool::ReadOnlyBash("ls".into()));
    }

    #[test]
    fn try_from_read_file_full() {
        let tool = Tool::try_from(call("read_file", r#"{"path":"a.txt"}"#)).unwrap();
        assert_eq!(tool, Tool::ReadFile("a.txt".into(), None, None));
    }

    #[test]
    fn try_from_read_file_with_range() {
        let tool =
            Tool::try_from(call("read_file", r#"{"path":"a.txt","start":2,"end":4}"#)).unwrap();
        assert_eq!(tool, Tool::ReadFile("a.txt".into(), Some(2), Some(4)));
    }

    #[test]
    fn try_from_write_file() {
        let tool =
            Tool::try_from(call("write_file", r#"{"path":"a.txt","content":"hi"}"#)).unwrap();
        assert_eq!(tool, Tool::WriteFile("a.txt".into(), "hi".into()));
    }

    #[test]
    fn try_from_edit_file() {
        let tool = Tool::try_from(call(
            "edit_file",
            r#"{"path":"a.txt","old_content":"a","new_content":"b"}"#,
        ))
        .unwrap();
        assert_eq!(tool, Tool::EditFile("a.txt".into(), "a".into(), "b".into()));
    }

    #[test]
    fn try_from_webfetch() {
        let tool = Tool::try_from(call("webfetch", r#"{"url":"https://example.com"}"#)).unwrap();
        assert_eq!(tool, Tool::WebFetch("https://example.com".into()));
    }

    #[test]
    fn try_from_invalid_json_is_error_with_raw() {
        let error = Tool::try_from(call("bash", r#"{"command":"ls""#)).unwrap_err();
        let message = error.to_string();

        let ToolError::InvalidArguments { name, raw, .. } = error else {
            panic!("expected InvalidArguments, got {message:?}");
        };
        assert_eq!(name, "bash");
        assert_eq!(raw, r#"{"command":"ls""#);
        assert!(message.contains("bash"));
    }

    #[test]
    fn try_from_missing_field_is_error() {
        let error = Tool::try_from(call("read_file", r#"{"start":1}"#)).unwrap_err();

        assert!(matches!(error, ToolError::InvalidArguments { .. }));
    }

    #[test]
    fn try_from_wrong_type_is_error() {
        let error =
            Tool::try_from(call("read_file", r#"{"path":"a.txt","start":"first"}"#)).unwrap_err();

        assert!(matches!(error, ToolError::InvalidArguments { .. }));
    }

    #[test]
    fn try_from_unknown_tool_is_error() {
        let error = Tool::try_from(call("nuke", "{}")).unwrap_err();

        assert!(matches!(error, ToolError::UnknownTool { name } if name == "nuke"));
    }

    #[test]
    fn try_from_ignores_hallucinated_fields() {
        let tool = Tool::try_from(call("bash", r#"{"command":"ls","vibes":42}"#)).unwrap();
        assert_eq!(tool, Tool::Bash("ls".into()));
    }

    #[test]
    fn try_from_ask_user_single_choice() {
        let tool = Tool::try_from(call(
            "ask_user",
            r#"{"questions":[{"prompt":"which db?","type":"single_choice","options":["pg","mysql"]}]}"#,
        ))
        .unwrap();
        assert_eq!(
            tool,
            Tool::AskUser(vec![Question {
                prompt: "which db?".into(),
                kind: QuestionKind::SingleChoice,
                options: vec!["pg".into(), "mysql".into()],
            }])
        );
    }

    #[test]
    fn try_from_ask_user_multi_and_free() {
        let tool = Tool::try_from(call(
            "ask_user",
            r#"{"questions":[{"prompt":"pick some","type":"multi_choice","options":["a","b"]},{"prompt":"name?","type":"free"}]}"#,
        ))
        .unwrap();
        assert_eq!(
            tool,
            Tool::AskUser(vec![
                Question {
                    prompt: "pick some".into(),
                    kind: QuestionKind::MultiChoice,
                    options: vec!["a".into(), "b".into()],
                },
                Question {
                    prompt: "name?".into(),
                    kind: QuestionKind::Free,
                    options: Vec::new(),
                },
            ])
        );
    }

    #[test]
    fn try_from_ask_user_choice_without_options_is_error() {
        let error = Tool::try_from(call(
            "ask_user",
            r#"{"questions":[{"prompt":"which?","type":"single_choice"}]}"#,
        ))
        .unwrap_err();
        assert!(matches!(error, ToolError::InvalidArguments { .. }));
    }

    #[test]
    fn try_from_ask_user_empty_questions_is_error() {
        let error = Tool::try_from(call("ask_user", r#"{"questions":[]}"#)).unwrap_err();
        assert!(matches!(error, ToolError::InvalidArguments { .. }));
    }

    #[tokio::test]
    async fn ask_user_is_not_executable_directly() {
        let tool = Tool::AskUser(vec![Question {
            prompt: "q?".into(),
            kind: QuestionKind::Free,
            options: Vec::new(),
        }]);
        let error = tool.invoke(Duration::from_secs(1)).await.unwrap_err();
        assert!(matches!(error, ToolError::InteractiveNotExecutable));
    }

    #[test]
    fn try_from_submit_plan_stages() {
        let tool = Tool::try_from(call(
            "submit_plan",
            r#"{"stages":[{"title":"data","tasks":["entity","migration"]},{"title":"api","tasks":["controller"]}]}"#,
        ))
        .unwrap();
        assert_eq!(
            tool,
            Tool::SubmitPlan(vec![
                PlanStage {
                    title: "data".into(),
                    tasks: vec!["entity".into(), "migration".into()],
                },
                PlanStage {
                    title: "api".into(),
                    tasks: vec!["controller".into()],
                },
            ])
        );
    }

    #[test]
    fn try_from_submit_plan_empty_stages_falls_back_to_a_stage() {
        let tool = Tool::try_from(call("submit_plan", r#"{"stages":[]}"#)).unwrap();
        assert!(matches!(tool, Tool::SubmitPlan(_)));
    }

    #[test]
    fn try_from_submit_plan_stage_without_tasks_falls_back_to_a_stage() {
        let tool = Tool::try_from(call(
            "submit_plan",
            r#"{"stages":[{"title":"data","tasks":[]}]}"#,
        ))
        .unwrap();
        assert!(matches!(tool, Tool::SubmitPlan(_)));
    }

    #[test]
    fn plan_text_renders_the_stages() {
        let stages = vec![
            PlanStage {
                title: "data".into(),
                tasks: vec!["entity".into(), "migration".into()],
            },
            PlanStage {
                title: "api".into(),
                tasks: vec!["controller".into()],
            },
        ];
        assert_eq!(
            plan_text(&stages),
            "Step 1 of 2: data\n- entity\n- migration\n\nStep 2 of 2: api\n- controller"
        );
    }

    fn serve_once(response: &str) -> (String, std::thread::JoinHandle<()>) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let response = response.to_string();
        let handle = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = [0u8; 4096];
            let _ = stream.read(&mut buf);
            let _ = stream.write_all(response.as_bytes());
        });
        (format!("http://{addr}"), handle)
    }

    #[tokio::test]
    async fn webfetch_returns_the_body() {
        let (url, handle) = serve_once(
            "HTTP/1.1 200 OK\r\nContent-Length: 11\r\nConnection: close\r\n\r\nhello world",
        );

        let tool = Tool::WebFetch(url);

        assert_eq!(tool.invoke(timeout()).await.unwrap(), "hello world");
        handle.join().unwrap();
    }

    #[tokio::test]
    async fn webfetch_error_status_is_error() {
        let (url, handle) = serve_once(
            "HTTP/1.1 404 Not Found\r\nContent-Length: 9\r\nConnection: close\r\n\r\nnot found",
        );

        let tool = Tool::WebFetch(url);

        let error = tool.invoke(timeout()).await.unwrap_err();
        assert!(matches!(
            error,
            ToolError::WebFetchStatus { status: 404, .. }
        ));
        handle.join().unwrap();
    }

    #[tokio::test]
    async fn webfetch_connection_refused_is_error() {
        let tool = Tool::WebFetch("http://127.0.0.1:1".into());

        let error = tool.invoke(timeout()).await.unwrap_err();

        assert!(matches!(error, ToolError::WebFetchFailed { .. }));
    }

    #[tokio::test]
    async fn webfetch_hanging_server_is_timed_out() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let (release, released) = std::sync::mpsc::channel();
        let handle = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = [0u8; 4096];
            let _ = stream.read(&mut buf);
            let _ = released.recv();
        });

        let tool = Tool::WebFetch(format!("http://{addr}"));

        let error = tool.invoke(Duration::from_secs(1)).await.unwrap_err();

        assert!(matches!(
            error,
            ToolError::TimedOut {
                tool: "webfetch",
                seconds: 1
            }
        ));
        let _ = release.send(());
        handle.join().unwrap();
    }

    #[test]
    fn schemas_are_consistent_with_parser() {
        let all = [
            Tool::Bash(String::new()),
            Tool::ReadOnlyBash(String::new()),
            Tool::ReadFile(String::new(), None, None),
            Tool::WriteFile(String::new(), String::new()),
            Tool::EditFile(String::new(), String::new(), String::new()),
            Tool::WebFetch(String::new()),
            Tool::BgRun(String::new()),
        ];
        for definition in tool_definitions(&all) {
            let properties = definition
                .function
                .parameters
                .as_ref()
                .and_then(|p| p.get("properties"))
                .and_then(serde_json::Value::as_object)
                .unwrap();

            let mut document = serde_json::Map::new();
            for (key, value) in properties {
                let sample = match value
                    .get("type")
                    .and_then(serde_json::Value::as_str)
                    .unwrap()
                {
                    "string" => serde_json::json!("x"),
                    "integer" => serde_json::json!(1),
                    other => panic!("unexpected schema type {other}"),
                };
                document.insert(key.clone(), sample);
            }

            let tool_call = call(
                &definition.function.name,
                &serde_json::to_string(&document).unwrap(),
            );
            assert!(
                Tool::try_from(tool_call).is_ok(),
                "schema for {} does not parse",
                definition.function.name
            );
        }
    }

    #[test]
    fn parse_stages_reads_the_stages_array() {
        let stages = parse_stages(
            r#"{"stages":[{"title":"One","tasks":["a","b"]},{"title":"Two","tasks":["c"]}]}"#,
        );
        assert_eq!(stages.len(), 2);
        assert_eq!(stages[0].title, "One");
        assert_eq!(stages[0].tasks, vec!["a".to_string(), "b".to_string()]);
        assert_eq!(stages[1].title, "Two");
    }

    #[test]
    fn parse_stages_accepts_a_task_string_not_just_an_array() {
        let stages = parse_stages(r#"{"stages":[{"title":"One","tasks":"just a string"}]}"#);
        assert_eq!(stages.len(), 1);
        assert_eq!(stages[0].tasks, vec!["just a string".to_string()]);
    }

    #[test]
    fn parse_stages_accepts_bare_string_stages() {
        let stages = parse_stages(r#"{"stages":["first step","second step"]}"#);
        assert_eq!(stages.len(), 2);
        assert_eq!(stages[0].title, "first step");
        assert_eq!(stages[0].tasks, vec!["first step".to_string()]);
    }

    #[test]
    fn parse_stages_falls_back_to_the_plan_field() {
        let stages = parse_stages(r#"{"plan":"the whole plan as text"}"#);
        assert_eq!(stages.len(), 1);
        assert_eq!(stages[0].title, "Plan");
        assert_eq!(stages[0].tasks, vec!["the whole plan as text".to_string()]);
    }

    #[test]
    fn parse_stages_accepts_a_double_encoded_stages_string() {
        let inner = r#"[{"title":"One","tasks":["a","b"]},{"title":"Two","tasks":["c"]}]"#;
        let args = format!("{{\"stages\":{}}}", serde_json::to_string(inner).unwrap());
        let stages = parse_stages(&args);
        assert_eq!(stages.len(), 2);
        assert_eq!(stages[0].title, "One");
        assert_eq!(stages[0].tasks, vec!["a".to_string(), "b".to_string()]);
        assert_eq!(stages[1].title, "Two");
    }

    #[test]
    fn parse_stages_never_returns_the_raw_json() {
        let stages = parse_stages(r#"{"stages":{"weird":"shape"}}"#);
        assert_eq!(stages.len(), 1);
        assert!(
            !stages[0].tasks[0].contains('{'),
            "must not leak raw json: {}",
            stages[0].tasks[0]
        );
    }
}
