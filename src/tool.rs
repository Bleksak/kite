use std::process::Stdio;
use std::time::Duration;

use tokio::io::AsyncReadExt;
use tokio::process::Command;

use openai_oxide::types::chat::{
    FunctionDef, Tool as OpenAITool, ToolCall as OpenAIToolCall,
};
use serde::Deserialize;
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
    TimedOut {
        tool: &'static str,
        seconds: u64,
    },

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

#[derive(Debug, PartialEq)]
pub enum Tool {
    Bash(String),
    ReadFile(String, Option<usize>, Option<usize>),
    WriteFile(String, String),
    EditFile(String, String, String),
}

pub enum ToolOutput {
    Before(String),
    After,
    Hidden,
}

const OUTPUT_LIMIT: usize = 10_000;

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

impl Tool {
    pub fn label(&self) -> &'static str {
        match &self {
            Tool::Bash(_) => "bash",
            Tool::ReadFile(..) => "read_file",
            Tool::WriteFile(_, _) => "write_file",
            Tool::EditFile(_, _, _) => "edit_file",
        }
    }

    pub fn header(&self) -> String {
        match self {
            Tool::Bash(_) => "bash".to_string(),
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
        }
    }

    pub fn output(&self) -> ToolOutput {
        match self {
            Tool::Bash(cmd) => ToolOutput::Before(cmd.clone()),
            Tool::ReadFile(..) => ToolOutput::After,
            Tool::WriteFile(_, content) => ToolOutput::Before(content.clone()),
            Tool::EditFile(..) => ToolOutput::Hidden,
        }
    }

    pub fn description(&self) -> &'static str {
        match &self {
            Tool::Bash(_) => "Run a bash script",
            Tool::ReadFile(_, _, _) => {
                "Read a file, optionally a line range (1-based start and end line, both inclusive)"
            }
            Tool::WriteFile(_, _) => "Write a file",
            Tool::EditFile(_, _, _) => "Edit a file",
        }
    }

    pub async fn invoke(&self, timeout: Duration) -> Result<String, ToolError> {
        match &self {
            Tool::Bash(script) => {
                let mut child = Command::new("bash")
                    .arg("-c")
                    .arg(script)
                    .stdin(Stdio::null())
                    .stdout(Stdio::piped())
                    .stderr(Stdio::piped())
                    .spawn()
                    .map_err(ToolError::Io)?;

                let collected = tokio::time::timeout(timeout, async {
                    let mut stdout = Vec::new();
                    let mut stderr = Vec::new();
                    if let Some(mut out) = child.stdout.take() {
                        out.read_to_end(&mut stdout).await.map_err(ToolError::Io)?;
                    }
                    if let Some(mut err) = child.stderr.take() {
                        err.read_to_end(&mut stderr).await.map_err(ToolError::Io)?;
                    }
                    let status = child.wait().await.map_err(ToolError::Io)?;
                    Ok::<(std::process::ExitStatus, Vec<u8>, Vec<u8>), ToolError>((
                        status, stdout, stderr,
                    ))
                })
                .await;

                let (status, stdout, stderr) = match collected {
                    Ok(Ok(value)) => value,
                    Ok(Err(error)) => return Err(error),
                    Err(_) => {
                        let _ = child.kill().await;
                        return Err(ToolError::TimedOut {
                            tool: self.label(),
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
                        tool: self.label(),
                        status: status.code().unwrap_or(-1),
                        output: cap_output(combined),
                    });
                }

                Ok(cap_output(
                    String::from_utf8_lossy(&stdout).into_owned(),
                ))
            }
            Tool::ReadFile(file, start, end) => {
                let contents = std::fs::read_to_string(file).map_err(ToolError::Io)?;
                let lines: Vec<&str> = contents.lines().collect();
                let start = start.unwrap_or(1).saturating_sub(1).min(lines.len());
                let end = end
                    .map_or(lines.len(), |end| end.min(lines.len()))
                    .max(start);
                Ok(lines[start..end].join("\n"))
            }
            Tool::WriteFile(file, content) => {
                std::fs::write(file, content).map_err(ToolError::Io)?;
                Ok(String::new())
            }
            Tool::EditFile(file, old_content, new_content) => {
                let mut contents = std::fs::read_to_string(file).map_err(ToolError::Io)?;
                let occurrences = contents.matches(old_content).count();
                if occurrences != 1 {
                    return Err(ToolError::AmbiguousEdit {
                        path: file.clone(),
                        occurrences,
                    });
                }

                let old_index = contents.find(old_content).unwrap();
                contents.replace_range(old_index..old_index + old_content.len(), new_content);

                std::fs::write(file, contents).map_err(ToolError::Io)?;
                Ok(String::new())
            }
        }
    }
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
            "read_file" => parse_args::<ReadFileArgs>(&function.name, &function.arguments)
                .map(|a| Tool::ReadFile(a.path, a.start, a.end)),
            "write_file" => parse_args::<WriteFileArgs>(&function.name, &function.arguments)
                .map(|a| Tool::WriteFile(a.path, a.content)),
            "edit_file" => parse_args::<EditFileArgs>(&function.name, &function.arguments)
                .map(|a| Tool::EditFile(a.path, a.old_content, a.new_content)),
            other => Err(ToolError::UnknownTool {
                name: other.to_string(),
            }),
        }
    }
}

pub fn tool_definitions() -> Vec<OpenAITool> {
    let tools = [
        Tool::Bash(String::new()),
        Tool::ReadFile(String::new(), None, None),
        Tool::WriteFile(String::new(), String::new()),
        Tool::EditFile(String::new(), String::new(), String::new()),
    ];

    tools
        .map(|tool| OpenAITool {
            type_: "function".to_string(),
            function: FunctionDef {
                name: tool.label().to_string(),
                description: Some(tool.description().to_string()),
                parameters: Some(parameters(&tool)),
                strict: None,
            },
        })
        .to_vec()
}

fn parameters(tool: &Tool) -> serde_json::Value {
    match tool {
        Tool::Bash(_) => json!({
            "type": "object",
            "properties": {
                "command": { "type": "string", "description": "the shell command to run" }
            },
            "required": ["command"]
        }),
        Tool::ReadFile(..) => json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "path of the file to read" },
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
    }
}

#[cfg(test)]
mod test {
    use crate::tool::{tool_definitions, Tool, ToolError};
    use openai_oxide::types::chat::{FunctionCall, ToolCall as OpenAIToolCall};
    use std::time::Duration;
    use test_files::TestFiles;

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

        assert_eq!(result, "");
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

        assert_eq!(result, "");
        assert!(file.exists());
        assert_eq!(std::fs::read_to_string(file).unwrap(), "testsion: 3");
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

        let error = tool
            .invoke(Duration::from_secs(1))
            .await
            .unwrap_err();

        assert!(matches!(
            error,
            ToolError::TimedOut { tool: "bash", seconds: 1 }
        ));
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
    fn schemas_are_consistent_with_parser() {
        for definition in tool_definitions() {
            let properties = definition
                .function
                .parameters
                .as_ref()
                .and_then(|p| p.get("properties"))
                .and_then(serde_json::Value::as_object)
                .unwrap();

            let mut document = serde_json::Map::new();
            for (key, value) in properties {
                let sample = match value.get("type").and_then(serde_json::Value::as_str).unwrap() {
                    "string" => serde_json::json!("x"),
                    "integer" => serde_json::json!(1),
                    other => panic!("unexpected schema type {other}"),
                };
                document.insert(key.clone(), sample);
            }

            let tool_call = call(&definition.function.name, &serde_json::to_string(&document).unwrap());
            assert!(Tool::try_from(tool_call).is_ok(), "schema for {} does not parse", definition.function.name);
        }
    }
}
