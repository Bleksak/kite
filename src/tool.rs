use std::process::Command;

#[derive(Debug, thiserror::Error)]
pub enum ToolError {
    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error("{tool} exited with {status}\n{stderr}")]
    NonZeroExit {
        tool: &'static str,
        status: i32,
        stderr: String,
    },

    #[error("expected exactly one occurrence of old content in {path}, found {occurrences}")]
    AmbiguousEdit {
        path: String,
        occurrences: usize,
    },
}

#[derive(Debug)]
pub enum Tool {
    Bash(String),
    ReadFile(String, Option<usize>, Option<usize>),
    WriteFile(String, String),
    EditFile(String, String, String),
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

    pub fn description(&self) -> &'static str {
        match &self {
            Tool::Bash(_) => "Run a bash script",
            Tool::ReadFile(_, _, _) => "Read a file, optionally a line range (1-based start and end line, both inclusive)",
            Tool::WriteFile(_, _) => "Write a file",
            Tool::EditFile(_, _, _) => "Edit a file",
        }
    }

    pub fn invoke(&self) -> Result<String, ToolError> {
        match &self {
            Tool::Bash(script) => {
                let output = Command::new("bash")
                    .arg("-c")
                    .arg(script)
                    .output()
                    .map_err(ToolError::Io)?;

                if !output.status.success() {
                    return Err(ToolError::NonZeroExit {
                        tool: self.label(),
                        status: output.status.code().unwrap_or(-1),
                        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
                    });
                }

                Ok(String::from_utf8_lossy_owned(output.stdout))
            }
            Tool::ReadFile(file, start, end) => {
                let contents = std::fs::read_to_string(file).map_err(ToolError::Io)?;
                let lines: Vec<&str> = contents.lines().collect();
                let start = start.unwrap_or(1).saturating_sub(1).min(lines.len());
                let end = end.map_or(lines.len(), |end| end.min(lines.len())).max(start);
                Ok(lines[start..end].join("\n"))
            },
            Tool::WriteFile(file, content) => {
                std::fs::write(file, content).map_err(ToolError::Io)?;
                Ok(String::new())
            },
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
            },
        }
    }
}

#[cfg(test)]
mod test {
    use crate::tool::{Tool, ToolError};
    use test_files::TestFiles;

    #[test]
    fn bash_hello_world() {
        let script = r#"
            echo "hello world"
        "#;

        let tool = Tool::Bash(script.into());

        let result = tool.invoke().unwrap();

        assert_eq!(result, "hello world\n");
    }

    #[test]
    fn read_file() {
        let temp_dir = TestFiles::new();
        temp_dir.file("hello.txt", "hello world");
        let file = temp_dir.path().join("hello.txt");

        assert!(file.exists());

        let tool = Tool::ReadFile(file.to_string_lossy().into(), None, None);

        let result = tool.invoke().unwrap();

        assert_eq!(result, "hello world");
    }

    #[test]
    fn read_file_full_multiline_strips_trailing_newline() {
        let temp_dir = TestFiles::new();
        temp_dir.file("a.txt", "one\ntwo\nthree\n");
        let file = temp_dir.path().join("a.txt");

        let tool = Tool::ReadFile(file.to_string_lossy().into(), None, None);

        assert_eq!(tool.invoke().unwrap(), "one\ntwo\nthree");
    }

    #[test]
    fn read_file_start_line_is_one_based() {
        let temp_dir = TestFiles::new();
        temp_dir.file("a.txt", "one\ntwo\nthree\n");
        let file = temp_dir.path().join("a.txt");

        let tool = Tool::ReadFile(file.to_string_lossy().into(), Some(2), None);

        assert_eq!(tool.invoke().unwrap(), "two\nthree");
    }

    #[test]
    fn read_file_start_and_end_are_inclusive() {
        let temp_dir = TestFiles::new();
        temp_dir.file("a.txt", "one\ntwo\nthree\n");
        let file = temp_dir.path().join("a.txt");

        let tool = Tool::ReadFile(file.to_string_lossy().into(), Some(2), Some(3));

        assert_eq!(tool.invoke().unwrap(), "two\nthree");
    }

    #[test]
    fn read_file_single_line() {
        let temp_dir = TestFiles::new();
        temp_dir.file("a.txt", "one\ntwo\nthree\n");
        let file = temp_dir.path().join("a.txt");

        let tool = Tool::ReadFile(file.to_string_lossy().into(), Some(1), Some(1));

        assert_eq!(tool.invoke().unwrap(), "one");
    }

    #[test]
    fn read_file_end_beyond_eof_returns_rest() {
        let temp_dir = TestFiles::new();
        temp_dir.file("a.txt", "one\ntwo\nthree\n");
        let file = temp_dir.path().join("a.txt");

        let tool = Tool::ReadFile(file.to_string_lossy().into(), Some(2), Some(10));

        assert_eq!(tool.invoke().unwrap(), "two\nthree");
    }

    #[test]
    fn read_file_start_beyond_eof_is_empty() {
        let temp_dir = TestFiles::new();
        temp_dir.file("a.txt", "one\n");
        let file = temp_dir.path().join("a.txt");

        let tool = Tool::ReadFile(file.to_string_lossy().into(), Some(99), None);

        assert_eq!(tool.invoke().unwrap(), "");
    }

    #[test]
    fn read_file_start_after_end_is_empty() {
        let temp_dir = TestFiles::new();
        temp_dir.file("a.txt", "one\ntwo\nthree\n");
        let file = temp_dir.path().join("a.txt");

        let tool = Tool::ReadFile(file.to_string_lossy().into(), Some(5), Some(2));

        assert_eq!(tool.invoke().unwrap(), "");
    }

    #[test]
    fn write_file() {
        let temp_dir = TestFiles::new();
        let file = temp_dir.path().join("hello.txt");

        assert!(!file.exists());

        let tool = Tool::WriteFile(file.to_string_lossy().into(), "hello world".into());

        let result = tool.invoke().unwrap();

        assert_eq!(result, "");
        assert!(file.exists());
    }

    #[test]
    fn edit_file() {
        let temp_dir = TestFiles::new();
        temp_dir.file("hello.txt", r#"---
version: 3"#);

        let file = temp_dir.path().join("hello.txt");

        assert!(file.exists());

        let tool = Tool::EditFile(file.to_string_lossy().into(), "---\nver".into(), "test".into());

        let result = tool.invoke().unwrap();

        assert_eq!(result, "");
        assert!(file.exists());
        assert_eq!(std::fs::read_to_string(file).unwrap(), "testsion: 3");
    }

    #[test]
    fn edit_file_ambiguous_old_content_is_error() {
        let temp_dir = TestFiles::new();
        temp_dir.file("a.txt", "a b a");
        let file = temp_dir.path().join("a.txt");

        let tool = Tool::EditFile(file.to_string_lossy().into(), "a".into(), "c".into());

        let result = tool.invoke().unwrap_err();

        assert!(matches!(result, ToolError::AmbiguousEdit { occurrences: 2, .. }));
        assert_eq!(std::fs::read_to_string(file).unwrap(), "a b a");
    }

    #[test]
    fn edit_file_missing_old_content_is_error() {
        let temp_dir = TestFiles::new();
        temp_dir.file("a.txt", "hello world");
        let file = temp_dir.path().join("a.txt");

        let tool = Tool::EditFile(file.to_string_lossy().into(), "nope".into(), "x".into());

        let result = tool.invoke().unwrap_err();

        assert!(matches!(result, ToolError::AmbiguousEdit { occurrences: 0, .. }));
        assert_eq!(std::fs::read_to_string(file).unwrap(), "hello world");
    }

    #[test]
    fn bash_nonzero_exit_is_error() {
        let tool = Tool::Bash("exit 3".into());

        let result = tool.invoke().unwrap_err();

        assert!(matches!(result, ToolError::NonZeroExit { status: 3, .. }));
    }

    #[test]
    fn bash_stderr_is_collected_in_error() {
        let tool = Tool::Bash(r#"echo "oops" 1>&2; exit 1"#.into());

        let error = tool.invoke().unwrap_err();
        let message = error.to_string();

        assert!(message.contains("bash exited with 1"));
        assert!(message.contains("oops"));
    }

    #[test]
    fn bash_missing_command_is_error_127() {
        let error = Tool::Bash("definitely-not-a-command".into()).invoke().unwrap_err();

        let ToolError::NonZeroExit { status, stderr, .. } = error else {
            panic!("expected NonZeroExit, got {error:?}");
        };

        assert_eq!(status, 127);
        assert!(stderr.contains("not found"));
    }

    #[test]
    fn bash_empty_output_is_ok() {
        let tool = Tool::Bash("true".into());

        assert_eq!(tool.invoke().unwrap(), "");
    }

    #[test]
    fn bash_stderr_on_success_is_ignored() {
        let tool = Tool::Bash(r#"echo "out"; echo "warn" 1>&2"#.into());

        assert_eq!(tool.invoke().unwrap(), "out\n");
    }

    #[test]
    fn bash_stdout_non_ascii_roundtrip() {
        let tool = Tool::Bash("printf 'héllo 🚀'".into());

        assert_eq!(tool.invoke().unwrap(), "héllo 🚀");
    }

    #[test]
    fn read_file_missing_is_error() {
        let tool = Tool::ReadFile("/nonexistent/nope.txt".into(), None, None);

        let error = tool.invoke().unwrap_err();

        assert!(matches!(
            error,
            ToolError::Io(e) if e.kind() == std::io::ErrorKind::NotFound
        ));
    }

    #[test]
    fn read_file_non_utf8_is_error() {
        let temp_dir = TestFiles::new();
        let file = temp_dir.path().join("binary.bin");
        std::fs::write(&file, [0xff, 0xfe, 0x00]).unwrap();

        let tool = Tool::ReadFile(file.to_string_lossy().into(), None, None);

        let error = tool.invoke().unwrap_err();

        assert!(matches!(
            error,
            ToolError::Io(e) if e.kind() == std::io::ErrorKind::InvalidData
        ));
    }

    #[test]
    fn write_file_overwrites_existing() {
        let temp_dir = TestFiles::new();
        temp_dir.file("a.txt", "old");
        let file = temp_dir.path().join("a.txt");

        let tool = Tool::WriteFile(file.to_string_lossy().into(), "new".into());

        tool.invoke().unwrap();
        assert_eq!(std::fs::read_to_string(file).unwrap(), "new");
    }

    #[test]
    fn edit_file_delete_text() {
        let temp_dir = TestFiles::new();
        temp_dir.file("a.txt", "hello world");
        let file = temp_dir.path().join("a.txt");

        let tool = Tool::EditFile(file.to_string_lossy().into(), "hello ".into(), "".into());

        tool.invoke().unwrap();
        assert_eq!(std::fs::read_to_string(file).unwrap(), "world");
    }

    #[test]
    fn edit_file_matches_literally_not_regex() {
        let temp_dir = TestFiles::new();
        temp_dir.file("a.txt", "a.b and aXb");
        let file = temp_dir.path().join("a.txt");

        let tool = Tool::EditFile(file.to_string_lossy().into(), "a.b".into(), "a_b".into());

        tool.invoke().unwrap();
        assert_eq!(std::fs::read_to_string(file).unwrap(), "a_b and aXb");
    }

    #[test]
    fn edit_file_multiline_and_non_ascii() {
        let temp_dir = TestFiles::new();
        temp_dir.file("a.txt", "héllo line1\nline2\nline3");
        let file = temp_dir.path().join("a.txt");

        let tool = Tool::EditFile(file.to_string_lossy().into(), "héllo line1\nline2".into(), "hej X".into());

        tool.invoke().unwrap();
        assert_eq!(std::fs::read_to_string(file).unwrap(), "hej X\nline3");
    }
}
