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
    Python(String),
    Bash(String),
    ReadFile(String),
    WriteFile(String, String),
    EditFile(String, String, String),
}

impl Tool {
    pub fn label(&self) -> &'static str {
        match &self {
            Tool::Python(_) => "python",
            Tool::Bash(_) => "bash",
            Tool::ReadFile(_) => "read_file",
            Tool::WriteFile(_, _) => "write_file",
            Tool::EditFile(_, _, _) => "edit_file",
        }
    }

    pub fn description(&self) -> &'static str {
        match &self {
            Tool::Python(_) => "Run a python script",
            Tool::Bash(_) => "Run a bash script",
            Tool::ReadFile(_) => "Read a file",
            Tool::WriteFile(_, _) => "Write a file",
            Tool::EditFile(_, _, _) => "Edit a file",
        }
    }

    pub fn invoke(&self) -> Result<String, ToolError> {
        match &self {
            Tool::Python(file) => {
                let command = Command::new("python3")
                    .arg(file)
                    .output()
                    .map_err(ToolError::Io)?;

                Ok(String::from_utf8_lossy_owned(command.stdout))
            }
            Tool::Bash(script) => {
                let command = Command::new("bash")
                    .arg("-c")
                    .arg(script)
                    .output()
                    .map_err(ToolError::Io)?;

                Ok(String::from_utf8_lossy_owned(command.stdout))
            },
            Tool::ReadFile(file) => {
                let contents = std::fs::read_to_string(file).map_err(ToolError::Io)?;
                Ok(contents)
            },
            Tool::WriteFile(file, content) => {
                std::fs::write(file, content).map_err(ToolError::Io)?;
                Ok(String::new())
            },
            Tool::EditFile(file, old_content, new_content) => {
                let mut contents = std::fs::read_to_string(file).map_err(ToolError::Io)?;
                
                // require exactly one occurrence of the old content
                let occurrences = contents.matches(old_content).count();
                if occurrences != 1 {
                    return Err(ToolError::AmbiguousEdit {
                        path: file.clone(),
                        occurrences,
                    });
                }

                // replace the (unique) old content with the new content
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
    fn python_hello_world() {
        let temp_dir = TestFiles::new();
        temp_dir.file("hello.py", "print('hello world')");
        let file = temp_dir.path().join("hello.py");

        assert!(file.exists());

        let tool = Tool::Python(file.to_string_lossy().into());

        let result = tool.invoke().unwrap();

        assert_eq!(result, "hello world\n");
    }

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

        let tool = Tool::ReadFile(file.to_string_lossy().into());

        let result = tool.invoke().unwrap();

        assert_eq!(result, "hello world");
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
}
