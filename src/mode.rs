use crate::tool::Tool;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Mode {
    Yolo,
}

impl Mode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Yolo => "yolo",
        }
    }

    pub fn base_tools(&self) -> Vec<Tool> {
        match self {
            Self::Yolo => vec![
                Tool::Bash(String::new()),
                Tool::ReadOnlyBash(String::new()),
                Tool::ReadFile(String::new(), None, None),
                Tool::WriteFile(String::new(), String::new()),
                Tool::EditFile(String::new(), String::new(), String::new()),
                Tool::WebFetch(String::new()),
                Tool::BgRun(String::new()),
            ],
        }
    }

    pub fn allows(&self, tool: &Tool) -> bool {
        tool_allowed(&self.base_tools(), tool)
    }

    pub fn system_prompt(&self) -> &'static str {
        match self {
            Self::Yolo => "You are a coding agent. Use the tools to accomplish tasks. For long-running commands (tests, builds, dev servers), use bg_run instead of bash; its result is reported automatically when the task finishes. Your configuration and session history live in .kite/: previous sessions are stored as JSON transcripts in .kite/sessions/ and background task logs in .kite/tasks/ — read them when the user refers to previous work. Before quoting or summarizing any file's content, re-read it. Never answer from remembered file content — files may have changed since you last saw them.",
        }
    }
}

pub fn tool_allowed(tools: &[Tool], tool: &Tool) -> bool {
    tools.iter().any(|allowed| allowed.label() == tool.label())
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn yolo_schema_contains_every_tool() {
        let tools = Mode::Yolo.base_tools();
        let definitions = crate::tool::tool_definitions(&tools);
        let names = definitions
            .iter()
            .map(|tool| tool.function.name.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            names,
            vec![
                "bash", "readonly_bash", "read_file", "write_file", "edit_file", "webfetch", "bg_run"
            ]
        );
    }

    #[test]
    fn yolo_allows_its_own_tools() {
        assert!(Mode::Yolo.allows(&Tool::Bash("ls".into())));
        assert!(Mode::Yolo.allows(&Tool::BgRun("sleep 1".into())));
    }

    #[test]
    fn a_tool_outside_the_mode_set_is_rejected() {
        let restricted = vec![Tool::ReadFile(String::new(), None, None)];
        assert!(tool_allowed(&restricted, &Tool::ReadFile("a".into(), None, None)));
        assert!(!tool_allowed(&restricted, &Tool::Bash("rm -rf /".into())));
        assert!(!tool_allowed(&restricted, &Tool::WriteFile("a".into(), "x".into())));
    }
}
