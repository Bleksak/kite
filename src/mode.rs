use crate::tool::Tool;

#[derive(Debug, PartialEq, Eq, Clone, Copy, clap::ValueEnum)]
pub enum Mode {
    Yolo,
    Plan,
}

impl Mode {
    pub fn next(self) -> Self {
        match self {
            Self::Yolo => Self::Plan,
            Self::Plan => Self::Yolo,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Yolo => "yolo",
            Self::Plan => "plan",
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
            Self::Plan => vec![
                Tool::ReadFile(String::new(), None, None),
                Tool::Bash(String::new()),
                Tool::SubmitPlan(String::new()),
            ],
        }
    }

    pub fn terminator(self) -> Option<&'static str> {
        match self {
            Self::Yolo => None,
            Self::Plan => Some("submit_plan"),
        }
    }

    pub fn allows(&self, tool: &Tool) -> bool {
        tool_allowed(&self.base_tools(), tool)
    }

    pub fn system_prompt(&self) -> &'static str {
        match self {
            Self::Yolo => "You are a coding agent. Use the tools to accomplish tasks. For long-running commands (tests, builds, dev servers), use bg_run instead of bash; its result is reported automatically when the task finishes. Your configuration and session history live in .kite/: previous sessions are stored as JSON transcripts in .kite/sessions/ and background task logs in .kite/tasks/ — read them when the user refers to previous work. Before quoting or summarizing any file's content, re-read it. Never answer from remembered file content — files may have changed since you last saw them.",
            Self::Plan => "You are a planning agent. Investigate the codebase with read_file and read-only bash commands, then call submit_plan with a concrete, step-by-step implementation plan. Do not modify any files. Call submit_plan alone, without other tools.",
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
    fn plan_schema_contains_only_its_tools() {
        let tools = Mode::Plan.base_tools();
        let definitions = crate::tool::tool_definitions(&tools);
        let names = definitions
            .iter()
            .map(|tool| tool.function.name.as_str())
            .collect::<Vec<_>>();
        assert_eq!(names, vec!["read_file", "bash", "submit_plan"]);
    }

    #[test]
    fn plan_terminator_is_submit_plan_and_yolo_has_none() {
        assert_eq!(Mode::Plan.terminator(), Some("submit_plan"));
        assert_eq!(Mode::Yolo.terminator(), None);
    }

    #[test]
    fn plan_allows_its_tools_and_rejects_the_rest() {
        assert!(Mode::Plan.allows(&Tool::ReadFile("a".into(), None, None)));
        assert!(Mode::Plan.allows(&Tool::Bash("ls".into())));
        assert!(Mode::Plan.allows(&Tool::SubmitPlan("plan".into())));
        assert!(!Mode::Plan.allows(&Tool::WriteFile("a".into(), "x".into())));
        assert!(!Mode::Plan.allows(&Tool::EditFile("a".into(), "x".into(), "y".into())));
        assert!(!Mode::Plan.allows(&Tool::BgRun("sleep 1".into())));
    }

    #[test]
    fn yolo_rejects_submit_plan() {
        assert!(!Mode::Yolo.allows(&Tool::SubmitPlan("plan".into())));
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
