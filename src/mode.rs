use crate::tool::Tool;

#[derive(Debug, PartialEq, Eq, Clone, Copy, clap::ValueEnum)]
pub enum Mode {
    Yolo,
    Plan,
    Implement,
}

impl Mode {
    pub fn next(self) -> Self {
        match self {
            Self::Yolo => Self::Plan,
            Self::Plan => Self::Yolo,
            Self::Implement => Self::Yolo,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Yolo => "yolo",
            Self::Plan => "plan",
            Self::Implement => "implement",
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
                Tool::AskUser(Vec::new()),
            ],
            Self::Plan => vec![
                Tool::ReadFile(String::new(), None, None),
                Tool::ReadOnlyBash(String::new()),
                Tool::AskUser(Vec::new()),
                Tool::SubmitPlan(Vec::new()),
            ],
            Self::Implement => vec![
                Tool::ReadFile(String::new(), None, None),
                Tool::ReadOnlyBash(String::new()),
                Tool::Bash(String::new()),
                Tool::WriteFile(String::new(), String::new()),
                Tool::EditFile(String::new(), String::new(), String::new()),
                Tool::WebFetch(String::new()),
                Tool::BgRun(String::new()),
                Tool::AskUser(Vec::new()),
                Tool::Escalate(String::new()),
            ],
        }
    }

    pub fn terminator(self) -> Option<&'static str> {
        match self {
            Self::Yolo => None,
            Self::Plan => Some("submit_plan"),
            Self::Implement => Some("escalate"),
        }
    }

    pub fn allows(&self, tool: &Tool) -> bool {
        tool_allowed(&self.base_tools(), tool)
    }

    pub fn system_prompt(&self) -> &'static str {
        match self {
            Self::Yolo => {
                "You are a coding agent. Do the minimum work the task requires: answer directly when you can, and only read files or run commands when the task needs them — do not explore the codebase on your own. For long-running commands (tests, builds, dev servers), use bg_run instead of bash; its result is reported automatically when the task finishes. When you need a decision, preference, or information only the user can provide, call ask_user and wait for the answer — do not guess. Your configuration and session history live in .kite/: previous sessions are stored as JSON transcripts in .kite/sessions/ and background task logs in .kite/tasks/ — read them when the user refers to previous work. If you rely on file content you saw earlier in this session, re-read the file before quoting or summarizing it — files may have changed. When a user message contains @path/to/file, the file's content is included in the message; a pruned placeholder means the content was removed after use — re-read the file with read_file."
            }
            Self::Plan => {
                "You are a planning agent. Investigate only what the task requires: read the files you need to understand the change, not the whole codebase, and do not re-read a file you have already read. As soon as you have enough to plan, call submit_plan with a concrete implementation plan split into stages — do not keep investigating. Each stage is a self-contained set of tasks that can be implemented and verified on its own; the user reviews each stage's implementation before the next stage starts, so order the stages from foundation to finish. Do not modify any files."
            }
            Self::Implement => {
                "You are an implementation agent. Execute the current stage step by step. Verify your work (build, tests) with bash or bg_run. Do not start later stages; the user reviews this stage before the next one begins. If you hit a blocker you cannot resolve, call escalate alone with a description of the blocker; do not guess around it."
            }
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

    #[test]
    fn plan_schema_contains_only_its_tools() {
        let tools = Mode::Plan.base_tools();
        let definitions = crate::tool::tool_definitions(&tools);
        let names = definitions
            .iter()
            .map(|tool| tool.function.name.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            names,
            vec!["read_file", "readonly_bash", "ask_user", "submit_plan"]
        );
    }

    #[test]
    fn plan_terminator_is_submit_plan_and_yolo_has_none() {
        assert_eq!(Mode::Plan.terminator(), Some("submit_plan"));
        assert_eq!(Mode::Yolo.terminator(), None);
    }

    #[test]
    fn plan_allows_its_tools_and_rejects_the_rest() {
        assert!(Mode::Plan.allows(&Tool::ReadFile("a".into(), None, None)));
        assert!(Mode::Plan.allows(&Tool::ReadOnlyBash("ls".into())));
        assert!(Mode::Plan.allows(&Tool::SubmitPlan(Vec::new())));
        assert!(!Mode::Plan.allows(&Tool::Bash("ls".into())));
        assert!(!Mode::Plan.allows(&Tool::WriteFile("a".into(), "x".into())));
        assert!(!Mode::Plan.allows(&Tool::EditFile("a".into(), "x".into(), "y".into())));
        assert!(!Mode::Plan.allows(&Tool::BgRun("sleep 1".into())));
    }

    #[test]
    fn implement_schema_contains_its_tools() {
        let tools = Mode::Implement.base_tools();
        let definitions = crate::tool::tool_definitions(&tools);
        let names = definitions
            .iter()
            .map(|tool| tool.function.name.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            names,
            vec![
                "read_file",
                "readonly_bash",
                "bash",
                "write_file",
                "edit_file",
                "webfetch",
                "bg_run",
                "ask_user",
                "escalate"
            ]
        );
    }

    #[test]
    fn implement_terminator_is_escalate_and_rejects_submit_plan() {
        assert_eq!(Mode::Implement.terminator(), Some("escalate"));
        assert!(!Mode::Implement.allows(&Tool::SubmitPlan(Vec::new())));
        assert!(Mode::Implement.allows(&Tool::Escalate("blocker".into())));
        assert!(Mode::Implement.allows(&Tool::WriteFile("a".into(), "x".into())));
    }

    #[test]
    fn tab_cycles_yolo_and_plan_only() {
        assert_eq!(Mode::Yolo.next(), Mode::Plan);
        assert_eq!(Mode::Plan.next(), Mode::Yolo);
    }

    #[test]
    fn yolo_rejects_submit_plan() {
        assert!(!Mode::Yolo.allows(&Tool::SubmitPlan(Vec::new())));
    }

    #[test]
    fn yolo_allows_its_own_tools() {
        assert!(Mode::Yolo.allows(&Tool::Bash("ls".into())));
        assert!(Mode::Yolo.allows(&Tool::BgRun("sleep 1".into())));
    }

    #[test]
    fn a_tool_outside_the_mode_set_is_rejected() {
        let restricted = vec![Tool::ReadFile(String::new(), None, None)];
        assert!(tool_allowed(
            &restricted,
            &Tool::ReadFile("a".into(), None, None)
        ));
        assert!(!tool_allowed(&restricted, &Tool::Bash("rm -rf /".into())));
        assert!(!tool_allowed(
            &restricted,
            &Tool::WriteFile("a".into(), "x".into())
        ));
    }
}
