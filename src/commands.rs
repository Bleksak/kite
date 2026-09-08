#[derive(Debug, Clone, PartialEq)]
pub enum Command {
    Help,
}

impl Command {
    pub fn parse(input: &str) -> Option<Command> {
        let token = input.split_whitespace().next()?;

        match token {
            "/help" => Some(Command::Help),
            _ => None,
        }
    }

    pub fn output(&self) -> String {
        match self {
            Command::Help => help_text(),
        }
    }
}

fn help_text() -> String {
    "**keybindings**\n\n\
     - C-s — open/close the session picker\n\
     - C-q — open/close the background tasks overlay\n\
     - C-t — cycle the thinking level\n\
     - Tab — switch the mode (yolo ↔ plan)\n\
     - C-p — open/close the plan popup\n\
     - C-g — open/close the review popup\n\
     - C-esc — cancel the running turn\n\
     - C-c — quit\n\n\
     **commands**\n\n\
     - /help — show this message"
        .to_string()
}

#[cfg(test)]
mod test {
    use super::Command;

    #[test]
    fn parse_recognizes_help() {
        assert_eq!(Command::parse("/help"), Some(Command::Help));
        assert_eq!(Command::parse("/help please"), Some(Command::Help));
    }

    #[test]
    fn parse_ignores_plain_prompts_paths_and_unknown_commands() {
        assert_eq!(Command::parse("hello"), None);
        assert_eq!(Command::parse(""), None);
        assert_eq!(Command::parse("/"), None);
        assert_eq!(Command::parse("/asdf"), None);
        assert_eq!(Command::parse("/asdf bar"), None);
        assert_eq!(Command::parse("/home/user/file.rs"), None);
        assert_eq!(Command::parse("fix /home/user/file.rs"), None);
        assert_eq!(Command::parse("fix /help"), None);
    }

    #[test]
    fn help_output_lists_the_keybindings() {
        let out = Command::Help.output();
        assert!(out.contains("C-s"));
        assert!(out.contains("C-esc"));
        assert!(out.contains("/help"));
    }
}
