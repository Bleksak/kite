use std::io::Write;

use crate::agent::{AgentEvent, AnswerGate, ThinkingMode};

pub struct Renderer<W: Write, E: Write> {
    tty: bool,
    out: W,
    err: E,
    gate: AnswerGate,
    thinking_open: bool,
    output_open: bool,
}

impl<W: Write, E: Write> Renderer<W, E> {
    pub fn new(tty: bool, out: W, err: E) -> Renderer<W, E> {
        Renderer {
            tty,
            out,
            err,
            gate: AnswerGate::new(),
            thinking_open: false,
            output_open: false,
        }
    }

    pub fn on_event(&mut self, event: AgentEvent) {
        match event {
            AgentEvent::Tokens(chunk) => {
                self.close_output();

                let (mode, text) = self.gate.on_chunk(&chunk);

                if mode == ThinkingMode::Live
                    && let Some(text) = &chunk.thinking
                {
                    self.open_thinking();
                    let _ = write!(self.err, "{text}");
                }

                if let Some(text) = text {
                    self.close_thinking();
                    let _ = write!(self.out, "{text}");
                }
            }
            AgentEvent::ToolStarted { header, body } => {
                self.close_thinking();
                self.close_output();
                self.open_output();
                let _ = writeln!(self.out, "{header}");
                if let Some(body) = body {
                    let _ = writeln!(self.out, "{body}");
                }
            }
            AgentEvent::ToolResult(body) => {
                let _ = writeln!(self.out, "{body}");
                self.close_output();
            }
        }

        let _ = self.out.flush();
        let _ = self.err.flush();
    }

    pub fn finish(&mut self) {
        self.close_thinking();
        self.close_output();
        if let Some(remaining) = self.gate.finish() {
            let _ = writeln!(self.out, "{remaining}");
        }
        let _ = writeln!(self.out);
        let _ = self.out.flush();
    }

    fn open_thinking(&mut self) {
        if self.thinking_open {
            return;
        }
        self.thinking_open = true;
        let _ = write!(
            self.err,
            "{}",
            if self.tty { "\x1b[2m<thinking>\n" } else { "<thinking>\n" }
        );
    }

    fn close_thinking(&mut self) {
        if !self.thinking_open {
            return;
        }
        self.thinking_open = false;
        let _ = write!(
            self.err,
            "{}",
            if self.tty { "\x1b[0m</thinking>\n" } else { "</thinking>\n" }
        );
    }

    fn open_output(&mut self) {
        self.output_open = true;
        let _ = write!(
            self.out,
            "{}",
            if self.tty { "\x1b[2m<output>\n" } else { "<output>\n" }
        );
    }

    fn close_output(&mut self) {
        if !self.output_open {
            return;
        }
        self.output_open = false;
        let _ = write!(
            self.out,
            "{}",
            if self.tty { "\x1b[0m</output>\n" } else { "</output>\n" }
        );
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::agent::ChunkTokens;

    fn render(tty: bool, events: Vec<AgentEvent>) -> (String, String) {
        let mut out = Vec::new();
        let mut err = Vec::new();
        let mut renderer = Renderer::new(tty, &mut out, &mut err);
        for event in events {
            renderer.on_event(event);
        }
        renderer.finish();
        (
            String::from_utf8(out).unwrap(),
            String::from_utf8(err).unwrap(),
        )
    }

    fn thinking(text: &str) -> AgentEvent {
        AgentEvent::Tokens(ChunkTokens {
            thinking: Some(text.into()),
            text: None,
        })
    }

    fn text(content: &str) -> AgentEvent {
        AgentEvent::Tokens(ChunkTokens {
            thinking: None,
            text: Some(content.into()),
        })
    }

    #[test]
    fn thinking_goes_to_err_between_tags() {
        let (out, err) = render(
            false,
            vec![thinking("Let me think. "), text("42")],
        );

        assert_eq!(err, "<thinking>\nLet me think. </thinking>\n");
        assert_eq!(out, "42\n");
    }

    #[test]
    fn output_block_closes_when_the_next_tool_starts() {
        let (out, err) = render(
            false,
            vec![
                AgentEvent::ToolStarted {
                    header: "bash".into(),
                    body: Some("echo out".into()),
                },
                AgentEvent::ToolStarted {
                    header: "read_file: a.txt".into(),
                    body: None,
                },
                AgentEvent::ToolResult("line one".into()),
            ],
        );

        assert_eq!(
            out,
            "<output>\nbash\necho out\n</output>\n<output>\nread_file: a.txt\nline one\n</output>\n\n"
        );
        assert!(err.is_empty());
    }

    #[test]
    fn tool_event_closes_an_open_thinking_run() {
        let (out, err) = render(
            false,
            vec![
                thinking("thinking"),
                AgentEvent::ToolStarted {
                    header: "bash".into(),
                    body: Some("ls".into()),
                },
            ],
        );

        assert_eq!(err, "<thinking>\nthinking</thinking>\n");
        assert_eq!(out, "<output>\nbash\nls\n</output>\n\n");
    }

    #[test]
    fn straggler_thinking_after_answer_is_hidden() {
        let (out, err) = render(
            false,
            vec![
                thinking("reasoning "),
                text("answer"),
                thinking("straggler"),
            ],
        );

        assert_eq!(err, "<thinking>\nreasoning </thinking>\n");
        assert_eq!(out, "answer\n");
    }

    #[test]
    fn output_block_closes_when_thinking_resumes() {
        let (out, err) = render(
            false,
            vec![
                AgentEvent::ToolStarted {
                    header: "write_file: a.txt".into(),
                    body: Some("content".into()),
                },
                thinking("next turn thinking"),
                text("answer"),
            ],
        );

        assert_eq!(out, "<output>\nwrite_file: a.txt\ncontent\n</output>\nanswer\n");
        assert_eq!(err, "<thinking>\nnext turn thinking</thinking>\n");
    }

    #[test]
    fn tty_wraps_scopes_in_dim() {
        let (out, err) = render(true, vec![thinking("think"), text("done")]);

        assert_eq!(err, "\x1b[2m<thinking>\nthink\x1b[0m</thinking>\n");
        assert_eq!(out, "done\n");
    }
}
