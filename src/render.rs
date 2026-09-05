use std::io::Write;

use crate::agent::{AgentEvent, AnswerGate, ThinkingMode};

pub struct Renderer<W: Write, E: Write> {
    out_tty: bool,
    err_tty: bool,
    out: W,
    err: E,
    gate: AnswerGate,
    thinking_open: bool,
    output_open: bool,
    output_header: Option<String>,
}

impl<W: Write, E: Write> Renderer<W, E> {
    pub fn new(out_tty: bool, err_tty: bool, out: W, err: E) -> Renderer<W, E> {
        Renderer {
            out_tty,
            err_tty,
            out,
            err,
            gate: AnswerGate::new(),
            thinking_open: false,
            output_open: false,
            output_header: None,
        }
    }

    pub fn on_event(&mut self, event: AgentEvent) {
        match event {
            AgentEvent::CompletionStarted => {
                self.close_thinking();
                if let Some(remaining) = self.gate.finish() {
                    let _ = writeln!(self.out, "{remaining}");
                    let _ = self.out.flush();
                }
                self.gate = AnswerGate::new();
            }
            AgentEvent::Tokens(chunk) => {
                self.close_output();

                let (mode, text) = self.gate.on_chunk(&chunk);

                if mode == ThinkingMode::Live
                    && let Some(text) = &chunk.thinking
                {
                    self.open_thinking();
                    let _ = write!(self.err, "{text}");
                    let _ = self.err.flush();
                }

                if let Some(text) = text {
                    self.close_thinking();
                    let _ = write!(self.out, "{text}");
                    let _ = self.out.flush();
                }
            }
            AgentEvent::ToolStarted { header, body } => {
                self.close_thinking();
                self.close_output();
                self.open_output(&header);
                let _ = writeln!(self.out, "{header}");
                if let Some(body) = body {
                    let _ = writeln!(self.out, "{body}");
                }
                let _ = self.out.flush();
            }
            AgentEvent::ToolResult { header, body } => {
                if self.output_header.as_deref() != Some(header.as_str()) {
                    self.close_output();
                    self.open_output(&header);
                    let _ = writeln!(self.out, "{header}");
                }
                let _ = writeln!(self.out, "{body}");
                self.close_output();
                let _ = self.out.flush();
            }
        }
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
            if self.err_tty { "\x1b[2m<thinking>\n" } else { "<thinking>\n" }
        );
        let _ = self.err.flush();
    }

    fn close_thinking(&mut self) {
        if !self.thinking_open {
            return;
        }
        self.thinking_open = false;
        let _ = write!(
            self.err,
            "{}",
            if self.err_tty { "</thinking>\x1b[0m\n" } else { "</thinking>\n" }
        );
        let _ = self.err.flush();
    }

    fn open_output(&mut self, header: &str) {
        self.output_open = true;
        self.output_header = Some(header.to_string());
        let _ = write!(
            self.out,
            "{}",
            if self.out_tty { "\x1b[2m<output>\n" } else { "<output>\n" }
        );
        let _ = self.out.flush();
    }

    fn close_output(&mut self) {
        if !self.output_open {
            return;
        }
        self.output_open = false;
        self.output_header = None;
        let _ = write!(
            self.out,
            "{}",
            if self.out_tty { "</output>\x1b[0m\n" } else { "</output>\n" }
        );
        let _ = self.out.flush();
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::agent::ChunkTokens;

    fn render(out_tty: bool, err_tty: bool, events: Vec<AgentEvent>) -> (String, String) {
        let mut out = Vec::new();
        let mut err = Vec::new();
        let mut renderer = Renderer::new(out_tty, err_tty, &mut out, &mut err);
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
                AgentEvent::ToolResult {
                    header: "read_file: a.txt".into(),
                    body: "line one".into(),
                },
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
    fn thinking_is_live_again_on_the_next_completion() {
        let (out, err) = render(
            false,
            false,
            vec![
                thinking("first round "),
                text("narration"),
                AgentEvent::CompletionStarted,
                thinking("second round "),
                text("answer"),
            ],
        );

        assert_eq!(
            err,
            "<thinking>\nfirst round </thinking>\n<thinking>\nsecond round </thinking>\n"
        );
        assert_eq!(out, "narrationanswer\n");
    }

    #[test]
    fn buffered_text_flushes_at_the_completion_boundary() {
        let (out, err) = render(
            false,
            false,
            vec![
                AgentEvent::Tokens(ChunkTokens {
                    thinking: Some("glued".into()),
                    text: Some("narration".into()),
                }),
                AgentEvent::CompletionStarted,
                thinking("next round"),
            ],
        );

        assert_eq!(
            err,
            "<thinking>\nglued</thinking>\n<thinking>\nnext round</thinking>\n"
        );
        assert_eq!(out, "narration\n\n");
    }

    #[test]
    fn dim_codes_follow_each_stream_independently() {
        let (out, err) = render(
            false,
            true,
            vec![
                AgentEvent::ToolStarted {
                    header: "bash".into(),
                    body: Some("ls".into()),
                },
                thinking("think"),
                text("done"),
            ],
        );

        assert_eq!(out, "<output>\nbash\nls\n</output>\ndone\n");
        assert_eq!(err, "\x1b[2m<thinking>\nthink</thinking>\x1b[0m\n");
    }

    #[test]
    fn parallel_tool_results_get_their_own_blocks() {
        let (out, err) = render(
            false,
            false,
            vec![
                AgentEvent::ToolStarted {
                    header: "read_file: a.txt".into(),
                    body: None,
                },
                AgentEvent::ToolStarted {
                    header: "read_file: b.txt".into(),
                    body: None,
                },
                AgentEvent::ToolResult {
                    header: "read_file: a.txt".into(),
                    body: "alpha".into(),
                },
                AgentEvent::ToolResult {
                    header: "read_file: b.txt".into(),
                    body: "beta".into(),
                },
            ],
        );

        assert_eq!(
            out,
            "<output>\nread_file: a.txt\n</output>\n<output>\nread_file: b.txt\n</output>\n<output>\nread_file: a.txt\nalpha\n</output>\n<output>\nread_file: b.txt\nbeta\n</output>\n\n"
        );
        assert!(err.is_empty());
    }

    #[test]
    fn tty_wraps_scopes_in_dim() {
        let (out, err) = render(true, true, vec![thinking("think"), text("done")]);

        assert_eq!(err, "\x1b[2m<thinking>\nthink</thinking>\x1b[0m\n");
        assert_eq!(out, "done\n");
    }
}
