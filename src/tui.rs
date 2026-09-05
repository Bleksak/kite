use std::mem;

use crossterm::cursor;
use crossterm::event::{self, Event as TermEvent, KeyCode, KeyModifiers};
use crossterm::terminal;
use crossterm::ExecutableCommand;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use ratatui::{Frame, Terminal};

use crate::agent::{Agent, AgentEvent, AnswerGate, ThinkingMode};

pub enum TuiEvent {
    Agent(AgentEvent),
    TurnDone { prompt: u64, completion: u64 },
    TurnError(String),
}

pub struct TuiRenderer {
    gate: AnswerGate,
    scrollback: Vec<Line<'static>>,
    thinking: String,
    tool_header: Option<String>,
}

impl TuiRenderer {
    pub fn new() -> TuiRenderer {
        TuiRenderer {
            gate: AnswerGate::new(),
            scrollback: Vec::new(),
            thinking: String::new(),
            tool_header: None,
        }
    }

    pub fn scrollback(&self) -> &[Line<'static>] {
        &self.scrollback
    }

    pub fn on_event(&mut self, event: AgentEvent) {
        match event {
            AgentEvent::CompletionStarted => {
                self.close_thinking();
                if let Some(remaining) = self.gate.finish() {
                    self.push_lines(&remaining, Style::default());
                }
                self.gate = AnswerGate::new();
            }
            AgentEvent::Tokens(chunk) => {
                self.tool_header = None;
                let (mode, text) = self.gate.on_chunk(&chunk);
                if mode == ThinkingMode::Live
                    && let Some(thinking) = &chunk.thinking
                {
                    self.thinking.push_str(thinking);
                }
                if let Some(text) = text {
                    self.close_thinking();
                    self.push_lines(&text, Style::default());
                }
            }
            AgentEvent::ToolStarted { header, body } => {
                self.close_thinking();
                self.tool_header = Some(header.clone());
                self.scrollback
                    .push(Line::from(Span::styled(format!("⚙ {header}"), Style::default().bold())));
                if let Some(body) = body {
                    self.push_indented(&body);
                }
            }
            AgentEvent::ToolResult { header, body } => {
                if self.tool_header.as_deref() != Some(header.as_str()) {
                    self.tool_header = Some(header.clone());
                    self.scrollback
                        .push(Line::from(Span::styled(format!("⚙ {header}"), Style::default().bold())));
                }
                self.push_indented(&body);
                self.tool_header = None;
            }
        }
    }

    pub fn finish(&mut self) {
        self.close_thinking();
        self.tool_header = None;
        if let Some(remaining) = self.gate.finish() {
            self.push_lines(&remaining, Style::default());
        }
    }

    fn close_thinking(&mut self) {
        if self.thinking.is_empty() {
            return;
        }
        let thinking = mem::take(&mut self.thinking);
        self.push_lines(&thinking, Style::default().dim());
    }

    fn push_indented(&mut self, body: &str) {
        for line in body.split('\n') {
            if line.is_empty() {
                self.scrollback.push(Line::default());
            } else {
                self.scrollback
                    .push(Line::from(Span::styled(format!("  {line}"), Style::default().dim())));
            }
        }
    }

    fn push_lines(&mut self, text: &str, style: Style) {
        for line in text.split('\n') {
            if line.is_empty() {
                self.scrollback.push(Line::default());
            } else {
                self.scrollback.push(Line::from(Span::styled(line.to_string(), style)));
            }
        }
    }
}

pub struct TuiState {
    pub renderer: TuiRenderer,
    pub scroll: usize,
    pub input: String,
    pub running: bool,
    pub error: Option<String>,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub model: String,
}

impl TuiState {
    pub fn new(model: String) -> TuiState {
        TuiState {
            renderer: TuiRenderer::new(),
            scroll: 0,
            input: String::new(),
            running: false,
            error: None,
            prompt_tokens: 0,
            completion_tokens: 0,
            model,
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum KeyAction {
    None,
    Submit(String),
    Quit,
}

const PAGE: usize = 10;

pub fn handle_key(state: &mut TuiState, event: &TermEvent) -> KeyAction {
    let TermEvent::Key(key) = event else {
        return KeyAction::None;
    };
    if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
        return KeyAction::Quit;
    }
    if state.running {
        return match key.code {
            KeyCode::PageUp => {
                state.scroll = (state.scroll + PAGE).min(state.renderer.scrollback().len());
                KeyAction::None
            }
            KeyCode::PageDown => {
                state.scroll = state.scroll.saturating_sub(PAGE);
                KeyAction::None
            }
            _ => KeyAction::None,
        };
    }
    match key.code {
        KeyCode::Enter => {
            if state.input.is_empty() {
                KeyAction::None
            } else {
                KeyAction::Submit(mem::take(&mut state.input))
            }
        }
        KeyCode::Backspace => {
            state.input.pop();
            KeyAction::None
        }
        KeyCode::PageUp => {
            state.scroll = (state.scroll + PAGE).min(state.renderer.scrollback().len());
            KeyAction::None
        }
        KeyCode::PageDown => {
            state.scroll = state.scroll.saturating_sub(PAGE);
            KeyAction::None
        }
        KeyCode::Char(c) => {
            if c == 'q' && state.input.is_empty() {
                KeyAction::Quit
            } else {
                state.input.push(c);
                KeyAction::None
            }
        }
        _ => KeyAction::None,
    }
}

pub fn draw(frame: &mut Frame, state: &TuiState) {
    let area = frame.area();
    let chunks = Layout::new(
        Direction::Vertical,
        [Constraint::Length(3), Constraint::Min(1), Constraint::Length(3)],
    )
    .split(area);

    let status = Line::from(vec![
        Span::styled(state.model.clone(), Style::default().bold()),
        Span::raw(format!(
            "  ·  {} prompt / {} completion tok",
            state.prompt_tokens, state.completion_tokens
        )),
    ]);
    frame.render_widget(
        Paragraph::new(status).block(Block::default().borders(Borders::ALL)),
        chunks[0],
    );

    let main = Paragraph::new(state.renderer.scrollback())
        .wrap(Wrap { trim: true })
        .scroll((state.scroll as u16, 0))
        .block(Block::default().borders(Borders::ALL));
    frame.render_widget(main, chunks[1]);

    let input_line = if state.running {
        Line::from(Span::styled("working…".to_string(), Style::default().dim()))
    } else if let Some(error) = &state.error {
        Line::from(Span::styled(error.clone(), Style::default().red()))
    } else {
        Line::from(vec![
            Span::styled("> ".to_string(), Style::default().bold()),
            Span::raw(state.input.clone()),
            Span::styled("█".to_string(), Style::default().bold()),
        ])
    };
    frame.render_widget(
        Paragraph::new(input_line).block(Block::default().borders(Borders::ALL)),
        chunks[2],
    );
}

pub async fn run(agent: Agent, model: String) -> Result<(), Box<dyn std::error::Error>> {
    let (agent_tx, mut agent_rx) = tokio::sync::mpsc::unbounded_channel::<TuiEvent>();
    let (input_tx, mut input_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    let (key_tx, mut key_rx) = tokio::sync::mpsc::unbounded_channel::<TermEvent>();

    std::io::stdout().execute(terminal::EnterAlternateScreen)?;
    std::io::stdout().execute(cursor::Hide)?;
    terminal::enable_raw_mode()?;
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = std::io::stdout().execute(terminal::LeaveAlternateScreen);
        let _ = std::io::stdout().execute(cursor::Show);
        default_hook(info);
    }));

    let backend = CrosstermBackend::new(std::io::stdout());
    let mut terminal = Terminal::new(backend)?;
    terminal.clear()?;

    std::thread::spawn(move || {
        while let Ok(term_event) = event::read() {
            if key_tx.send(term_event).is_err() {
                break;
            }
        }
    });

    let agent_task = tokio::spawn(async move {
        let mut agent = agent;
        while let Some(task) = input_rx.recv().await {
            let tx = agent_tx.clone();
            match agent
                .chat(
                    &task,
                    &mut |event| {
                        let _ = tx.send(TuiEvent::Agent(event));
                    },
                )
                .await
            {
                Ok(_answer) => {
                    let _ = tx.send(TuiEvent::TurnDone {
                        prompt: agent.context.total_prompt_tokens,
                        completion: agent.context.total_completion_tokens,
                    });
                }
                Err(error) => {
                    let _ = tx.send(TuiEvent::TurnError(error.to_string()));
                }
            }
        }
        drop(agent_tx);
    });

    let mut state = TuiState::new(model);
    terminal.draw(|frame| draw(frame, &state))?;

    loop {
        tokio::select! {
            event = agent_rx.recv() => {
                match event {
                    Some(TuiEvent::Agent(event)) => state.renderer.on_event(event),
                    Some(TuiEvent::TurnDone { prompt, completion }) => {
                        state.renderer.finish();
                        state.running = false;
                        state.prompt_tokens = prompt;
                        state.completion_tokens = completion;
                    }
                    Some(TuiEvent::TurnError(message)) => {
                        state.renderer.finish();
                        state.running = false;
                        state.error = Some(message);
                    }
                    None => break,
                }
            }
            key = key_rx.recv() => {
                let Some(term_event) = key else {
                    break;
                };
                match handle_key(&mut state, &term_event) {
                    KeyAction::Submit(task) => {
                        state.error = None;
                        state.running = true;
                        state.scroll = 0;
                        input_tx.send(task)?;
                    }
                    KeyAction::Quit => break,
                    KeyAction::None => {}
                }
            }
        }
        terminal.draw(|frame| draw(frame, &state))?;
    }

    agent_task.abort();
    std::io::stdout().execute(terminal::LeaveAlternateScreen)?;
    std::io::stdout().execute(cursor::Show)?;
    terminal::disable_raw_mode()?;
    println!(
        "session over: {} prompt / {} completion tokens",
        state.prompt_tokens, state.completion_tokens
    );
    Ok(())
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::agent::ChunkTokens;
    use crossterm::event::KeyEvent;
    use ratatui::backend::TestBackend;
    use ratatui::style::Modifier;

    fn lines(events: Vec<AgentEvent>) -> Vec<Line<'static>> {
        let mut renderer = TuiRenderer::new();
        for event in events {
            renderer.on_event(event);
        }
        renderer.finish();
        renderer.scrollback().to_vec()
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

    fn describe(lines: &[Line<'static>]) -> Vec<(String, Modifier)> {
        lines
            .iter()
            .map(|line| {
                let span = line.spans.first().cloned().unwrap_or_default();
                (span.content.to_string(), span.style.add_modifier)
            })
            .collect()
    }

    fn key(code: KeyCode) -> TermEvent {
        TermEvent::Key(KeyEvent::new(code, KeyModifiers::NONE))
    }

    #[test]
    fn thinking_then_answer() {
        let described = describe(&lines(vec![thinking("Let me think. "), text("42")]));

        assert_eq!(
            described,
            vec![
                ("Let me think. ".into(), Modifier::DIM),
                ("42".into(), Modifier::empty()),
            ]
        );
    }

    #[test]
    fn straggler_thinking_is_hidden() {
        let described = describe(&lines(vec![
            thinking("reasoning "),
            text("answer"),
            thinking("straggler"),
        ]));

        assert_eq!(
            described,
            vec![
                ("reasoning ".into(), Modifier::DIM),
                ("answer".into(), Modifier::empty()),
            ]
        );
    }

    #[test]
    fn tool_event_closes_thinking() {
        let described = describe(&lines(vec![
            thinking("thinking"),
            AgentEvent::ToolStarted {
                header: "bash".into(),
                body: Some("ls".into()),
            },
        ]));

        assert_eq!(
            described,
            vec![
                ("thinking".into(), Modifier::DIM),
                ("⚙ bash".into(), Modifier::BOLD),
                ("  ls".into(), Modifier::DIM),
            ]
        );
    }

    #[test]
    fn buffered_text_flushes_at_the_completion_boundary() {
        let described = describe(&lines(vec![
            AgentEvent::Tokens(ChunkTokens {
                thinking: Some("glued".into()),
                text: Some("narration".into()),
            }),
            AgentEvent::CompletionStarted,
            thinking("next round"),
        ]));

        assert_eq!(
            described,
            vec![
                ("glued".into(), Modifier::DIM),
                ("narration".into(), Modifier::empty()),
                ("next round".into(), Modifier::DIM),
            ]
        );
    }

    #[test]
    fn thinking_is_live_again_on_the_next_completion() {
        let described = describe(&lines(vec![
            thinking("first "),
            text("narration"),
            AgentEvent::CompletionStarted,
            thinking("second "),
            text("answer"),
        ]));

        assert_eq!(
            described,
            vec![
                ("first ".into(), Modifier::DIM),
                ("narration".into(), Modifier::empty()),
                ("second ".into(), Modifier::DIM),
                ("answer".into(), Modifier::empty()),
            ]
        );
    }

    #[test]
    fn tool_result_with_matching_header_skips_the_header() {
        let described = describe(&lines(vec![
            AgentEvent::ToolStarted {
                header: "read_file: a.txt".into(),
                body: None,
            },
            AgentEvent::ToolResult {
                header: "read_file: a.txt".into(),
                body: "line one".into(),
            },
        ]));

        assert_eq!(
            described,
            vec![
                ("⚙ read_file: a.txt".into(), Modifier::BOLD),
                ("  line one".into(), Modifier::DIM),
            ]
        );
    }

    #[test]
    fn parallel_tool_results_get_their_own_blocks() {
        let described = describe(&lines(vec![
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
        ]));

        assert_eq!(
            described,
            vec![
                ("⚙ read_file: a.txt".into(), Modifier::BOLD),
                ("⚙ read_file: b.txt".into(), Modifier::BOLD),
                ("⚙ read_file: a.txt".into(), Modifier::BOLD),
                ("  alpha".into(), Modifier::DIM),
                ("⚙ read_file: b.txt".into(), Modifier::BOLD),
                ("  beta".into(), Modifier::DIM),
            ]
        );
    }

    #[test]
    fn enter_submits_the_input() {
        let mut state = TuiState::new("model".into());
        state.input = "hello".into();

        assert_eq!(
            handle_key(&mut state, &key(KeyCode::Enter)),
            KeyAction::Submit("hello".into())
        );
        assert!(state.input.is_empty());
    }

    #[test]
    fn enter_with_empty_input_does_nothing() {
        let mut state = TuiState::new("model".into());

        assert_eq!(handle_key(&mut state, &key(KeyCode::Enter)), KeyAction::None);
    }

    #[test]
    fn backspace_pops_the_input() {
        let mut state = TuiState::new("model".into());
        state.input = "ab".into();

        assert_eq!(handle_key(&mut state, &key(KeyCode::Backspace)), KeyAction::None);
        assert_eq!(state.input, "a");
    }

    #[test]
    fn q_quits_only_with_empty_input() {
        let mut state = TuiState::new("model".into());
        assert_eq!(handle_key(&mut state, &key(KeyCode::Char('q'))), KeyAction::Quit);

        state.input = "x".into();
        assert_eq!(handle_key(&mut state, &key(KeyCode::Char('q'))), KeyAction::None);
        assert_eq!(state.input, "xq");
    }

    #[test]
    fn ctrl_c_quits() {
        let mut state = TuiState::new("model".into());
        let event = TermEvent::Key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL));

        assert_eq!(handle_key(&mut state, &event), KeyAction::Quit);
    }

    #[test]
    fn page_keys_scroll_the_main_pane() {
        let mut state = TuiState::new("model".into());
        for _ in 0..3 {
            state.renderer.on_event(text("x\ny\nz\n"));
        }

        assert_eq!(handle_key(&mut state, &key(KeyCode::PageUp)), KeyAction::None);
        assert_eq!(state.scroll, 10);
        assert_eq!(handle_key(&mut state, &key(KeyCode::PageUp)), KeyAction::None);
        assert_eq!(state.scroll, 12);
        assert_eq!(handle_key(&mut state, &key(KeyCode::PageDown)), KeyAction::None);
        assert_eq!(state.scroll, 2);
    }

    #[test]
    fn typing_is_ignored_while_running() {
        let mut state = TuiState::new("model".into());
        state.running = true;
        state.input = "keep".into();

        assert_eq!(handle_key(&mut state, &key(KeyCode::Char('a'))), KeyAction::None);
        assert_eq!(state.input, "keep");
        assert_eq!(
            handle_key(&mut state, &key(KeyCode::Enter)),
            KeyAction::None
        );
        assert_eq!(
            handle_key(
                &mut state,
                &TermEvent::Key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL))
            ),
            KeyAction::Quit
        );
    }

    #[test]
    fn frame_renders_status_main_and_input() {
        let mut state = TuiState::new("llama".into());
        state.renderer.on_event(text("hello"));
        state.prompt_tokens = 100;
        state.completion_tokens = 5;

        let backend = TestBackend::new(60, 12);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|frame| draw(frame, &state)).unwrap();

        let buffer = terminal.backend().buffer();
        let joined: String = (0..12)
            .flat_map(|y| {
                (0..60).map(move |x| buffer.cell((x, y)).unwrap().symbol().to_string())
            })
            .collect();
        assert!(joined.contains("llama"));
        assert!(joined.contains("100 prompt / 5 completion tok"));
        assert!(joined.contains("hello"));
        assert!(joined.contains(">"));
    }
}
