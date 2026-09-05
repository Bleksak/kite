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

use tui_markdown;

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
    answer: String,
    turn_answer: String,
    answer_start: Option<usize>,
    tool_header: Option<String>,
}

impl TuiRenderer {
    pub fn new() -> TuiRenderer {
        TuiRenderer {
            gate: AnswerGate::new(),
            scrollback: Vec::new(),
            thinking: String::new(),
            answer: String::new(),
            turn_answer: String::new(),
            answer_start: None,
            tool_header: None,
        }
    }

    pub fn scrollback(&self) -> &[Line<'static>] {
        &self.scrollback
    }

    pub fn push_user(&mut self, text: &str) {
        self.close_thinking();
        self.tool_header = None;
        for line in text.split('\n') {
            if line.is_empty() {
                self.scrollback.push(Line::default());
            } else {
                self.scrollback
                    .push(Line::from(Span::styled(line.to_string(), Style::default().bold())));
            }
        }
    }

    pub fn on_event(&mut self, event: AgentEvent) {
        match event {
            AgentEvent::CompletionStarted => {
                self.close_thinking();
                if let Some(remaining) = self.gate.finish() {
                    self.push_answer_text(&remaining);
                    self.turn_answer.push_str(&remaining);
                }
                self.flush_answer();
                self.render_answer_markdown();
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
                    self.push_answer_text(&text);
                    self.turn_answer.push_str(&text);
                }
            }
            AgentEvent::ToolStarted { header, body } => {
                self.flush_answer();
                self.close_thinking();
                self.tool_header = Some(header.clone());
                self.scrollback
                    .push(Line::from(Span::styled(format!("⚙ {header}"), Style::default().bold())));
                if let Some(body) = body {
                    self.push_indented(&body);
                }
            }
            AgentEvent::ToolResult { header, body } => {
                self.flush_answer();
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
            self.push_answer_text(&remaining);
            self.turn_answer.push_str(&remaining);
        }
        self.flush_answer();
        self.render_answer_markdown();
    }

    fn close_thinking(&mut self) {
        if self.thinking.is_empty() {
            return;
        }
        self.flush_answer();
        let thinking = mem::take(&mut self.thinking);
        for line in thinking.split('\n') {
            if line.is_empty() {
                self.scrollback.push(Line::default());
            } else {
                self.scrollback
                    .push(Line::from(Span::styled(line.to_string(), Style::default().dim())));
            }
        }
    }

    fn push_answer_text(&mut self, text: &str) {
        self.answer.push_str(text);
        while let Some(index) = self.answer.find('\n') {
            let line = self.answer[..index].to_string();
            self.answer.drain(..=index);
            self.mark_answer_start();
            if line.is_empty() {
                self.scrollback.push(Line::default());
            } else {
                self.scrollback.push(Line::from(line));
            }
        }
    }

    fn flush_answer(&mut self) {
        if self.answer.is_empty() {
            return;
        }
        let line = mem::take(&mut self.answer);
        self.mark_answer_start();
        self.scrollback.push(Line::from(line));
    }

    fn mark_answer_start(&mut self) {
        if self.answer_start.is_none() {
            self.answer_start = Some(self.scrollback.len());
        }
    }

    fn render_answer_markdown(&mut self) {
        if self.turn_answer.trim().is_empty() {
            return;
        }
        let Some(start) = self.answer_start else {
            return;
        };
        if !has_markdown(&self.turn_answer) {
            self.turn_answer.clear();
            self.answer_start = None;
            return;
        }
        let rendered = tui_markdown::from_str(&self.turn_answer);
        let lines: Vec<Line<'static>> = rendered
            .lines
            .iter()
            .map(|line| Line {
                style: line.style,
                alignment: line.alignment,
                spans: line
                    .spans
                    .iter()
                    .map(|span| Span {
                        content: std::borrow::Cow::Owned(span.content.to_string()),
                        style: span.style,
                    })
                    .collect(),
            })
            .collect();
        self.scrollback.splice(start.., lines);
        self.turn_answer.clear();
        self.answer_start = None;
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
}

fn has_markdown(text: &str) -> bool {
    const MARKERS: [&str; 7] = ["**", "`", "# ", "```", "- ", "[", "> "];
    MARKERS.iter().any(|marker| text.contains(marker))
}

pub struct TuiState {
    pub renderer: TuiRenderer,
    pub scroll: usize,
    pub following: bool,
    pub viewport: usize,
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
            following: true,
            viewport: 24,
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
                page_up(state);
                KeyAction::None
            }
            KeyCode::PageDown => {
                page_down(state);
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
                state.following = true;
                KeyAction::Submit(mem::take(&mut state.input))
            }
        }
        KeyCode::Backspace => {
            state.input.pop();
            KeyAction::None
        }
        KeyCode::PageUp => {
            page_up(state);
            KeyAction::None
        }
        KeyCode::PageDown => {
            page_down(state);
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

fn page_up(state: &mut TuiState) {
    state.scroll = state.scroll.saturating_sub(PAGE);
    state.following = false;
}

fn page_down(state: &mut TuiState) {
    let max = state.max_scroll();
    state.scroll = (state.scroll + PAGE).min(max);
    state.following = state.scroll == max;
}

impl TuiState {
    fn max_scroll(&self) -> usize {
        self.renderer.scrollback().len().saturating_sub(self.viewport)
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

    let scrollback = state.renderer.scrollback();
    let viewport = chunks[1].height as usize;
    let start = state.scroll.min(scrollback.len().saturating_sub(viewport));
    let main = Paragraph::new(&scrollback[start..])
        .wrap(Wrap { trim: true })
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
                if state.following {
                    state.scroll = state.max_scroll();
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
                        state.renderer.push_user(&task);
                        state.following = true;
                        state.scroll = state.max_scroll();
                        input_tx.send(task)?;
                    }
                    KeyAction::Quit => break,
                    KeyAction::None => {}
                }
            }
        }
        state.viewport = terminal
            .size()
            .map(|size| size.height.saturating_sub(6) as usize)
            .unwrap_or(24);
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
    fn answer_chunks_flow_into_one_line() {
        let described = describe(&lines(vec![text("Hello "), text("world")]));

        assert_eq!(
            described,
            vec![("Hello world".into(), Modifier::empty())]
        );
    }

    #[test]
    fn answer_newlines_split_lines() {
        let described = describe(&lines(vec![text("first\nsecond")]));

        assert_eq!(
            described,
            vec![
                ("first".into(), Modifier::empty()),
                ("second".into(), Modifier::empty()),
            ]
        );
    }

    #[test]
    fn user_question_is_pushed_to_the_scrollback() {
        let mut renderer = TuiRenderer::new();
        renderer.push_user("do the thing");
        renderer.finish();

        let described = describe(renderer.scrollback());
        assert_eq!(
            described,
            vec![("do the thing".into(), Modifier::BOLD)]
        );
    }

    #[test]
    fn page_keys_scroll_the_main_pane() {
        let mut state = TuiState::new("model".into());
        for _ in 0..3 {
            state.renderer.on_event(text("x\ny\nz\nw\n"));
        }
        state.renderer.finish();
        state.viewport = 2;
        state.following = true;
        state.scroll = state.max_scroll();

        assert_eq!(handle_key(&mut state, &key(KeyCode::PageUp)), KeyAction::None);
        assert_eq!(state.scroll, 0);
        assert!(!state.following);
        assert_eq!(handle_key(&mut state, &key(KeyCode::PageUp)), KeyAction::None);
        assert_eq!(state.scroll, 0);
        assert_eq!(handle_key(&mut state, &key(KeyCode::PageDown)), KeyAction::None);
        assert_eq!(state.scroll, 10);
        assert!(state.following);
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
    fn markdown_answer_renders_styled_lines_at_the_boundary() {
        let rendered = lines(vec![
            text("Some **bold** text"),
            AgentEvent::CompletionStarted,
        ]);

        assert_eq!(rendered.len(), 1);
        assert_eq!(rendered[0].spans.len(), 3);
        assert_eq!(rendered[0].spans[1].content, "bold");
        assert!(rendered[0].spans[1].style.add_modifier.contains(Modifier::BOLD));
    }

    #[test]
    fn markdown_heading_and_paragraph_render_styled() {
        let rendered = lines(vec![
            text("# Head\n\n**Bold**"),
            AgentEvent::CompletionStarted,
        ]);

        assert_eq!(rendered.len(), 3);
        assert!(rendered[0].style.add_modifier.contains(Modifier::BOLD));
        assert!(rendered[0].style.add_modifier.contains(Modifier::UNDERLINED));
        assert!(rendered[2].spans[0].style.add_modifier.contains(Modifier::BOLD));
    }

    #[test]
    fn thinking_is_not_markdown_rendered() {
        let described = describe(&lines(vec![thinking("**not** rendered"), text("done")]));

        assert_eq!(
            described,
            vec![
                ("**not** rendered".into(), Modifier::DIM),
                ("done".into(), Modifier::empty()),
            ]
        );
    }

    #[test]
    fn tool_output_is_not_markdown_rendered() {
        let described = describe(&lines(vec![AgentEvent::ToolStarted {
            header: "bash".into(),
            body: Some("**cmd**".into()),
        }]));

        assert_eq!(
            described,
            vec![
                ("⚙ bash".into(), Modifier::BOLD),
                ("  **cmd**".into(), Modifier::DIM),
            ]
        );
    }

    #[test]
    fn each_completion_renders_its_own_answer() {
        let rendered = lines(vec![
            text("first **a**"),
            AgentEvent::CompletionStarted,
            text("second **b**"),
        ]);

        assert_eq!(rendered.len(), 2);
        assert!(rendered[0].spans[1].style.add_modifier.contains(Modifier::BOLD));
        assert_eq!(rendered[0].spans[1].content, "a");
        assert!(rendered[1].spans[1].style.add_modifier.contains(Modifier::BOLD));
        assert_eq!(rendered[1].spans[1].content, "b");
    }

    #[test]
    fn frame_renders_status_main_and_input() {
        let mut state = TuiState::new("llama".into());
        state.renderer.on_event(text("hello"));
        state.renderer.finish();
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
