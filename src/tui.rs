use std::collections::HashMap;
use std::mem;
use std::sync::LazyLock;

use crossterm::cursor;
use crossterm::event::{self, Event as TermEvent, KeyCode, KeyModifiers};
use crossterm::terminal;
use crossterm::ExecutableCommand;
use ratatui::backend::CrosstermBackend;
use ratatui::buffer::{Buffer, Cell};
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap, Widget};
use ratatui::{Frame, Terminal};

use tui_markdown::{from_str_with_options, Options, StyleSheet};

#[derive(Clone)]
struct TuiStyleSheet;

impl StyleSheet for TuiStyleSheet {
    fn code_block_fence(&self) -> &str {
        ""
    }
}

static MD_OPTIONS: LazyLock<Options<TuiStyleSheet>> = LazyLock::new(|| Options::new(TuiStyleSheet));

use crate::agent::{Agent, AgentEvent, AnswerGate, ThinkingMode};

pub enum TuiEvent {
    Agent(AgentEvent),
    TurnDone { prompt: u64, completion: u64 },
    TurnError(String),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BlockKind {
    User,
    Thinking,
    Answer,
    Other,
}

pub struct TuiRenderer {
    gate: AnswerGate,
    scrollback: Vec<Line<'static>>,
    blocks: Vec<BlockKind>,
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
            blocks: Vec::new(),
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

    pub fn blocks(&self) -> &[BlockKind] {
        &self.blocks
    }

    fn push_line(&mut self, line: Line<'static>, block: BlockKind) {
        self.scrollback.push(line);
        self.blocks.push(block);
    }

    pub fn push_user(&mut self, text: &str) {
        self.close_thinking();
        self.tool_header = None;
        self.push_line(padding_line(), BlockKind::User);
        for line in text.trim_end_matches('\n').split('\n') {
            if line.is_empty() {
                self.push_line(Line::default(), BlockKind::User);
            } else {
                self.push_line(
                    Line::from(Span::styled(
                        strip_vs16(line),
                        Style::default()
                            .bold()
                            .fg(Color::Rgb(255, 255, 255)),
                    )),
                    BlockKind::User,
                );
            }
        }
        self.push_line(padding_line(), BlockKind::User);
    }

    pub fn on_event(&mut self, event: AgentEvent) {
        match event {
            AgentEvent::CompletionStarted => {
                self.close_thinking();
                if let Some(remaining) = self.gate.finish() {
                    self.push_answer_text(&remaining);
                    self.turn_answer.push_str(&strip_vs16(&remaining));
                }
                self.end_answer_block();
                self.gate = AnswerGate::new();
            }
            AgentEvent::Tokens(chunk) => {
                self.tool_header = None;
                let (mode, text) = self.gate.on_chunk(&chunk);
                if mode == ThinkingMode::Live
                    && let Some(thinking) = &chunk.thinking
                {
                    if self.thinking.is_empty() && !thinking.is_empty() {
                        self.push_line(padding_line(), BlockKind::Thinking);
                    }
                    self.thinking.push_str(thinking);
                }
                if let Some(text) = text {
                    self.close_thinking();
                    self.push_answer_text(&text);
                    self.turn_answer.push_str(&strip_vs16(&text));
                }
            }
            AgentEvent::ToolStarted { header, body } => {
                self.end_answer_block();
                self.close_thinking();
                self.tool_header = Some(header.clone());
                self.push_line(
                    Line::from(Span::styled(format!("⚙ {header}"), Style::default().bold())),
                    BlockKind::Other,
                );
                if let Some(body) = body {
                    self.push_indented(&body);
                }
            }
            AgentEvent::ToolResult { header, body } => {
                self.flush_answer();
                if self.tool_header.as_deref() != Some(header.as_str()) {
                    self.tool_header = Some(header.clone());
                    self.push_line(
                        Line::from(Span::styled(format!("⚙ {header}"), Style::default().bold())),
                        BlockKind::Other,
                    );
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
            self.turn_answer.push_str(&strip_vs16(&remaining));
        }
        self.end_answer_block();
    }

    fn end_answer_block(&mut self) {
        self.flush_answer();
        let had_answer = self.answer_start.is_some();
        self.render_answer_markdown();
        if had_answer {
            self.push_line(padding_line(), BlockKind::Answer);
        }
    }

    fn close_thinking(&mut self) {
        if self.thinking.is_empty() {
            return;
        }
        self.flush_answer();
        let thinking = mem::take(&mut self.thinking);
        for line in thinking
            .trim_start_matches('\n')
            .trim_end_matches('\n')
            .split('\n') {
            if line.is_empty() {
                self.push_line(Line::default(), BlockKind::Thinking);
            } else {
                self.push_line(
                    Line::from(Span::styled(strip_vs16(line), Style::default().fg(Color::Black))),
                    BlockKind::Thinking,
                );
            }
        }
        self.push_line(padding_line(), BlockKind::Thinking);
    }

    fn push_answer_text(&mut self, text: &str) {
        self.answer.push_str(text);
        while let Some(index) = self.answer.find('\n') {
            let line = strip_vs16(&self.answer[..index]);
            self.answer.drain(..=index);
            if line.trim().is_empty() && self.answer_start.is_none() {
                continue;
            }
            if self.answer_start.is_none() {
                self.push_line(padding_line(), BlockKind::Answer);
            }
            self.mark_answer_start();
            if line.is_empty() {
                self.push_line(Line::default(), BlockKind::Answer);
            } else {
                self.push_line(Line::from(line), BlockKind::Answer);
            }
        }
    }

    fn flush_answer(&mut self) {
        if self.answer.is_empty() {
            return;
        }
        let line = strip_vs16(&mem::take(&mut self.answer));
        if line.trim().is_empty() && self.answer_start.is_none() {
            return;
        }
        if self.answer_start.is_none() {
            self.push_line(padding_line(), BlockKind::Answer);
        }
        self.mark_answer_start();
        self.push_line(Line::from(line), BlockKind::Answer);
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
        let (prepared, restore) = prepare_markdown(&self.turn_answer);
        let rendered = from_str_with_options(&prepared, &MD_OPTIONS);
        let lines: Vec<Line<'static>> = rendered
            .lines
            .iter()
            .map(|line| {
                let original = line.to_string();
                let spans: Vec<Span> = match restore.get(&original) {
                    Some(rule) => {
                        let style = line.spans.first().map(|s| s.style).unwrap_or_default();
                        vec![Span {
                            content: std::borrow::Cow::Owned(rule.clone()),
                            style,
                        }]
                    }
                    None => line
                        .spans
                        .iter()
                        .map(|span| Span {
                            content: std::borrow::Cow::Owned(span.content.to_string()),
                            style: span.style,
                        })
                        .collect(),
                };
                Line {
                    style: line.style,
                    alignment: line.alignment,
                    spans,
                }
            })
            .collect();
        let count = lines.len();
        self.scrollback.splice(start.., lines);
        self.blocks.splice(
            start..,
            std::iter::repeat(BlockKind::Answer).take(count).collect::<Vec<_>>(),
        );
        self.turn_answer.clear();
        self.answer_start = None;
    }

    fn push_indented(&mut self, body: &str) {
        for line in body.split('\n') {
            if line.is_empty() {
                self.push_line(Line::default(), BlockKind::Other);
            } else {
                self.push_line(
                    Line::from(Span::styled(
                        format!("  {}", strip_vs16(line)),
                        Style::default().dim(),
                    )),
                    BlockKind::Other,
                );
            }
        }
    }
}

fn padding_line() -> Line<'static> {
    Line::from(Span::styled(
        " ".to_string(),
        Style::default().fg(Color::Red),
    ))
}

fn strip_vs16(text: &str) -> String {
    if text.contains('\u{FE0F}') {
        text.replace('\u{FE0F}', "")
    } else {
        text.to_string()
    }
}

fn has_markdown(text: &str) -> bool {
    const MARKERS: [&str; 7] = ["**", "`", "# ", "```", "- ", "[", "> "];
    MARKERS.iter().any(|marker| text.contains(marker))
}

fn prepare_markdown(text: &str) -> (String, HashMap<String, String>) {
    let mut out: Vec<String> = Vec::new();
    let mut restore: HashMap<String, String> = HashMap::new();
    let mut in_fence = false;
    let mut fence_marker = "";
    for line in text.lines() {
        let trimmed = line.trim_end();
        let body = trimmed.trim_start();
        let indent = trimmed.len() - body.len();
        let fence = if body.starts_with("```") {
            "```"
        } else if body.starts_with("~~~") {
            "~~~"
        } else {
            ""
        };
        if !fence.is_empty() {
            if in_fence {
                if fence == fence_marker {
                    in_fence = false;
                }
            } else {
                in_fence = true;
                fence_marker = fence;
            }
            out.push(trimmed.to_string());
            continue;
        }
        if in_fence {
            out.push(trimmed.to_string());
            continue;
        }
        let compact = body.replace(' ', "");
        let is_rule = compact.len() >= 3
            && indent < 4
            && (compact.chars().all(|c| c == '*') || compact.chars().all(|c| c == '_'));
        if is_rule {
            if out.last().is_some_and(|last| !last.trim().is_empty()) {
                out.push(String::new());
            }
            let token = format!("kiterule{}", restore.len());
            restore.insert(token.clone(), body.to_string());
            out.push(token);
        } else {
            out.push(trimmed.to_string());
        }
    }
    let mut joined: Vec<String> = Vec::with_capacity(out.len() + 1);
    for (i, line) in out.iter().enumerate() {
        joined.push(line.clone());
        if line.starts_with("kiterule") && i + 1 < out.len() && !out[i + 1].trim().is_empty() {
            joined.push(String::new());
        }
    }
    (joined.join("\n"), restore)
}

pub struct TuiState {
    pub renderer: TuiRenderer,
    pub scroll: usize,
    pub following: bool,
    pub viewport: usize,
    pub pane_width: usize,
    tail_cache: (usize, usize, usize, usize),
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
            viewport: 22,
            pane_width: 118,
            tail_cache: (0, 0, 0, 0),
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
    fn max_scroll(&mut self) -> usize {
        let len = self.renderer.scrollback().len();
        let (cl, cw, cv, cs) = self.tail_cache;
        if cl == len && cw == self.pane_width && cv == self.viewport {
            return cs;
        }
        let result = tail_start(len, self.renderer.scrollback(), self.pane_width as u16, self.viewport);
        self.tail_cache = (len, self.pane_width, self.viewport, result);
        result
    }
}

fn fits_viewport(lines: &[Line<'static>], width: u16, viewport: usize) -> bool {
    if lines.is_empty() {
        return true;
    }
    let area = Rect::new(0, 0, width, (viewport + 1) as u16);
    let mut buffer = Buffer::empty(area);
    Paragraph::new(lines).wrap(Wrap { trim: false }).render(area, &mut buffer);
    (0..width).all(|x| {
        buffer
            .cell((x, viewport as u16))
            .map_or(true, |cell| *cell == Cell::default())
    })
}

fn tail_start(len: usize, lines: &[Line<'static>], width: u16, viewport: usize) -> usize {
    let first = len.saturating_sub(viewport);
    if fits_viewport(&lines[first..], width, viewport) {
        return first;
    }
    let mut lo = first;
    let mut hi = len;
    while lo < hi {
        let mid = (lo + hi) / 2;
        if fits_viewport(&lines[mid..], width, viewport) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    lo
}

fn display_lines(
    scrollback: &[Line<'static>],
    blocks: &[BlockKind],
    width: usize,
) -> Vec<Line<'static>> {
    scrollback
        .iter()
        .zip(blocks.iter())
        .map(|(line, block)| {
            let (text_background, block_background) = match block {
                BlockKind::User => (None, Some(Color::Rgb(128, 128, 128))),
                BlockKind::Thinking => (None, Some(Color::Rgb(192, 192, 192))),
                _ => (None, None),
            };
            if text_background.is_none() && block_background.is_none() {
                return Line {
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
                };
            }
            let text_background = text_background.or(block_background);
            let mut spans: Vec<Span> = line
                .spans
                .iter()
                .map(|span| {
                    let style = match text_background {
                        Some(bg) => span.style.patch(Style::default().bg(bg)),
                        None => span.style,
                    };
                    Span {
                        content: std::borrow::Cow::Owned(span.content.to_string()),
                        style,
                    }
                })
                .collect();
            let current = line.width();
            if current < width {
                if let Some(bg) = block_background {
                    spans.push(Span::styled(
                        " ".repeat(width - current),
                        Style::default().bg(bg),
                    ));
                } else {
                    spans.push(Span::styled(
                        " ".repeat(width - current),
                        Style::default(),
                    ));
                }
            }
            Line {
                style: line.style,
                alignment: line.alignment,
                spans,
            }
        })
        .collect()
}

pub fn draw(frame: &mut Frame, state: &TuiState, start: usize) {
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

    let display = display_lines(state.renderer.scrollback(), state.renderer.blocks(), state.pane_width);
    let display = &display[start.min(display.len())..];
    let main = Paragraph::new(display)
        .wrap(Wrap { trim: false })
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::DarkGray)),
        );
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
    terminal.draw(|frame| draw(frame, &state, 0))?;

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
            .map(|size| size.height.saturating_sub(8) as usize)
            .unwrap_or(22);
        state.pane_width = terminal
            .size()
            .map(|size| size.width.saturating_sub(2) as usize)
            .unwrap_or(118);
        let start = state.scroll.min(state.max_scroll());
        terminal.draw(|frame| draw(frame, &state, start))?;
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
                (" ".into(), Modifier::empty()),
                ("Let me think. ".into(), Modifier::empty()),
                (" ".into(), Modifier::empty()),
                (" ".into(), Modifier::empty()),
                ("42".into(), Modifier::empty()),
                (" ".into(), Modifier::empty()),
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
                (" ".into(), Modifier::empty()),
                ("reasoning ".into(), Modifier::empty()),
                (" ".into(), Modifier::empty()),
                (" ".into(), Modifier::empty()),
                ("answer".into(), Modifier::empty()),
                (" ".into(), Modifier::empty()),
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
                (" ".into(), Modifier::empty()),
                ("thinking".into(), Modifier::empty()),
                (" ".into(), Modifier::empty()),
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
                (" ".into(), Modifier::empty()),
                ("glued".into(), Modifier::empty()),
                (" ".into(), Modifier::empty()),
                (" ".into(), Modifier::empty()),
                ("narration".into(), Modifier::empty()),
                (" ".into(), Modifier::empty()),
                (" ".into(), Modifier::empty()),
                ("next round".into(), Modifier::empty()),
                (" ".into(), Modifier::empty()),
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
                (" ".into(), Modifier::empty()),
                ("first ".into(), Modifier::empty()),
                (" ".into(), Modifier::empty()),
                (" ".into(), Modifier::empty()),
                ("narration".into(), Modifier::empty()),
                (" ".into(), Modifier::empty()),
                (" ".into(), Modifier::empty()),
                ("second ".into(), Modifier::empty()),
                (" ".into(), Modifier::empty()),
                (" ".into(), Modifier::empty()),
                ("answer".into(), Modifier::empty()),
                (" ".into(), Modifier::empty()),
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
            vec![
                (" ".into(), Modifier::empty()),
                ("Hello world".into(), Modifier::empty()),
                (" ".into(), Modifier::empty()),
            ]
        );
    }

    #[test]
    fn answer_newlines_split_lines() {
        let described = describe(&lines(vec![text("first\nsecond")]));

        assert_eq!(
            described,
            vec![
                (" ".into(), Modifier::empty()),
                ("first".into(), Modifier::empty()),
                ("second".into(), Modifier::empty()),
                (" ".into(), Modifier::empty()),
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
            vec![
                (" ".into(), Modifier::empty()),
                ("do the thing".into(), Modifier::BOLD),
                (" ".into(), Modifier::empty()),
            ]
        );
    }

    #[test]
    fn user_block_stores_white_text_and_the_block_kind() {
        let mut renderer = TuiRenderer::new();
        renderer.push_user("do the thing");
        renderer.finish();

        let line = &renderer.scrollback()[1];
        let span = &line.spans[0];
        assert_eq!(span.style.fg, Some(Color::Rgb(255, 255, 255)));
        assert_eq!(span.style.bg, None);
        assert!(span.style.add_modifier.contains(Modifier::BOLD));
        assert_eq!(renderer.blocks()[1], BlockKind::User);
    }

    #[test]
    fn display_lines_pad_user_and_thinking_blocks_to_full_width() {
        let mut renderer = TuiRenderer::new();
        renderer.push_user("hi");
        renderer.on_event(thinking("think"));
        renderer.on_event(text("ans"));
        renderer.finish();

        let display = display_lines(renderer.scrollback(), renderer.blocks(), 40);
        let user = &display[1];
        assert_eq!(user.width(), 40);
        assert_eq!(user.spans[0].style.bg, Some(Color::Rgb(128, 128, 128)));
        assert_eq!(user.spans.last().unwrap().style.bg, Some(Color::Rgb(128, 128, 128)));
        let thinking = &display[4];
        assert_eq!(thinking.width(), 40);
        assert_eq!(thinking.spans.last().unwrap().style.bg, Some(Color::Rgb(192, 192, 192)));
        assert_eq!(thinking.spans[0].style.fg, Some(Color::Black));
        let answer = &display[7];
        assert!(answer.spans.iter().all(|s| s.style.bg.is_none()));
    }

    #[test]
    fn a_leading_blank_line_in_the_answer_is_not_pushed() {
        let described = describe(&lines(vec![text("\n\nHello"), text("\nworld")]));

        assert_eq!(
            described,
            vec![
                (" ".into(), Modifier::empty()),
                ("Hello".into(), Modifier::empty()),
                ("world".into(), Modifier::empty()),
                (" ".into(), Modifier::empty()),
            ]
        );
    }

    #[test]
    fn a_leading_newline_in_thinking_is_not_pushed() {
        let described = describe(&lines(vec![thinking("\n\nLet me think."), text("42")]));

        assert_eq!(
            described,
            vec![
                (" ".into(), Modifier::empty()),
                ("Let me think.".into(), Modifier::empty()),
                (" ".into(), Modifier::empty()),
                (" ".into(), Modifier::empty()),
                ("42".into(), Modifier::empty()),
                (" ".into(), Modifier::empty()),
            ]
        );
    }

    #[test]
    fn an_internal_blank_line_in_the_answer_is_kept() {
        let described = describe(&lines(vec![text("one\n\ntwo")]));

        assert_eq!(
            described,
            vec![
                (" ".into(), Modifier::empty()),
                ("one".into(), Modifier::empty()),
                ("".into(), Modifier::empty()),
                ("two".into(), Modifier::empty()),
                (" ".into(), Modifier::empty()),
            ]
        );
    }

    #[test]
    fn tail_start_accounts_for_wrapped_lines() {
        let mut state = TuiState::new("model".into());
        state.viewport = 4;
        state.pane_width = 20;
        for _ in 0..3 {
            state.renderer.on_event(text(&format!("{}\n", "a".repeat(40))));
        }
        state.renderer.finish();

        assert_eq!(state.max_scroll(), 3);
    }





    #[test]
    fn tail_shows_the_last_line_of_a_large_scrollback() {
        let mut state = TuiState::new("model".into());
        state.viewport = 22;
        state.pane_width = 118;
        for i in 0..30 {
            let width = if i % 3 == 0 { 300 } else { 20 };
            state.renderer.on_event(text(&format!("line {} {}\n", i, "x".repeat(width))));
        }
        state.renderer.finish();
        state.following = true;
        state.scroll = state.max_scroll();

        let scrollback = state.renderer.scrollback();
        let last_line = scrollback.last().unwrap().to_string();
        let mut buffer = Buffer::empty(Rect::new(0, 0, 118, 24));
        Paragraph::new(&scrollback[state.scroll..])
            .wrap(Wrap { trim: false })
            .render(Rect::new(0, 0, 118, 22), &mut buffer);
        let mut rows = Vec::new();
        for y in 0..22 {
            let mut row = String::new();
            for x in 0..118 {
                row.push_str(buffer.cell((x, y)).unwrap().symbol());
            }
            rows.push(row);
        }
        let joined = rows.join("\n");
        assert!(
            joined.contains(&last_line),
            "last line {last_line} missing from tail viewport:\n{joined}"
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
        assert_eq!(state.scroll, 2);
        assert!(!state.following);
        assert_eq!(handle_key(&mut state, &key(KeyCode::PageUp)), KeyAction::None);
        assert_eq!(state.scroll, 0);
        assert_eq!(handle_key(&mut state, &key(KeyCode::PageDown)), KeyAction::None);
        assert_eq!(state.scroll, 10);
        assert!(!state.following);
        assert_eq!(handle_key(&mut state, &key(KeyCode::PageDown)), KeyAction::None);
        assert_eq!(state.scroll, 12);
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

        assert_eq!(rendered.len(), 3);
        assert_eq!(rendered[1].spans.len(), 3);
        assert_eq!(rendered[1].spans[1].content, "bold");
        assert!(rendered[1].spans[1].style.add_modifier.contains(Modifier::BOLD));
    }

    #[test]
    fn markdown_heading_and_paragraph_render_styled() {
        let rendered = lines(vec![
            text("# Head\n\n**Bold**"),
            AgentEvent::CompletionStarted,
        ]);

        assert_eq!(rendered.len(), 5);
        assert!(rendered[1].style.add_modifier.contains(Modifier::BOLD));
        assert!(rendered[1].style.add_modifier.contains(Modifier::UNDERLINED));
        assert!(rendered[3].spans[0].style.add_modifier.contains(Modifier::BOLD));
    }

    #[test]
    fn thinking_is_not_markdown_rendered() {
        let described = describe(&lines(vec![thinking("**not** rendered"), text("done")]));

        assert_eq!(
            described,
            vec![
                (" ".into(), Modifier::empty()),
                ("**not** rendered".into(), Modifier::empty()),
                (" ".into(), Modifier::empty()),
                (" ".into(), Modifier::empty()),
                ("done".into(), Modifier::empty()),
                (" ".into(), Modifier::empty()),
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

        assert_eq!(rendered.len(), 6);
        assert!(rendered[1].spans[1].style.add_modifier.contains(Modifier::BOLD));
        assert_eq!(rendered[1].spans[1].content, "a");
        assert!(rendered[4].spans[1].style.add_modifier.contains(Modifier::BOLD));
        assert_eq!(rendered[4].spans[1].content, "b");
    }

    #[test]
    fn code_block_renders_without_fences() {
        let rendered = lines(vec![
            text("```python\nx = 1\n```\n"),
            AgentEvent::CompletionStarted,
        ]);

        for line in &rendered {
            assert!(!line.to_string().contains("```"));
        }
    }

    #[test]
    fn asterisk_horizontal_rule_is_preserved() {
        let rendered = lines(vec![
            text("para\n\n***\n\nnext"),
            AgentEvent::CompletionStarted,
        ]);
        let described: Vec<String> = rendered.iter().map(|l| l.to_string()).collect();

        assert!(described.contains(&"***".into()), "{described:?}");
        assert!(!described.iter().any(|l| l == "---"), "{described:?}");
    }

    #[test]
    fn dash_horizontal_rule_stays_a_dash_rule() {
        let rendered = lines(vec![
            text("para\n\n---\n\nnext"),
            AgentEvent::CompletionStarted,
        ]);
        let described: Vec<String> = rendered.iter().map(|l| l.to_string()).collect();

        assert!(described.contains(&"---".into()), "{described:?}");
    }

    #[test]
    fn rule_adjacent_to_paragraph_is_not_merged() {
        let rendered = lines(vec![
            text("para\n***\nnext"),
            AgentEvent::CompletionStarted,
        ]);
        let described: Vec<String> = rendered.iter().map(|l| l.to_string()).collect();

        assert!(described.contains(&"***".into()), "{described:?}");
        assert!(
            !described.iter().any(|l| l.contains("para ***") || l.contains("*** next")),
            "{described:?}"
        );
    }

    #[test]
    fn rule_like_line_inside_code_fence_is_untouched() {
        let rendered = lines(vec![
            text("```bash\n***\n```
"),
            AgentEvent::CompletionStarted,
        ]);
        let described: Vec<String> = rendered.iter().map(|l| l.to_string()).collect();

        assert!(described.contains(&"***".into()), "{described:?}");
    }

    #[test]
    fn underscore_rule_is_preserved() {
        let rendered = lines(vec![
            text("para\n\n___\n\nnext"),
            AgentEvent::CompletionStarted,
        ]);
        let described: Vec<String> = rendered.iter().map(|l| l.to_string()).collect();

        assert!(described.contains(&"___".into()), "{described:?}");
        assert!(!described.iter().any(|l| l == "---"), "{described:?}");
    }

    #[test]
    fn code_block_gets_syntax_highlighting() {
        let rendered = lines(vec![
            text("```python\nx = 1\n```\n"),
            AgentEvent::CompletionStarted,
        ]);

        let has_color = rendered
            .iter()
            .any(|line| line.spans.iter().any(|span| span.style.fg.is_some()));
        assert!(has_color);
    }




    #[test]
    fn scroll_does_not_leave_stale_cells() {
        let backend = TestBackend::new(40, 5);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|frame| {
                Paragraph::new(vec![Line::from(
                    "a very long line of text that wraps around the corner",
                )])
                .wrap(Wrap { trim: false })
                .render(frame.area(), frame.buffer_mut());
            })
            .unwrap();
        terminal
            .draw(|frame| {
                Paragraph::new(vec![Line::from("short")])
                    .wrap(Wrap { trim: false })
                    .render(frame.area(), frame.buffer_mut());
            })
            .unwrap();
        let buffer = terminal.backend().buffer();
        for y in 0..5 {
            for x in 5..40 {
                let cell = buffer.cell((x, y)).unwrap();
                assert_eq!(
                    cell.symbol(),
                    " ",
                    "stale cell at ({x}, {y}): {}",
                    cell.symbol()
                );
            }
        }
    }

    #[test]
    fn vs16_emoji_are_stripped_from_the_scrollback() {
        let rendered = lines(vec![
            text("# \u{23F8}\u{FE0F} Note\n\nbody"),
            AgentEvent::CompletionStarted,
        ]);
        let described: Vec<String> = rendered.iter().map(|l| l.to_string()).collect();

        assert!(
            !described.iter().any(|l| l.contains('\u{FE0F}')),
            "VS16 selector must not reach the scrollback: {described:?}"
        );
        assert!(
            described.iter().any(|l| l.contains("\u{23F8}")),
            "the emoji itself must survive: {described:?}"
        );
    }

    #[test]
    fn vs16_emoji_are_stripped_from_live_answer_lines() {
        let mut renderer = TuiRenderer::new();
        renderer.on_event(text("line with \u{23F8}\u{FE0F} emoji\n"));
        renderer.finish();

        let described: Vec<String> = renderer
            .scrollback()
            .iter()
            .map(|l| l.to_string())
            .collect();
        assert!(
            !described.iter().any(|l| l.contains('\u{FE0F}')),
            "VS16 selector must not reach the scrollback: {described:?}"
        );
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
        terminal.draw(|frame| draw(frame, &state, 0)).unwrap();

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
