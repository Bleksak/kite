use std::collections::HashMap;
use std::mem;
use std::path::Path;
use std::sync::{Arc, Mutex};

use crossterm::cursor;
use crossterm::event::{self, Event as TermEvent, KeyCode, KeyModifiers};
use crossterm::terminal;
use crossterm::ExecutableCommand;
use ratatui::backend::CrosstermBackend;
use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap, Widget};
use ratatui::{Frame, Terminal};


use crate::agent::Agent;
use crate::thinking::ThinkingLevel;
use crate::mode::Mode;
use crate::context::Context;
use crate::paths::CONTEXT_DIR;
use crate::session::{Cursor, Scroller, Session};
use crate::session_store::{self, SessionFile};
use crate::stream::AgentEvent;
use crate::transcript::BlockKind;

pub enum TuiEvent {
    Agent { session: u64, event: AgentEvent },
    TurnDone { session: u64, context: Context },
    TurnError { session: u64, message: String },
}

pub struct TuiState {
    pub sessions: Vec<Session>,
    pub active: usize,
    pub viewport: usize,
    pub pane_width: usize,
    pub model: String,
    pub thinking: Arc<Mutex<ThinkingLevel>>,
    pub mode: Arc<Mutex<Mode>>,
    pub picker_open: bool,
    pub picker_cursor: Cursor,
    pub picker_query: String,
    pub picker_rename: Option<String>,
    pub tasks_open: bool,
    pub tasks_cursor: Cursor,
    pub task_output_id: Option<String>,
    pub task_output_scroll: Scroller,
    next_id: u64,
}

impl TuiState {
    pub fn new(model: String) -> TuiState {
        TuiState {
            sessions: vec![Session::new(0)],
            active: 0,
            viewport: 22,
            pane_width: 118,
            model,
            thinking: Arc::new(std::sync::Mutex::new(ThinkingLevel::Off)),
            mode: Arc::new(std::sync::Mutex::new(Mode::Yolo)),
            picker_open: false,
            picker_cursor: Cursor::default(),
            picker_query: String::new(),
            picker_rename: None,
            tasks_open: false,
            tasks_cursor: Cursor::default(),
            task_output_id: None,
            task_output_scroll: Scroller::default(),
            next_id: 1,
        }
    }

    pub fn session(&mut self) -> &mut Session {
        &mut self.sessions[self.active]
    }

    pub fn with_thinking(mut self, cell: Arc<Mutex<ThinkingLevel>>) -> TuiState {
        self.thinking = cell;
        self
    }

    pub fn with_mode(mut self, cell: Arc<Mutex<Mode>>) -> TuiState {
        self.mode = cell;
        self
    }

    pub fn with_sessions(model: String, loaded: Vec<(u64, SessionFile)>) -> TuiState {
        let mut state = TuiState::new(model);
        state.sessions = Vec::new();
        state.next_id = 0;
        for (id, file) in loaded {
            let mut session = Session::new(id);
            session.label = file.label;
            session.prompt_tokens = file.context.total_prompt_tokens;
            session.completion_tokens = file.context.total_completion_tokens;
            session.context = Some(file.context);
            state.sessions.push(session);
            state.next_id = state.next_id.max(id + 1);
        }
        state.sessions.push(Session::new(state.next_id));
        state.active = state.sessions.len() - 1;
        state.next_id += 1;
        state
    }

    pub fn filtered(&self) -> Vec<usize> {
        if self.picker_query.is_empty() {
            return (0..self.sessions.len()).collect();
        }
        let query = self.picker_query.to_lowercase();
        (0..self.sessions.len())
            .filter(|i| {
                self.sessions[*i].label.to_lowercase().contains(&query)
                    || i.to_string().contains(&query)
            })
            .collect()
    }
}

#[derive(Debug, PartialEq)]
pub enum KeyAction {
    None,
    Submit(String),
    Quit,
    NewSession,
    CloseSession,
}

const PAGE: usize = 10;

fn task_output_scroll_max(state: &TuiState) -> usize {
    let Some(id) = &state.task_output_id else {
        return 0;
    };
    let Some(task) = crate::bg::REGISTRY.list().into_iter().find(|t| t.id == *id) else {
        return 0;
    };
    let text = std::fs::read_to_string(&task.output_path).unwrap_or_default();
    let total = text.lines().count();
    let visible = state.viewport.saturating_sub(10);
    total.saturating_sub(visible)
}

pub fn handle_key(state: &mut TuiState, event: &TermEvent) -> KeyAction {
    if let TermEvent::Mouse(mouse) = event {
        if state.task_output_id.is_some() {
            let max = task_output_scroll_max(state);
            return match mouse.kind {
                event::MouseEventKind::ScrollUp => {
                    state.task_output_scroll.toward_top(3);
                    KeyAction::None
                }
                event::MouseEventKind::ScrollDown => {
                    state.task_output_scroll.toward_bottom(3, max);
                    KeyAction::None
                }
                _ => KeyAction::None,
            };
        }
        let (pane_width, viewport) = (state.pane_width, state.viewport);
        let session = state.session();
        return match mouse.kind {
            event::MouseEventKind::ScrollUp => {
                mouse_up(session);
                KeyAction::None
            }
            event::MouseEventKind::ScrollDown => {
                mouse_down(session, pane_width, viewport);
                KeyAction::None
            }
            _ => KeyAction::None,
        };
    }
    let TermEvent::Key(key) = event else {
        return KeyAction::None;
    };
    if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
        return KeyAction::Quit;
    }
    if state.task_output_id.is_some() {
        if key.modifiers.contains(KeyModifiers::CONTROL) {
            return match key.code {
                KeyCode::Char('q') => {
                    state.task_output_id = None;
                    state.tasks_open = false;
                    KeyAction::None
                }
                _ => KeyAction::None,
            };
        }
        let scroll_max = task_output_scroll_max(state);
        return match key.code {
            KeyCode::Esc | KeyCode::Char('q') => {
                state.task_output_id = None;
                KeyAction::None
            }
            KeyCode::Char('j') | KeyCode::Down => {
                state.task_output_scroll.toward_bottom(3, scroll_max);
                KeyAction::None
            }
            KeyCode::Char('k') | KeyCode::Up => {
                state.task_output_scroll.toward_top(3);
                KeyAction::None
            }
            KeyCode::PageDown => {
                state.task_output_scroll.toward_bottom(8, scroll_max);
                KeyAction::None
            }
            KeyCode::PageUp => {
                state.task_output_scroll.toward_top(8);
                KeyAction::None
            }
            KeyCode::Home => {
                state.task_output_scroll.home();
                KeyAction::None
            }
            KeyCode::End => {
                state.task_output_scroll.end(scroll_max);
                KeyAction::None
            }
            _ => KeyAction::None,
        };
    }
    if state.picker_open {
        if state.picker_rename.is_some() {
            return match key.code {
                KeyCode::Enter => {
                    let buf = state.picker_rename.take().unwrap();
                    if let Some(i) = state.filtered().get(state.picker_cursor.pos).copied() {
                        state.sessions[i].label = buf;
                    }
                    KeyAction::None
                }
                KeyCode::Esc => {
                    state.picker_rename = None;
                    KeyAction::None
                }
                KeyCode::Backspace => {
                    if let Some(buf) = &mut state.picker_rename {
                        buf.pop();
                    }
                    KeyAction::None
                }
                KeyCode::Char(c) => {
                    if let Some(buf) = &mut state.picker_rename {
                        buf.push(c);
                    }
                    KeyAction::None
                }
                _ => KeyAction::None,
            };
        }
        if key.modifiers.contains(KeyModifiers::CONTROL) {
            return match key.code {
                KeyCode::Char('j') => {
                    state.picker_cursor.down(state.filtered().len());
                    KeyAction::None
                }
                KeyCode::Char('k') => {
                    state.picker_cursor.up(state.filtered().len());
                    KeyAction::None
                }
                KeyCode::Char('n') => {
                    state.sessions.push(Session::new(state.next_id));
                    state.active = state.sessions.len() - 1;
                    state.next_id += 1;
                    state.picker_open = false;
                    KeyAction::NewSession
                }
                KeyCode::Char('x') => {
                    let filtered = state.filtered();
                    if let Some(i) = filtered.get(state.picker_cursor.pos).copied()
                        && state.sessions.len() > 1
                        && !state.sessions[i].running
                    {
                        state.sessions.remove(i);
                        if state.active >= state.sessions.len() {
                            state.active = state.sessions.len() - 1;
                        }
                    }
                    state.picker_cursor.clamp(state.filtered().len());
                    KeyAction::CloseSession
                }
                KeyCode::Char('r') => {
                    state.picker_rename = Some(String::new());
                    KeyAction::None
                }
                KeyCode::Char('q') => {
                    state.tasks_open = true;
                    state.picker_open = false;
                    KeyAction::None
                }
                KeyCode::Char('s') => {
                    state.picker_open = false;
                    KeyAction::None
                }
                _ => KeyAction::None,
            };
        }
        return match key.code {
            KeyCode::Up => {
                state.picker_cursor.up(state.filtered().len());
                KeyAction::None
            }
            KeyCode::Down => {
                state.picker_cursor.down(state.filtered().len());
                KeyAction::None
            }
            KeyCode::Enter => {
                let filtered = state.filtered();
                if let Some(i) = filtered.get(state.picker_cursor.pos).copied() {
                    state.active = i;
                }
                state.picker_open = false;
                KeyAction::None
            }
            KeyCode::Esc => {
                state.picker_open = false;
                KeyAction::None
            }
            KeyCode::Backspace => {
                state.picker_query.pop();
                state.picker_cursor.clamp(state.filtered().len());
                KeyAction::None
            }
            KeyCode::Char(c) => {
                state.picker_query.push(c);
                state.picker_cursor.clamp(state.filtered().len());
                KeyAction::None
            }
            _ => KeyAction::None,
        };
    }
    if state.tasks_open {
        let tasks = crate::bg::REGISTRY.list();
        let len = tasks.len();
        if key.modifiers.contains(KeyModifiers::CONTROL) {
            return match key.code {
                KeyCode::Char('j') => {
                    state.tasks_cursor.down(len);
                    KeyAction::None
                }
                KeyCode::Char('k') => {
                    state.tasks_cursor.up(len);
                    KeyAction::None
                }
                KeyCode::Char('s') => {
                    state.picker_open = true;
                    state.tasks_open = false;
                    KeyAction::None
                }
                KeyCode::Char('q') => {
                    state.tasks_open = false;
                    KeyAction::None
                }
                _ => KeyAction::None,
            };
        }
        return match key.code {
            KeyCode::Up => {
                state.tasks_cursor.up(len);
                KeyAction::None
            }
            KeyCode::Down => {
                state.tasks_cursor.down(len);
                KeyAction::None
            }
            KeyCode::Char('j') => {
                state.tasks_cursor.down(len);
                KeyAction::None
            }
            KeyCode::Char('k') => {
                state.tasks_cursor.up(len);
                KeyAction::None
            }
            KeyCode::Char('x') => {
                if let Some(task) = tasks.get(state.tasks_cursor.pos) {
                    let _ = crate::bg::REGISTRY.kill(&task.id);
                }
                KeyAction::None
            }
            KeyCode::Enter => {
                if let Some(task) = tasks.get(state.tasks_cursor.pos) {
                    state.task_output_id = Some(task.id.clone());
                    state.task_output_scroll = Scroller::at_tail();
                }
                KeyAction::None
            }
            KeyCode::Char('q') | KeyCode::Esc => {
                state.tasks_open = false;
                KeyAction::None
            }
            _ => KeyAction::None,
        };
    }
    if key.modifiers.contains(KeyModifiers::CONTROL) {
        return match key.code {
            KeyCode::Char('s') => {
                state.picker_open = true;
                state.tasks_open = false;
                state.picker_cursor.set(state.active);
                KeyAction::None
            }
            KeyCode::Char('q') => {
                state.tasks_open = true;
                state.picker_open = false;
                state.tasks_cursor.set(0);
                KeyAction::None
            }
            KeyCode::Char('t') => {
                let mut level = state.thinking.lock().unwrap();
                *level = level.next();
                KeyAction::None
            }
            _ => KeyAction::None,
        };
    }
    if key.code == KeyCode::Tab && !key.modifiers.contains(KeyModifiers::CONTROL) {
        let mut mode = state.mode.lock().unwrap();
        *mode = mode.next();
        return KeyAction::None;
    }
    let (pane_width, viewport) = (state.pane_width, state.viewport);
    let session = state.session();
    if session.running {
        return match key.code {
            KeyCode::PageUp => {
                page_up(session);
                KeyAction::None
            }
            KeyCode::PageDown => {
                page_down(session, pane_width, viewport);
                KeyAction::None
            }
            _ => KeyAction::None,
        };
    }
    match key.code {
        KeyCode::Enter => {
            if session.input.is_empty() {
                KeyAction::None
            } else {
                let task = mem::take(&mut session.input);
                if session.label.is_empty() {
                    session.label = task.trim().chars().take(24).collect();
                }
                session.scroller.set_following(true);
                KeyAction::Submit(task)
            }
        }
        KeyCode::Backspace => {
            session.input.pop();
            KeyAction::None
        }
        KeyCode::PageUp => {
            page_up(session);
            KeyAction::None
        }
        KeyCode::PageDown => {
            page_down(session, pane_width, viewport);
            KeyAction::None
        }
        KeyCode::Home => {
            to_top(session);
            KeyAction::None
        }
        KeyCode::End => {
            to_bottom(session, pane_width, viewport);
            KeyAction::None
        }
        KeyCode::Char(c) => {
            session.input.push(c);
            KeyAction::None
        }
        _ => KeyAction::None,
    }
}

fn page_up(session: &mut Session) {
    session.scroller.toward_top(PAGE);
}

fn page_down(session: &mut Session, pane_width: usize, viewport: usize) {
    let max = session.max_scroll(pane_width, viewport);
    session.scroller.toward_bottom(PAGE, max);
}

fn to_top(session: &mut Session) {
    session.scroller.home();
}

fn to_bottom(session: &mut Session, pane_width: usize, viewport: usize) {
    let max = session.max_scroll(pane_width, viewport);
    session.scroller.end(max);
}

const MOUSE: usize = 3;

fn mouse_up(session: &mut Session) {
    session.scroller.toward_top(MOUSE);
}

fn mouse_down(session: &mut Session, pane_width: usize, viewport: usize) {
    let max = session.max_scroll(pane_width, viewport);
    session.scroller.toward_bottom(MOUSE, max);
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
                BlockKind::User => (None, Some(Color::Rgb(52, 53, 65))),
                BlockKind::Thinking => (None, None),
                BlockKind::ToolRunning => (None, Some(Color::Rgb(40, 40, 50))),
                BlockKind::ToolDone => (None, Some(Color::Rgb(40, 50, 40))),
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

struct Fill;

impl Widget for Fill {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let spaces = " ".repeat(area.width as usize);
        for y in 0..area.height {
            buf.set_string(area.x, area.y + y, &spaces, Style::default());
        }
    }
}

pub fn draw(frame: &mut Frame, state: &TuiState, start: usize) {
    let area = frame.area();
    let chunks = Layout::new(
        Direction::Vertical,
        [Constraint::Length(3), Constraint::Min(1), Constraint::Length(3)],
    )
    .split(area);
    let session = &state.sessions[state.active];

    let mut status_spans: Vec<Span> = vec![
        Span::styled(state.model.clone(), Style::default().bold()),
        Span::raw(format!(
            "  ·  {} prompt / {} completion tok",
            session.prompt_tokens, session.completion_tokens
        )),
        Span::styled(
            format!("  ·  [{}/{}]", state.active + 1, state.sessions.len()),
            Style::default().fg(Color::Rgb(95, 135, 255)),
        ),
    ];
    let running_bg = crate::bg::REGISTRY
        .list()
        .into_iter()
        .filter(|task| matches!(task.status, crate::bg::BgStatus::Running))
        .count();
    if running_bg > 0 {
        status_spans.push(Span::styled(
            format!("  ·  ⏺{running_bg} bg"),
            Style::default().fg(Color::Rgb(0x81, 0xa2, 0xbe)),
        ));
    }
    let thinking_level = *state.thinking.lock().unwrap();
    status_spans.push(Span::styled(
        format!("  ·  💭 {}", thinking_level.label()),
        Style::default().fg(Color::Rgb(0x81, 0xa2, 0xbe)),
    ));
    let status = Line::from(status_spans);
    frame.render_widget(
        Paragraph::new(status).block(Block::default().borders(Borders::ALL)),
        chunks[0],
    );

    let display = display_lines(session.renderer.scrollback(), session.renderer.blocks(), state.pane_width);
    let display = &display[start.min(display.len())..];
    let main = Paragraph::new(display)
        .wrap(Wrap { trim: false })
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Rgb(95, 135, 255))),
        );
    frame.render_widget(Fill, chunks[1]);
    frame.render_widget(main, chunks[1]);

    if state.picker_open {
        let filtered = state.filtered();
        let renaming = state.picker_rename.clone();
        let list_height = filtered.len().max(1) as u16;
        let height = list_height + 3;
        let width = 52u16.min(chunks[1].width.saturating_sub(2));
        let x = chunks[1].x + (chunks[1].width.saturating_sub(width)) / 2;
        let y = chunks[1].y + (chunks[1].height.saturating_sub(height)) / 2;
        let lines: Vec<Line> = if filtered.is_empty() {
            vec![Line::from(Span::styled(
                " no matches",
                Style::default().fg(Color::Rgb(102, 102, 102)),
            ))]
        } else {
            filtered
                .iter()
                .enumerate()
                .map(|(i, session_idx)| {
                    let selected = i == state.picker_cursor.pos;
                    let session = &state.sessions[*session_idx];
                    let marker = if selected { "›" } else { " " };
                    let status = if session.running { "working…" } else { "idle" };
                    let style = if selected {
                        Style::default().bold().fg(Color::Rgb(95, 135, 255))
                    } else {
                        Style::default()
                    };
                    let label = match &renaming {
                        Some(buf) if selected => {
                            if buf.is_empty() {
                                "rename…".to_string()
                            } else {
                                buf.clone()
                            }
                        }
                        _ => {
                            if session.label.is_empty() {
                                "—".to_string()
                            } else {
                                session.label.clone()
                            }
                        }
                    };
                    Line::from(vec![
                        Span::styled(format!(" {marker} {}  ", session_idx + 1), style),
                        Span::styled(label, style),
                        Span::styled(
                            format!("  {} · {} / {}", status, session.prompt_tokens, session.completion_tokens),
                            style,
                        ),
                    ])
                })
                .collect()
        };
        let hint = Line::from(Span::styled(
            " C-jk · enter · C-n · C-x · C-r · search",
            Style::default().fg(Color::Rgb(102, 102, 102)),
        ));
        let title = if renaming.is_some() {
            " rename".to_string()
        } else if state.picker_query.is_empty() {
            " sessions".to_string()
        } else {
            format!(" sessions · {}", state.picker_query)
        };
        let picker = Paragraph::new(
            lines
                .into_iter()
                .chain(std::iter::once(hint))
                .collect::<Vec<Line>>(),
        )
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Rgb(95, 135, 255)))
                .style(Style::default().bg(Color::Rgb(0x28, 0x28, 0x32)))
                .title(format!(" {} ", title)),
        );
        frame.render_widget(Fill, Rect::new(x, y, width, height));
        frame.render_widget(picker, Rect::new(x, y, width, height));
    }

    if state.tasks_open && state.task_output_id.is_none() {
        let tasks = crate::bg::REGISTRY.list();
        let list_height = tasks.len().max(1) as u16;
        let height = list_height + 3;
        let width = 52u16.min(chunks[1].width.saturating_sub(2));
        let x = chunks[1].x + (chunks[1].width.saturating_sub(width)) / 2;
        let y = chunks[1].y + (chunks[1].height.saturating_sub(height)) / 2;
        let lines: Vec<Line> = if tasks.is_empty() {
            vec![Line::from(Span::styled(
                " no tasks",
                Style::default().fg(Color::Rgb(102, 102, 102)),
            ))]
        } else {
            tasks
                .iter()
                .enumerate()
                .map(|(i, task)| {
                    let selected = i == state.tasks_cursor.pos;
                    let status = match &task.status {
                        crate::bg::BgStatus::Running => "running  ".to_string(),
                        crate::bg::BgStatus::Finished(None) => "killed   ".to_string(),
                        crate::bg::BgStatus::Finished(Some(0)) => "exit 0   ".to_string(),
                        crate::bg::BgStatus::Finished(Some(code)) => format!("exit {code:<4}") ,
                    };
                    let duration = task
                        .finished_at
                        .map(|finished| finished.duration_since(task.started_at))
                        .unwrap_or_else(|| std::time::Instant::now().duration_since(task.started_at));
                    let style = if selected {
                        Style::default().bold().fg(Color::Rgb(95, 135, 255))
                    } else {
                        Style::default()
                    };
                    let marker = if selected { "›" } else { " " };
                    let command: String = task.command.chars().take(28).collect();
                    Line::from(vec![
                        Span::styled(format!(" {marker} {}  ", task.id), style),
                        Span::styled(status, style),
                        Span::styled(
                            format!("{}  ", crate::bg::format_duration(duration)),
                            style,
                        ),
                        Span::styled(command, style),
                    ])
                })
                .collect()
        };
        let hint = Line::from(Span::styled(
            " jk · x kill · q close",
            Style::default().fg(Color::Rgb(102, 102, 102)),
        ));
        let tasks_box = Paragraph::new(
            lines
                .into_iter()
                .chain(std::iter::once(hint))
                .collect::<Vec<Line>>(),
        )
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Rgb(95, 135, 255)))
                .style(Style::default().bg(Color::Rgb(0x28, 0x28, 0x32)))
                .title(" background tasks ".to_string()),
        );
        frame.render_widget(Fill, Rect::new(x, y, width, height));
        frame.render_widget(tasks_box, Rect::new(x, y, width, height));
    }

    if let Some(id) = &state.task_output_id {
        let tasks = crate::bg::REGISTRY.list();
        if let Some(task) = tasks.iter().find(|t| t.id == *id) {
            let text = std::fs::read_to_string(&task.output_path).unwrap_or_default();
            let lines: Vec<&str> = text.lines().collect();
            let width = chunks[1].width;
            let height = chunks[1].height;
            let x = chunks[1].x;
            let y = chunks[1].y;
            let visible = height as usize - 4;
            let total = lines.len();
            let max = total.saturating_sub(visible);
            let start = if state.task_output_scroll.following() {
                max
            } else {
                state.task_output_scroll.offset().min(max)
            };
            let status = match &task.status {
                crate::bg::BgStatus::Running => "running".to_string(),
                crate::bg::BgStatus::Finished(None) => "killed".to_string(),
                crate::bg::BgStatus::Finished(Some(0)) => "exit 0".to_string(),
                crate::bg::BgStatus::Finished(Some(code)) => format!("exit {code}"),
            };
            let mut body: Vec<Line> = vec![Line::from(vec![
                Span::styled(format!("$ {}", task.command), Style::default().bold()),
                Span::styled(format!("  ·  {}", status), Style::default().fg(Color::Rgb(0x81, 0xa2, 0xbe))),
            ])];
            if total == 0 {
                body.push(Line::from(Span::styled(
                    if matches!(task.status, crate::bg::BgStatus::Running) {
                        " (no output yet — task is running) "
                    } else {
                        " (no output) "
                    },
                    Style::default().fg(Color::Rgb(102, 102, 102)),
                )));
            } else {
                body.extend(
                    lines[start..].iter().take(visible).map(|line| {
                        Line::from(Span::styled(
                            *line,
                            Style::default().fg(Color::Rgb(0x80, 0x80, 0x80)),
                        ))
                    }),
                );
            }
            let hint = Line::from(Span::styled(
                " jk scroll · q close · C-q close all",
                Style::default().fg(Color::Rgb(102, 102, 102)),
            ));
            let output_box = Paragraph::new(
                body
                    .into_iter()
                    .chain(std::iter::once(hint))
                    .collect::<Vec<Line>>(),
            )
            .wrap(Wrap { trim: false })
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::Rgb(95, 135, 255)))
                    .style(Style::default().bg(Color::Rgb(0x28, 0x28, 0x32)))
                    .title(format!(" task {} output ", task.id)),
            );
            frame.render_widget(Fill, Rect::new(x, y, width, height));
            frame.render_widget(output_box, Rect::new(x, y, width, height));
        }
    }

    let input_line = if session.running {
        Line::from(Span::styled("working…".to_string(), Style::default().dim()))
    } else if let Some(error) = &session.error {
        Line::from(Span::styled(error.clone(), Style::default().red()))
    } else {
        Line::from(vec![
            Span::styled("> ".to_string(), Style::default().bold()),
            Span::raw(session.input.clone()),
            Span::styled("█".to_string(), Style::default().bold()),
        ])
    };
    let mode = *state.mode.lock().unwrap();
    let mode_title = Line::from(Span::styled(
        format!(" {} {} ", if mode == Mode::Plan { "📋" } else { "⚒" }, mode.label()),
        Style::default()
            .bold()
            .fg(if mode == Mode::Plan {
                Color::Rgb(0xb5, 0xbd, 0x68)
            } else {
                Color::Rgb(0xd4, 0xd4, 0xd4)
            }),
    ));
    frame.render_widget(
        Paragraph::new(input_line)
            .block(Block::default().borders(Borders::ALL).title(mode_title)),
        chunks[2],
    );
}

fn spawn_agent(
    id: u64,
    agent: Agent,
    event_tx: tokio::sync::mpsc::UnboundedSender<TuiEvent>,
) -> (tokio::sync::mpsc::UnboundedSender<String>, tokio::task::AbortHandle) {
    let (input_tx, mut input_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    let task = tokio::spawn(async move {
        let mut agent = agent;
        let mut bg_rx = crate::bg::REGISTRY.subscribe();
        loop {
            tokio::select! {
                input = input_rx.recv() => {
                    let Some(input) = input else { break; };
                    let tx = event_tx.clone();
                    match agent
                        .chat(
                            &input,
                            &mut |event| {
                                let _ = tx.send(TuiEvent::Agent { session: id, event });
                            },
                        )
                        .await
                    {
                        Ok(_answer) => {
                            let context = agent.context.clone();
                            let _ = tx.send(TuiEvent::TurnDone { session: id, context });
                        }
                        Err(error) => {
                            let _ = tx.send(TuiEvent::TurnError {
                                session: id,
                                message: error.to_string(),
                            });
                        }
                    }
                }
                signal = bg_rx.recv() => {
                    if let Ok(task_id) = signal
                        && agent.owns_and_unseen(&task_id)
                    {
                        let tx = event_tx.clone();
                        match agent
                            .bg_turn(&mut |event| {
                                let _ = tx.send(TuiEvent::Agent { session: id, event });
                            })
                            .await
                        {
                            Ok(_answer) => {
                                let context = agent.context.clone();
                                let _ = tx.send(TuiEvent::TurnDone { session: id, context });
                            }
                            Err(error) => {
                                let _ = tx.send(TuiEvent::TurnError {
                                    session: id,
                                    message: error.to_string(),
                                });
                            }
                        }
                    }
                }
            }
        }
    });
    (input_tx, task.abort_handle())
}

pub async fn run(
    new_agent: impl Fn() -> Agent + Send + Sync + 'static,
    model: String,
    thinking: Arc<Mutex<ThinkingLevel>>,
    mode: Arc<Mutex<Mode>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let (agent_tx, mut agent_rx) = tokio::sync::mpsc::unbounded_channel::<TuiEvent>();
    let (key_tx, mut key_rx) = tokio::sync::mpsc::unbounded_channel::<TermEvent>();
    let new_agent = std::sync::Arc::new(new_agent);

    std::io::stdout().execute(terminal::EnterAlternateScreen)?;
    std::io::stdout().execute(cursor::Hide)?;
    terminal::enable_raw_mode()?;
    std::io::stdout().execute(event::EnableMouseCapture)?;
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = std::io::stdout().execute(terminal::LeaveAlternateScreen);
        let _ = std::io::stdout().execute(cursor::Show);
        let _ = std::io::stdout().execute(event::DisableMouseCapture);
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

    let mut state = TuiState::with_sessions(model, session_store::load_sessions(Path::new(CONTEXT_DIR)))
        .with_thinking(thinking)
        .with_mode(mode);
    let mut inputs: HashMap<u64, tokio::sync::mpsc::UnboundedSender<String>> = HashMap::new();
    let mut handles: HashMap<u64, tokio::task::AbortHandle> = HashMap::new();

    for session in &state.sessions {
        let mut agent = new_agent();
        if let Some(context) = &session.context {
            agent.context = context.clone();
        }
        let (input_tx, input_handle) = spawn_agent(session.id, agent, agent_tx.clone());
        inputs.insert(session.id, input_tx);
        handles.insert(session.id, input_handle);
    }

    for session in &mut state.sessions {
        if let Some(context) = &session.context {
            session.renderer.replay_context(context);
            let max = session.max_scroll(state.pane_width, state.viewport);
            session.scroller.end(max);
        }
    }

    terminal.draw(|frame| draw(frame, &state, 0))?;

    let mut session_watch = tokio::time::interval(std::time::Duration::from_secs(2));

    loop {
        tokio::select! {
            _ = session_watch.tick() => {
                let removed: Vec<u64> = state
                    .sessions
                    .iter()
                    .filter(|s| {
                        s.context.is_some()
                            && !session_store::session_file_exists(Path::new(CONTEXT_DIR), s.id)
                    })
                    .map(|s| s.id)
                    .collect();
                for id in removed {
                    if let Some(handle) = handles.remove(&id) {
                        handle.abort();
                    }
                    inputs.remove(&id);
                    let pos = state.sessions.iter().position(|s| s.id == id).unwrap();
                    state.sessions.remove(pos);
                    if state.active > pos && state.active > 0 {
                        state.active -= 1;
                    }
                    if state.active >= state.sessions.len() {
                        state.active = state.sessions.len().saturating_sub(1);
                    }
                }
            }
            event = agent_rx.recv() => {
                match event {
                    Some(TuiEvent::Agent { session: id, event }) => {
                        if let Some(session) = state.sessions.iter_mut().find(|s| s.id == id) {
                            session.renderer.on_event(event);
                            let max = session.max_scroll(state.pane_width, state.viewport);
                            session.scroller.follow_tail(max);
                        }
                    }
                    Some(TuiEvent::TurnDone {
                        session: id,
                        context,
                    }) => {
                        if let Some(session) = state.sessions.iter_mut().find(|s| s.id == id) {
                            session.renderer.finish();
                            session.running = false;
                            session.prompt_tokens = context.total_prompt_tokens;
                            session.completion_tokens = context.total_completion_tokens;
                            session.context = Some(context.clone());
                            if let Some(saved) = &session.context {
                                session_store::save_session(Path::new(CONTEXT_DIR), id, &session.label, saved);
                            }
                        }
                    }
                    Some(TuiEvent::TurnError { session: id, message }) => {
                        if let Some(session) = state.sessions.iter_mut().find(|s| s.id == id) {
                            session.renderer.finish();
                            session.running = false;
                            session.error = Some(message);
                        }
                    }
                    None => break,
                }
            }
            key = key_rx.recv() => {
                let Some(term_event) = key else {
                    break;
                };
                let ids_before: Vec<u64> = state.sessions.iter().map(|s| s.id).collect();
                match handle_key(&mut state, &term_event) {
                    KeyAction::Submit(task) => {
                        let id = state.sessions[state.active].id;
                        let session = &mut state.sessions[state.active];
                        session.error = None;
                        session.running = true;
                        session.renderer.push_user(&task);
                        let max = session.max_scroll(state.pane_width, state.viewport);
                        session.scroller.end(max);
                        inputs[&id].send(task)?;
                    }
                    KeyAction::Quit => break,
                    KeyAction::NewSession => {
                        let id = state.sessions[state.active].id;
                        let (input_tx, input_handle) =
                            spawn_agent(id, new_agent(), agent_tx.clone());
                        inputs.insert(id, input_tx);
                        handles.insert(id, input_handle);
                    }
                    KeyAction::CloseSession => {
                        let removed = ids_before
                            .into_iter()
                            .find(|id| !state.sessions.iter().any(|s| s.id == *id));
                        if let Some(id) = removed {
                            if let Some(handle) = handles.remove(&id) {
                                handle.abort();
                            }
                            inputs.remove(&id);
                            session_store::remove_session_file(Path::new(CONTEXT_DIR), id);
                        }
                    }
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
        let start = state.sessions[state.active]
            .scroller
            .offset()
            .min(state.sessions[state.active].max_scroll(state.pane_width, state.viewport));
        terminal.draw(|frame| draw(frame, &state, start))?;
    }

    for session in &state.sessions {
        if !session.running
            && let Some(context) = &session.context
            && session_store::session_file_exists(Path::new(CONTEXT_DIR), session.id)
        {
            session_store::save_session(Path::new(CONTEXT_DIR), session.id, &session.label, context);
        }
    }
    for handle in handles.values() {
        handle.abort();
    }
    std::io::stdout().execute(event::DisableMouseCapture)?;
    std::io::stdout().execute(terminal::LeaveAlternateScreen)?;
    std::io::stdout().execute(cursor::Show)?;
    terminal::disable_raw_mode()?;
    println!(
        "session over: {} prompt / {} completion tokens",
        state.sessions[state.active].prompt_tokens,
        state.sessions[state.active].completion_tokens
    );
    Ok(())
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::stream::ChunkTokens;
    use crate::message::Message;
    use crate::transcript::TuiRenderer;
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

    fn mouse(kind: event::MouseEventKind) -> TermEvent {
        TermEvent::Mouse(event::MouseEvent {
            kind,
            column: 0,
            row: 0,
            modifiers: KeyModifiers::NONE,
        })
    }

    #[test]
    fn mouse_wheel_scrolls_the_viewport() {
        let mut state = TuiState::new("model".into());
        for i in 0..30 {
            state.session().renderer.on_event(text(&format!("line {i}\n")));
        }
        state.session().renderer.finish();
        let (pw, vp) = (state.pane_width, state.viewport);
        let max = state.session().max_scroll(pw, vp);
        assert!(max > 0);

        handle_key(&mut state, &mouse(event::MouseEventKind::ScrollDown));
        assert_eq!(state.session().scroller.offset(), 3);
        assert!(!state.session().scroller.following());

        handle_key(&mut state, &mouse(event::MouseEventKind::ScrollUp));
        assert_eq!(state.session().scroller.offset(), 0);
        assert!(!state.session().scroller.following());

        for _ in 0..20 {
            handle_key(&mut state, &mouse(event::MouseEventKind::ScrollDown));
        }
        assert_eq!(state.session().scroller.offset(), max);
        assert!(state.session().scroller.following());
    }

    fn ctrl(c: char) -> TermEvent {
        TermEvent::Key(KeyEvent::new(KeyCode::Char(c), KeyModifiers::CONTROL))
    }

    #[test]
    fn ctrl_s_opens_the_picker_on_the_active_session() {
        let mut state = TuiState::new("model".into());
        state.sessions.push(Session::new(1));
        state.active = 1;

        handle_key(&mut state, &ctrl('s'));

        assert!(state.picker_open);
        assert_eq!(state.picker_cursor.pos, 1);

        handle_key(&mut state, &ctrl('s'));
        assert!(!state.picker_open);

        handle_key(&mut state, &ctrl('q'));
        assert!(state.tasks_open);

        handle_key(&mut state, &ctrl('q'));
        assert!(!state.tasks_open);
    }

    #[test]
    fn picker_navigation_selection_and_cancel() {
        let mut state = TuiState::new("model".into());
        state.sessions.push(Session::new(1));
        state.sessions.push(Session::new(2));
        state.active = 2;

        handle_key(&mut state, &ctrl('s'));
        handle_key(&mut state, &ctrl('k'));
        assert_eq!(state.picker_cursor.pos, 1);
        handle_key(&mut state, &ctrl('k'));
        assert_eq!(state.picker_cursor.pos, 0);
        handle_key(&mut state, &ctrl('j'));
        assert_eq!(state.picker_cursor.pos, 1);
        handle_key(&mut state, &key(KeyCode::Enter));
        assert_eq!(state.active, 1);
        assert!(!state.picker_open);

        handle_key(&mut state, &ctrl('s'));
        handle_key(&mut state, &key(KeyCode::Esc));
        assert!(!state.picker_open);
        assert_eq!(state.active, 1);
    }

    #[test]
    fn picker_n_creates_a_session_and_closes_the_picker() {
        let mut state = TuiState::new("model".into());

        handle_key(&mut state, &ctrl('s'));
        handle_key(&mut state, &ctrl('n'));

        assert_eq!(state.sessions.len(), 2);
        assert_eq!(state.active, 1);
        assert!(!state.picker_open);
    }

    #[test]
    fn picker_x_closes_the_cursor_session() {
        let mut state = TuiState::new("model".into());
        state.sessions.push(Session::new(1));
        state.active = 1;
        state.session().running = true;

        handle_key(&mut state, &ctrl('s'));
        handle_key(&mut state, &ctrl('x'));
        assert_eq!(state.sessions.len(), 2);

        state.session().running = false;
        handle_key(&mut state, &ctrl('x'));
        assert_eq!(state.sessions.len(), 1);
        assert_eq!(state.active, 0);

        handle_key(&mut state, &ctrl('s'));
        assert!(!state.picker_open);
        handle_key(&mut state, &ctrl('s'));
        handle_key(&mut state, &ctrl('x'));
        assert_eq!(state.sessions.len(), 1);
        assert!(state.picker_open);
    }

    #[test]
    fn picker_c_r_renames_the_cursor_session() {
        let mut state = TuiState::new("model".into());
        state.sessions.push(Session::new(1));
        state.sessions[1].label = "old name".into();

        handle_key(&mut state, &ctrl('s'));
        handle_key(&mut state, &ctrl('j'));
        handle_key(&mut state, &ctrl('r'));
        assert!(state.picker_rename.is_some());
        handle_key(&mut state, &key(KeyCode::Char('n')));
        handle_key(&mut state, &key(KeyCode::Char('e')));
        handle_key(&mut state, &key(KeyCode::Char('w')));
        handle_key(&mut state, &key(KeyCode::Backspace));
        handle_key(&mut state, &key(KeyCode::Enter));
        assert_eq!(state.sessions[1].label, "ne");
        assert!(state.picker_rename.is_none());

        handle_key(&mut state, &ctrl('r'));
        handle_key(&mut state, &key(KeyCode::Char('x')));
        handle_key(&mut state, &key(KeyCode::Esc));
        assert_eq!(state.sessions[1].label, "ne");
        assert!(state.picker_rename.is_none());
    }

    #[test]
    fn picker_typing_searches_and_backspace_clears() {
        let mut state = TuiState::new("model".into());
        state.picker_open = true;

        handle_key(&mut state, &key(KeyCode::Char('f')));
        assert_eq!(state.picker_query, "f");
        assert_eq!(state.session().input, "");
        handle_key(&mut state, &key(KeyCode::Backspace));
        assert_eq!(state.picker_query, "");
        assert!(state.picker_open);
    }

    #[test]
    fn picker_search_filters_sessions() {
        let mut state = TuiState::new("model".into());
        state.sessions.push(Session::new(1));
        state.sessions.push(Session::new(2));
        state.sessions[0].label = "fix login".into();
        state.sessions[1].label = "refactor parser".into();
        state.picker_open = true;

        handle_key(&mut state, &key(KeyCode::Char('l')));
        assert_eq!(state.filtered(), vec![0]);
        assert_eq!(state.picker_cursor.pos, 0);

        handle_key(&mut state, &key(KeyCode::Char('2')));
        assert!(state.filtered().is_empty());

        handle_key(&mut state, &key(KeyCode::Backspace));
        assert_eq!(state.filtered(), vec![0]);
    }

    #[tokio::test]
    async fn ctrl_q_opens_tasks_overlay_navigates_and_kills() {
        let id = crate::bg::REGISTRY.run("sleep 30").unwrap();
        let mut state = TuiState::new("model".into());

        handle_key(&mut state, &ctrl('q'));
        state.tasks_cursor.set(crate::bg::REGISTRY
            .list()
.iter()
.position(|t| t.id == id)
.unwrap());
        assert!(state.tasks_open);
        assert!(!state.picker_open);

        handle_key(&mut state, &ctrl('s'));
        assert!(state.picker_open);
        assert!(!state.tasks_open);

        handle_key(&mut state, &ctrl('q'));
        assert!(state.tasks_open);
        assert!(!state.picker_open);

        handle_key(&mut state, &key(KeyCode::Char('x')));
        for _ in 0..200 {
            if matches!(
                crate::bg::REGISTRY.list().into_iter().find(|t| t.id == id).unwrap().status,
                crate::bg::BgStatus::Finished(_)
            ) {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        assert!(matches!(
            crate::bg::REGISTRY.list().into_iter().find(|t| t.id == id).unwrap().status,
            crate::bg::BgStatus::Finished(_)
        ));

        handle_key(&mut state, &key(KeyCode::Esc));
        assert!(!state.tasks_open);
    }

    #[test]
    fn tasks_overlay_ignores_navigation_with_no_tasks() {
        let mut state = TuiState::new("model".into());

        handle_key(&mut state, &ctrl('q'));
        handle_key(&mut state, &key(KeyCode::Char('j')));
        handle_key(&mut state, &key(KeyCode::Char('k')));
        handle_key(&mut state, &key(KeyCode::Char('x')));
        assert_eq!(state.tasks_cursor.pos, 0);
        handle_key(&mut state, &key(KeyCode::Char('q')));
        assert!(!state.tasks_open);
    }

    #[tokio::test]
    async fn enter_shows_task_output_and_keys_navigate_and_close() {
        let id = crate::bg::REGISTRY.run("seq 1 30").unwrap();
        for _ in 0..200 {
            if matches!(
                crate::bg::REGISTRY.list().into_iter().find(|t| t.id == id).unwrap().status,
                crate::bg::BgStatus::Finished(_)
            ) {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        let mut state = TuiState::new("model".into());

        handle_key(&mut state, &ctrl('q'));
        state.tasks_cursor.set(crate::bg::REGISTRY
            .list()
.iter()
.position(|t| t.id == id)
.unwrap());
        handle_key(&mut state, &key(KeyCode::Enter));
        assert_eq!(state.task_output_id.as_deref(), Some(id.as_str()));

        handle_key(&mut state, &key(KeyCode::Char('j')));
        assert_eq!(state.task_output_scroll.offset(), 3);
        handle_key(&mut state, &key(KeyCode::Char('k')));
        assert_eq!(state.task_output_scroll.offset(), 0);
        handle_key(&mut state, &key(KeyCode::PageUp));
        assert_eq!(state.task_output_scroll.offset(), 0);
        handle_key(&mut state, &key(KeyCode::PageDown));
        assert_eq!(state.task_output_scroll.offset(), 8);
        handle_key(&mut state, &key(KeyCode::Home));
        assert_eq!(state.task_output_scroll.offset(), 0);
        assert!(!state.task_output_scroll.following());
        handle_key(&mut state, &key(KeyCode::End));
        assert_eq!(state.task_output_scroll.offset(), 18);
        assert!(state.task_output_scroll.following());

        handle_key(&mut state, &key(KeyCode::Esc));
        assert!(state.task_output_id.is_none());
        assert!(state.tasks_open);

        handle_key(&mut state, &ctrl('q'));
        assert!(!state.tasks_open);
    }

    #[tokio::test]
    async fn task_output_popup_renders_the_output() {
        let id = crate::bg::REGISTRY.run("echo popup-visible").unwrap();
        for _ in 0..200 {
            if matches!(
                crate::bg::REGISTRY.list().into_iter().find(|t| t.id == id).unwrap().status,
                crate::bg::BgStatus::Finished(_)
            ) {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        let mut state = TuiState::new("model".into());
        state.tasks_open = true;
        state.task_output_id = Some(id);

        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|frame| draw(frame, &state, 0)).unwrap();

        let buffer = terminal.backend().buffer();
        let joined: String = (0..24)
            .flat_map(|y| {
                (0..80).map(move |x| buffer.cell((x, y)).unwrap().symbol().to_string())
            })
            .collect();
        assert!(joined.contains("popup-visible"));
        assert!(joined.contains("task"));
    }

    #[tokio::test]
    async fn task_output_popup_is_opaque_over_chat() {
        let id = crate::bg::REGISTRY.run("seq 1 30").unwrap();
        for _ in 0..200 {
            if matches!(
                crate::bg::REGISTRY.list().into_iter().find(|t| t.id == id).unwrap().status,
                crate::bg::BgStatus::Finished(_)
            ) {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        let mut state = TuiState::new("model".into());
        for i in 0..5 {
            state.session().renderer.push_user(&format!("CHATLINE-{i}"));
        }
        state.tasks_open = true;
        state.task_output_id = Some(id);
        state.task_output_scroll.set_following(true);

        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|frame| draw(frame, &state, 0)).unwrap();

        let buffer = terminal.backend().buffer();
        let in_popup: String = (3..21)
            .flat_map(|y| {
                (0..80).map(move |x| buffer.cell((x, y)).unwrap().symbol().to_string())
            })
            .collect();
        assert!(!in_popup.contains("CHATLINE"));
        assert!(in_popup.contains("30"));
    }

    #[tokio::test]
    async fn mouse_wheel_scrolls_the_open_output_popup() {
        let id = crate::bg::REGISTRY.run("seq 1 30").unwrap();
        for _ in 0..200 {
            if matches!(
                crate::bg::REGISTRY.list().into_iter().find(|t| t.id == id).unwrap().status,
                crate::bg::BgStatus::Finished(_)
            ) {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        let mut state = TuiState::new("model".into());
        state.task_output_id = Some(id);
        state.task_output_scroll.offset = 10;

        handle_key(&mut state, &mouse(event::MouseEventKind::ScrollUp));
        assert_eq!(state.task_output_scroll.offset(), 7);
        handle_key(&mut state, &mouse(event::MouseEventKind::ScrollDown));
        assert_eq!(state.task_output_scroll.offset(), 10);
    }

    #[tokio::test]
    async fn output_scroll_clamps_at_the_bottom_and_scrolls_up_immediately() {
        let id = crate::bg::REGISTRY.run("seq 1 30").unwrap();
        for _ in 0..200 {
            if matches!(
                crate::bg::REGISTRY.list().into_iter().find(|t| t.id == id).unwrap().status,
                crate::bg::BgStatus::Finished(_)
            ) {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        let mut state = TuiState::new("model".into());
        state.task_output_id = Some(id);

        for _ in 0..10 {
            handle_key(&mut state, &key(KeyCode::Char('j')));
        }
        assert_eq!(state.task_output_scroll.offset(), 18);
        handle_key(&mut state, &key(KeyCode::Char('k')));
        assert_eq!(state.task_output_scroll.offset(), 15);
        handle_key(&mut state, &key(KeyCode::PageDown));
        assert_eq!(state.task_output_scroll.offset(), 18);
        handle_key(&mut state, &key(KeyCode::PageUp));
        assert_eq!(state.task_output_scroll.offset(), 10);
    }

    #[tokio::test]
    async fn task_output_popup_covers_chat_across_frames() {
        let id = crate::bg::REGISTRY.run("seq 1 30").unwrap();
        for _ in 0..200 {
            if matches!(
                crate::bg::REGISTRY.list().into_iter().find(|t| t.id == id).unwrap().status,
                crate::bg::BgStatus::Finished(_)
            ) {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        let mut state = TuiState::new("model".into());
        for i in 0..5 {
            state.session().renderer.push_user(&format!("CHATLINE-{i}"));
        }

        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|frame| draw(frame, &state, 0)).unwrap();

        state.tasks_open = true;
        state.task_output_id = Some(id);
        state.task_output_scroll.set_following(true);
        terminal.draw(|frame| draw(frame, &state, 0)).unwrap();

        let buffer = terminal.backend().buffer();
        let in_popup: String = (3..21)
            .flat_map(|y| {
                (0..80).map(move |x| buffer.cell((x, y)).unwrap().symbol().to_string())
            })
            .collect();
        assert!(!in_popup.contains("CHATLINE"));
        assert!(in_popup.contains("30"));
    }

    #[test]
    fn frame_shows_the_rename_buffer_on_the_cursor_line() {
        let mut state = TuiState::new("llama".into());
        state.sessions.push(Session::new(1));
        state.sessions[1].label = "old name".into();
        state.picker_open = true;
        state.picker_cursor.set(1);
        state.picker_rename = Some("new na".into());

        let backend = TestBackend::new(80, 12);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|frame| draw(frame, &state, 0)).unwrap();

        let buffer = terminal.backend().buffer();
        let joined: String = (0..12)
            .flat_map(|y| {
                (0..80).map(move |x| buffer.cell((x, y)).unwrap().symbol().to_string())
            })
            .collect();
        assert!(joined.contains("new na"));
        assert!(joined.contains("rename"));
        assert!(!joined.contains("old name"));
    }

    #[test]
    fn session_label_is_set_from_the_first_task() {
        let mut state = TuiState::new("model".into());
        state.session().input = "fix the login bug".into();

        assert_eq!(
            handle_key(&mut state, &key(KeyCode::Enter)),
            KeyAction::Submit("fix the login bug".into())
        );
        assert_eq!(state.session().label, "fix the login bug");

        state.session().input = "another task".into();
        handle_key(&mut state, &key(KeyCode::Enter));
        assert_eq!(state.session().label, "fix the login bug");
    }

    #[test]
    fn frame_renders_the_picker_over_the_main_pane() {
        let mut state = TuiState::new("llama".into());
        state.session().renderer.on_event(text("hello"));
        state.session().renderer.finish();
        state.sessions.push(Session::new(1));
        state.sessions[1].running = true;
        state.picker_open = true;

        let backend = TestBackend::new(80, 12);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|frame| draw(frame, &state, 0)).unwrap();

        let buffer = terminal.backend().buffer();
        let joined: String = (0..12)
            .flat_map(|y| {
                (0..80).map(move |x| buffer.cell((x, y)).unwrap().symbol().to_string())
            })
            .collect();
        assert!(joined.contains("sessions"));
        assert!(joined.contains("working…"));
        assert!(joined.contains("idle"));
        assert!(joined.contains("search"));
        assert!(joined.contains("hello"));
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
                ("  ls".into(), Modifier::empty()),
            ]
        );
    }

    fn colored(lines: &[Line<'static>]) -> Vec<(String, Option<Color>)> {
        lines
            .iter()
            .map(|line| {
                let span = line.spans.first().cloned().unwrap_or_default();
                (span.content.to_string(), span.style.fg)
            })
            .collect()
    }

    #[test]
    fn edit_file_result_renders_colored_diff_with_line_numbers() {
        let diff = "--- a.txt\n+++ a.txt\n@@ -1,3 +1,3 @@\n one\n-two\n+TWO\n three\n";
        let described = colored(&lines(vec![
            AgentEvent::ToolStarted {
                header: "edit_file: a.txt".into(),
                body: None,
            },
            AgentEvent::ToolResult {
                header: "edit_file: a.txt".into(),
                body: diff.into(),
            },
        ]));

        assert!(described.iter().any(|(c, col)| c == "    1   one" && col == &Some(Color::Rgb(128, 128, 128))));
        assert!(described.iter().any(|(c, col)| c == "-   2  two" && col == &Some(Color::Rgb(0xcc, 0x66, 0x66))));
        assert!(described.iter().any(|(c, col)| c == "+   2  TWO" && col == &Some(Color::Rgb(0xb5, 0xbd, 0x68))));
        assert!(described.iter().any(|(c, _)| c == "    3   three"));
        assert!(!described.iter().any(|(c, _)| c.starts_with("--- a.txt")));
        assert!(!described.iter().any(|(c, _)| c.starts_with("@@")));
    }

    #[test]
    fn edit_file_diff_line_numbers_reset_per_hunk() {
        let diff = "--- a.txt\n+++ a.txt\n@@ -1,2 +1,2 @@\n-a\n+b\n@@ -10,2 +10,2 @@\n-x\n+y\n";
        let described = colored(&lines(vec![
            AgentEvent::ToolStarted {
                header: "edit_file: a.txt".into(),
                body: None,
            },
            AgentEvent::ToolResult {
                header: "edit_file: a.txt".into(),
                body: diff.into(),
            },
        ]));

        assert!(described.iter().any(|(c, _)| c == "-   1  a"));
        assert!(described.iter().any(|(c, _)| c == "+   1  b"));
        assert!(described.iter().any(|(c, _)| c == "-  10  x"));
        assert!(described.iter().any(|(c, _)| c == "+  10  y"));
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
                ("narration".into(), Modifier::empty()),
                (" ".into(), Modifier::empty()),
                (" ".into(), Modifier::empty()),
                ("next round".into(), Modifier::empty()),
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
                ("narration".into(), Modifier::empty()),
                (" ".into(), Modifier::empty()),
                (" ".into(), Modifier::empty()),
                ("second ".into(), Modifier::empty()),
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
                (" ".into(), Modifier::empty()),
                ("⚙ read_file: a.txt".into(), Modifier::BOLD),
                ("  line one".into(), Modifier::empty()),
                (" ".into(), Modifier::empty()),
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
                (" ".into(), Modifier::empty()),
                ("⚙ read_file: a.txt".into(), Modifier::BOLD),
                (" ".into(), Modifier::empty()),
                ("⚙ read_file: b.txt".into(), Modifier::BOLD),
                ("  alpha".into(), Modifier::empty()),
                (" ".into(), Modifier::empty()),
                ("  beta".into(), Modifier::empty()),
                (" ".into(), Modifier::empty()),
            ]
        );
    }

    #[test]
    fn enter_submits_the_input() {
        let mut state = TuiState::new("model".into());
        state.session().input = "hello".into();

        assert_eq!(
            handle_key(&mut state, &key(KeyCode::Enter)),
            KeyAction::Submit("hello".into())
        );
        assert!(state.session().input.is_empty());
    }

    #[test]
    fn enter_with_empty_input_does_nothing() {
        let mut state = TuiState::new("model".into());

        assert_eq!(handle_key(&mut state, &key(KeyCode::Enter)), KeyAction::None);
    }

    #[test]
    fn backspace_pops_the_input() {
        let mut state = TuiState::new("model".into());
        state.session().input = "ab".into();

        assert_eq!(handle_key(&mut state, &key(KeyCode::Backspace)), KeyAction::None);
        assert_eq!(state.session().input, "a");
    }

    #[test]
    fn ctrl_c_quits() {
        let mut state = TuiState::new("model".into());
        let event = TermEvent::Key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL));

        assert_eq!(handle_key(&mut state, &event), KeyAction::Quit);
    }

    #[test]
    fn c_t_cycles_the_thinking_level() {
        let mut state = TuiState::new("model".into());
        let event = TermEvent::Key(KeyEvent::new(KeyCode::Char('t'), KeyModifiers::CONTROL));

        assert_eq!(*state.thinking.lock().unwrap(), crate::thinking::ThinkingLevel::Off);
        handle_key(&mut state, &event);
        assert_eq!(*state.thinking.lock().unwrap(), crate::thinking::ThinkingLevel::Low);
        handle_key(&mut state, &event);
        assert_eq!(*state.thinking.lock().unwrap(), crate::thinking::ThinkingLevel::Medium);
        handle_key(&mut state, &event);
        assert_eq!(*state.thinking.lock().unwrap(), crate::thinking::ThinkingLevel::High);
        handle_key(&mut state, &event);
        assert_eq!(*state.thinking.lock().unwrap(), crate::thinking::ThinkingLevel::XHigh);
        handle_key(&mut state, &event);
        assert_eq!(*state.thinking.lock().unwrap(), crate::thinking::ThinkingLevel::Off);
    }

    #[test]
    fn tab_switches_the_mode() {
        let mut state = TuiState::new("model".into());
        let event = TermEvent::Key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE));

        assert_eq!(*state.mode.lock().unwrap(), crate::mode::Mode::Yolo);
        handle_key(&mut state, &event);
        assert_eq!(*state.mode.lock().unwrap(), crate::mode::Mode::Plan);
        handle_key(&mut state, &event);
        assert_eq!(*state.mode.lock().unwrap(), crate::mode::Mode::Yolo);
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
        assert_eq!(span.style.fg, Some(Color::Rgb(212, 212, 212)));
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
        assert_eq!(user.spans[0].style.bg, Some(Color::Rgb(52, 53, 65)));
        assert_eq!(user.spans.last().unwrap().style.bg, Some(Color::Rgb(52, 53, 65)));
        let thinking = &display[4];
        assert_eq!(thinking.spans.last().unwrap().style.bg, None);
        assert_eq!(thinking.spans[0].style.fg, Some(Color::Rgb(128, 128, 128)));
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
    fn a_tool_block_turns_done_when_the_result_arrives() {
        let mut renderer = TuiRenderer::new();
        renderer.on_event(AgentEvent::ToolStarted {
            header: "bash".into(),
            body: Some("ls".into()),
        });
        assert!(renderer.blocks().iter().all(|b| *b == BlockKind::ToolRunning));
        assert_eq!(renderer.blocks().len(), 3);

        renderer.on_event(AgentEvent::ToolResult {
            header: "bash".into(),
            body: "ok".into(),
        });
        assert!(renderer.blocks().iter().all(|b| *b == BlockKind::ToolDone));
        assert_eq!(renderer.blocks().len(), 5);
    }

    #[test]
    fn an_unfinished_tool_block_stays_running() {
        let mut renderer = TuiRenderer::new();
        renderer.on_event(AgentEvent::ToolStarted {
            header: "bash".into(),
            body: None,
        });
        renderer.finish();
        assert!(renderer.blocks().iter().all(|b| *b == BlockKind::ToolRunning));
        assert_eq!(renderer.blocks().len(), 2);
    }

    #[test]
    fn parallel_tool_blocks_each_turn_done_on_their_own_result() {
        let mut renderer = TuiRenderer::new();
        renderer.on_event(AgentEvent::ToolStarted {
            header: "a".into(),
            body: None,
        });
        renderer.on_event(AgentEvent::ToolStarted {
            header: "b".into(),
            body: None,
        });
        renderer.on_event(AgentEvent::ToolResult {
            header: "a".into(),
            body: "alpha".into(),
        });
        assert_eq!(renderer.blocks()[0], BlockKind::ToolDone);
        assert_eq!(renderer.blocks()[2], BlockKind::ToolRunning);
        renderer.on_event(AgentEvent::ToolResult {
            header: "b".into(),
            body: "beta".into(),
        });
        assert!(renderer.blocks().iter().all(|b| *b == BlockKind::ToolDone));
    }

    #[test]
    fn a_trailing_newline_in_a_tool_result_is_not_pushed() {
        let mut renderer = TuiRenderer::new();
        renderer.on_event(AgentEvent::ToolStarted {
            header: "bash".into(),
            body: Some("echo out".into()),
        });
        renderer.on_event(AgentEvent::ToolResult {
            header: "bash".into(),
            body: "out\n".into(),
        });

        assert_eq!(renderer.scrollback().len(), 5);
        assert_eq!(renderer.scrollback()[3].spans[0].content.as_ref(), "  out");
    }

    #[test]
    fn display_lines_color_tool_blocks_by_state() {
        let mut renderer = TuiRenderer::new();
        renderer.on_event(AgentEvent::ToolStarted {
            header: "running".into(),
            body: None,
        });
        renderer.on_event(AgentEvent::ToolStarted {
            header: "done".into(),
            body: None,
        });
        renderer.on_event(AgentEvent::ToolResult {
            header: "done".into(),
            body: "ok".into(),
        });

        let display = display_lines(renderer.scrollback(), renderer.blocks(), 40);
        assert_eq!(display[0].spans.last().unwrap().style.bg, Some(Color::Rgb(40, 40, 50)));
        assert_eq!(display[3].spans.last().unwrap().style.bg, Some(Color::Rgb(40, 50, 40)));
    }

    #[test]
    fn tail_start_accounts_for_wrapped_lines() {
        let mut state = TuiState::new("model".into());
        state.viewport = 4;
        state.pane_width = 20;
        for _ in 0..3 {
            state.session().renderer.on_event(text(&format!("{}\n", "a".repeat(40))));
        }
        state.session().renderer.finish();
        let (pw, vp) = (state.pane_width, state.viewport);

        assert_eq!(state.session().max_scroll(pw, vp), 3);
    }

    #[test]
    fn tail_shows_the_last_line_of_a_large_scrollback() {
        let mut state = TuiState::new("model".into());
        state.viewport = 22;
        state.pane_width = 118;
        for i in 0..30 {
            let width = if i % 3 == 0 { 300 } else { 20 };
            state.session().renderer.on_event(text(&format!("line {} {}\n", i, "x".repeat(width))));
        }
        state.session().renderer.finish();
        state.session().scroller.set_following(true);
        let (pw, vp) = (state.pane_width, state.viewport);
        let max = state.session().max_scroll(pw, vp);
        state.session().scroller.end(max);

        let scrollback = state.session().renderer.scrollback().to_vec();
        let last_line = scrollback.last().unwrap().to_string();
        let scroll = state.session().scroller.offset();
        let mut buffer = Buffer::empty(Rect::new(0, 0, 118, 24));
        Paragraph::new(&scrollback[scroll..])
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
            state.session().renderer.on_event(text("x\ny\nz\nw\n"));
        }
        state.session().renderer.finish();
        state.viewport = 2;
        state.session().scroller.set_following(true);
        let (pw, vp) = (state.pane_width, state.viewport);
        let max = state.session().max_scroll(pw, vp);
        state.session().scroller.end(max);

        assert_eq!(handle_key(&mut state, &key(KeyCode::PageUp)), KeyAction::None);
        assert_eq!(state.session().scroller.offset(), 2);
        assert!(!state.session().scroller.following());
        assert_eq!(handle_key(&mut state, &key(KeyCode::PageUp)), KeyAction::None);
        assert_eq!(state.session().scroller.offset(), 0);
        assert_eq!(handle_key(&mut state, &key(KeyCode::PageDown)), KeyAction::None);
        assert_eq!(state.session().scroller.offset(), 10);
        assert!(!state.session().scroller.following());
        assert_eq!(handle_key(&mut state, &key(KeyCode::PageDown)), KeyAction::None);
        assert_eq!(state.session().scroller.offset(), 12);
        assert!(state.session().scroller.following());
        assert_eq!(handle_key(&mut state, &key(KeyCode::Home)), KeyAction::None);
        assert_eq!(state.session().scroller.offset(), 0);
        assert!(!state.session().scroller.following());
        assert_eq!(handle_key(&mut state, &key(KeyCode::End)), KeyAction::None);
        assert_eq!(state.session().scroller.offset(), max);
        assert!(state.session().scroller.following());
    }

    #[test]
    fn typing_is_ignored_while_running() {
        let mut state = TuiState::new("model".into());
        state.session().running = true;
        state.session().input = "keep".into();

        assert_eq!(handle_key(&mut state, &key(KeyCode::Char('a'))), KeyAction::None);
        assert_eq!(state.session().input, "keep");
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
        let heading = &rendered[1].spans[0];
        assert!(heading.style.add_modifier.contains(Modifier::BOLD));
        assert!(heading.style.add_modifier.contains(Modifier::UNDERLINED));
        assert!(rendered[3].spans[0].style.add_modifier.contains(Modifier::BOLD));
    }

    #[test]
    fn thinking_with_markdown_is_rendered() {
        let rendered = lines(vec![thinking("**bold** thinking"), text("done")]);

        let thinking_line = &rendered[1];
        assert!(thinking_line.spans.iter().any(|s| {
            s.content.as_ref() == "bold"
                && s.style.add_modifier.contains(Modifier::BOLD)
        }));
    }

    #[test]
    fn plain_thinking_is_not_reparsed() {
        let described = describe(&lines(vec![thinking("just thinking"), text("done")]));

        assert_eq!(
            described,
            vec![
                (" ".into(), Modifier::empty()),
                ("just thinking".into(), Modifier::empty()),
                (" ".into(), Modifier::empty()),
                ("done".into(), Modifier::empty()),
                (" ".into(), Modifier::empty()),
            ]
        );
    }

    #[test]
    fn user_input_with_markdown_is_rendered() {
        let mut renderer = TuiRenderer::new();
        renderer.push_user("make it **bold**");
        renderer.finish();

        let line = &renderer.scrollback()[1];
        assert!(line.spans.iter().any(|s| {
            s.content.as_ref() == "bold" && s.style.add_modifier.contains(Modifier::BOLD)
        }));
    }

    #[test]
    fn plain_user_input_is_not_reparsed() {
        let mut renderer = TuiRenderer::new();
        renderer.push_user("just a question");
        renderer.finish();

        let line = &renderer.scrollback()[1];
        assert_eq!(line.spans[0].content.as_ref(), "just a question");
        assert!(line.spans[0].style.add_modifier.contains(Modifier::BOLD));
    }

    #[test]
    fn user_markdown_gets_pi_base_and_plain_bold() {
        let mut renderer = TuiRenderer::new();
        renderer.push_user("plain **bold** and `code`");
        renderer.finish();

        let line = &renderer.scrollback()[1];
        let spans = &line.spans;
        assert_eq!(spans[0].style.fg, Some(Color::Rgb(212, 212, 212)));
        let bold_span = spans.iter().find(|s| s.content.as_ref() == "bold").unwrap();
        assert_eq!(bold_span.style.fg, Some(Color::Rgb(212, 212, 212)));
        assert!(bold_span.style.add_modifier.contains(Modifier::BOLD));
        let code_span = spans.iter().find(|s| s.content.as_ref() == "code").unwrap();
        assert_eq!(code_span.style.fg, Some(Color::Rgb(138, 190, 183)));
        assert_eq!(code_span.style.bg, None);
    }

    #[test]
    fn thinking_markdown_gets_black_base_and_black_bold() {
        let rendered = lines(vec![thinking("plain **bold**"), text("done")]);

        let line = &rendered[1];
        let spans = &line.spans;
        assert_eq!(spans[0].style.fg, Some(Color::Rgb(128, 128, 128)));
        let bold_span = spans.iter().find(|s| s.content.as_ref() == "bold").unwrap();
        assert_eq!(bold_span.style.fg, Some(Color::Rgb(128, 128, 128)));
        assert!(bold_span.style.add_modifier.contains(Modifier::BOLD));
    }

    #[test]
    fn headings_render_pi_style_without_markers() {
        let rendered = lines(vec![text("# One\n\n## Two"), AgentEvent::CompletionStarted]);

        let one = rendered.iter().find(|l| l.spans.iter().any(|s| s.content.as_ref() == "One")).unwrap();
        let one_span = one.spans.iter().find(|s| s.content.as_ref() == "One").unwrap();
        assert_eq!(one_span.style.fg, Some(Color::Rgb(0xf0, 0xc6, 0x74)));
        assert!(one_span.style.add_modifier.contains(Modifier::BOLD));
        assert!(one_span.style.add_modifier.contains(Modifier::UNDERLINED));
        assert!(!one.to_string().contains('#'));

        let two = rendered.iter().find(|l| l.spans.iter().any(|s| s.content.as_ref() == "Two")).unwrap();
        let two_span = two.spans.iter().find(|s| s.content.as_ref() == "Two").unwrap();
        assert_eq!(two_span.style.fg, Some(Color::Rgb(0xf0, 0xc6, 0x74)));
        assert!(two_span.style.add_modifier.contains(Modifier::BOLD));
        assert!(!two_span.style.add_modifier.contains(Modifier::UNDERLINED));
    }

    #[test]
    fn links_render_pi_style() {
        let rendered = lines(vec![text("[label](https://example.com)"), AgentEvent::CompletionStarted]);

        let label = rendered
            .iter()
            .find(|l| l.spans.iter().any(|s| s.content.as_ref() == "label"))
            .unwrap();
        let label_span = label.spans.iter().find(|s| s.content.as_ref() == "label").unwrap();
        assert_eq!(label_span.style.fg, Some(Color::Rgb(0x81, 0xa2, 0xbe)));
        assert!(label_span.style.add_modifier.contains(Modifier::UNDERLINED));
    }

    #[test]
    fn quotes_render_gray_italic() {
        let rendered = lines(vec![text("> quoted"), AgentEvent::CompletionStarted]);

        let quoted = rendered
            .iter()
            .find(|l| l.spans.iter().any(|s| s.content.as_ref() == "quoted"))
            .unwrap();
        let quoted_span = quoted.spans.iter().find(|s| s.content.as_ref() == "quoted").unwrap();
        assert_eq!(quoted_span.style.fg, Some(Color::Rgb(128, 128, 128)));
        assert!(quoted_span.style.add_modifier.contains(Modifier::ITALIC));
    }

    #[test]
    fn markdown_rendering_strips_vs16() {
        let mut renderer = TuiRenderer::new();
        renderer.push_user("icon \u{23F8}\u{FE0F} note");
        renderer.finish();

        for line in renderer.scrollback() {
            for span in &line.spans {
                assert!(!span.content.to_string().contains('\u{FE0F}'));
            }
        }
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
                (" ".into(), Modifier::empty()),
                ("⚙ bash".into(), Modifier::BOLD),
                ("  **cmd**".into(), Modifier::empty()),
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
    fn replay_context_renders_the_transcript() {
        use openai_oxide::types::chat::{FunctionCall, ToolCall};
        let mut renderer = TuiRenderer::new();
        let mut context = Context::new("sys", 100);
        context.messages.push(Message::User { content: "do the thing".into() });
        context.messages.push(Message::Assistant {
            content: None,
            tool_calls: vec![ToolCall {
                id: "call-1".into(),
                type_: "function".into(),
                function: FunctionCall {
                    name: "bash".into(),
                    arguments: r#"{"command":"ls"}"#.into(),
                },
            }],
        });
        context.messages.push(Message::Tool {
            tool_call_id: "call-1".into(),
            content: "file.txt".into(),
        });
        context.messages.push(Message::Assistant {
            content: Some("done".into()),
            tool_calls: vec![],
        });

        renderer.replay_context(&context);

        let described = describe(&renderer.scrollback());
        assert!(described.iter().any(|(c, _)| c.contains("do the thing")));
        assert!(described.iter().any(|(c, _)| c.contains("bash")));
        assert!(described.iter().any(|(c, _)| c.contains("ls")));
        assert!(described.iter().any(|(c, _)| c.contains("file.txt")));
        assert!(described.iter().any(|(c, _)| c.contains("done")));
    }

    #[test]
    fn with_sessions_restores_and_falls_back() {
        let mut context = Context::new("sys", 100);
        context.messages.push(Message::User { content: "hello".into() });
        context.total_prompt_tokens = 50;
        context.total_completion_tokens = 5;
        let file = SessionFile {
            label: "fix login".into(),
            context: context.clone(),
        };

        let state = TuiState::with_sessions("model".into(), vec![(3, file)]);
        assert_eq!(state.sessions.len(), 2);
        assert_eq!(state.sessions[0].label, "fix login");
        assert_eq!(state.sessions[0].context.as_ref().unwrap().messages.len(), 1);
        assert_eq!(state.sessions[0].prompt_tokens, 50);
        assert_eq!(state.sessions[0].completion_tokens, 5);
        assert_eq!(state.sessions[1].label, "");
        assert!(state.sessions[1].context.is_none());
        assert_eq!(state.active, 1);
        assert_eq!(state.next_id, 5);

        let state = TuiState::with_sessions("model".into(), vec![]);
        assert_eq!(state.sessions.len(), 1);
        assert_eq!(state.sessions[0].id, 0);
        assert_eq!(state.active, 0);
        assert_eq!(state.next_id, 1);
    }

    #[test]
    fn frame_renders_status_main_and_input() {
        let mut state = TuiState::new("llama".into());
        state.session().renderer.on_event(text("hello"));
        state.session().renderer.finish();
        state.session().prompt_tokens = 100;
        state.session().completion_tokens = 5;

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
        assert!(joined.contains("[1/1]"));
        assert!(joined.contains("hello"));
        assert!(joined.contains(">"));
    }
}
