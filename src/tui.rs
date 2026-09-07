use std::collections::HashMap;
use std::io::Read as _;
use std::io::Write as _;
use std::mem;
use std::path::Path;
use std::sync::{Arc, Mutex};

use bitflags::bitflags;
use ratatui::backend::TermwizBackend;
use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Widget, Wrap};
use ratatui::{Frame, Terminal};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum KeyCode {
    Char(char),
    Enter,
    Backspace,
    Insert,
    Delete,
    Left,
    Right,
    Up,
    Down,
    Home,
    End,
    PageUp,
    PageDown,
    Tab,
    Esc,
    F(u8),
}

bitflags! {
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub struct KeyModifiers: u8 {
        const NONE = 0;
        const SHIFT = 1;
        const CONTROL = 2;
        const ALT = 4;
        const SUPER = 8;
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct KeyEvent {
    pub code: KeyCode,
    pub modifiers: KeyModifiers,
}

impl KeyEvent {
    fn new(code: KeyCode, modifiers: KeyModifiers) -> Self {
        Self { code, modifiers }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum MouseEventKind {
    ScrollUp,
    ScrollDown,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct MouseEvent {
    pub kind: MouseEventKind,
}

#[derive(Clone, PartialEq, Eq)]
pub enum TermEvent {
    Key(KeyEvent),
    Mouse(MouseEvent),
    Paste(String),
}

use crate::agent::Agent;
use crate::context::Context;
use crate::mode::Mode;
use crate::paths::CONTEXT_DIR;
use crate::session::{Cursor, QuestionState, Scroller, Session};
use crate::session_store::{self, SessionFile};
use crate::stream::AgentEvent;
use crate::thinking::ThinkingLevel;
use crate::transcript::BlockKind;

pub enum TuiEvent {
    Agent { session: u64, event: AgentEvent },
    TurnDone { session: u64, context: Context },
    TurnError { session: u64, message: String },
    GatePending { session: u64, message: String },
    StageChanged { session: u64, mode: Option<Mode> },
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
            session.history = file.history;
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

fn map_termwiz_event(event: termwiz::input::InputEvent) -> Option<TermEvent> {
    use termwiz::input::{Modifiers, KeyCode as TermwizKeyCode};
    match event {
        termwiz::input::InputEvent::Key(key_event) => {
            let code = match key_event.key {
                TermwizKeyCode::Char('\r') => KeyCode::Enter,
                TermwizKeyCode::Char(c) => KeyCode::Char(c),
                TermwizKeyCode::Enter => KeyCode::Enter,
                TermwizKeyCode::Escape => KeyCode::Esc,
                TermwizKeyCode::Backspace => KeyCode::Backspace,
                TermwizKeyCode::Tab => KeyCode::Tab,
                TermwizKeyCode::PageUp => KeyCode::PageUp,
                TermwizKeyCode::PageDown => KeyCode::PageDown,
                TermwizKeyCode::End => KeyCode::End,
                TermwizKeyCode::Home => KeyCode::Home,
                TermwizKeyCode::Insert => KeyCode::Insert,
                TermwizKeyCode::Delete => KeyCode::Delete,
                TermwizKeyCode::LeftArrow => KeyCode::Left,
                TermwizKeyCode::RightArrow => KeyCode::Right,
                TermwizKeyCode::UpArrow => KeyCode::Up,
                TermwizKeyCode::DownArrow => KeyCode::Down,
                TermwizKeyCode::Function(n) => KeyCode::F(n),
                _ => return None,
            };
            let mut modifiers = KeyModifiers::NONE;
            if key_event.modifiers.contains(Modifiers::SHIFT) {
                modifiers |= KeyModifiers::SHIFT;
            }
            if key_event.modifiers.contains(Modifiers::ALT) {
                modifiers |= KeyModifiers::ALT;
            }
            if key_event.modifiers.contains(Modifiers::CTRL) {
                modifiers |= KeyModifiers::CONTROL;
            }
            if key_event.modifiers.contains(Modifiers::SUPER) {
                modifiers |= KeyModifiers::SUPER;
            }
            Some(TermEvent::Key(KeyEvent::new(code, modifiers)))
        }
        termwiz::input::InputEvent::Mouse(mouse) => {
            use termwiz::input::MouseButtons;
            if !mouse.mouse_buttons.contains(MouseButtons::VERT_WHEEL) {
                return None;
            }
            let kind = if mouse.mouse_buttons.contains(MouseButtons::WHEEL_POSITIVE) {
                MouseEventKind::ScrollUp
            } else {
                MouseEventKind::ScrollDown
            };
            Some(TermEvent::Mouse(MouseEvent { kind }))
        }
        termwiz::input::InputEvent::Paste(text) => Some(TermEvent::Paste(text)),
        _ => None,
    }
}

pub fn handle_key(state: &mut TuiState, event: &TermEvent) -> KeyAction {
    if let TermEvent::Mouse(mouse) = event {
        if state.task_output_id.is_some() {
            let max = task_output_scroll_max(state);
            return match mouse.kind {
                MouseEventKind::ScrollUp => {
                    state.task_output_scroll.toward_top(3);
                    KeyAction::None
                }
                MouseEventKind::ScrollDown => {
                    state.task_output_scroll.toward_bottom(3, max);
                    KeyAction::None
                }
            };
        }
        let (pane_width, viewport) = (state.pane_width, state.viewport);
        let session = state.session();
        return match mouse.kind {
            MouseEventKind::ScrollUp => {
                mouse_up(session);
                KeyAction::None
            }
            MouseEventKind::ScrollDown => {
                mouse_down(session, pane_width, viewport);
                KeyAction::None
            }
        };
    }
    let TermEvent::Key(key) = event else {
        if let TermEvent::Paste(text) = event {
            let session = &mut state.sessions[state.active];
            if !session.running {
                let mut chars: Vec<char> = session.input.chars().collect();
                for c in text.chars() {
                    chars.insert(session.input_cursor, c);
                    session.input_cursor += 1;
                }
                session.input = chars.into_iter().collect();
                session.history_index = None;
                session.error = None;
            }
            return KeyAction::None;
        }
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
        let session = state.session();
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
            KeyCode::Left => {
                if !session.running {
                    session.input_cursor = word_left(&session.input, session.input_cursor);
                }
                KeyAction::None
            }
            KeyCode::Right => {
                if !session.running {
                    session.input_cursor = word_right(&session.input, session.input_cursor);
                }
                KeyAction::None
            }
            KeyCode::Enter | KeyCode::Char('j') => {
                if !session.running {
                    insert_newline(session);
                }
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
    if state.session().question.is_some() {
        return handle_question_key(state, key);
    }
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
            KeyCode::Home => {
                to_top(session);
                KeyAction::None
            }
            KeyCode::End => {
                to_bottom(session, pane_width, viewport);
                KeyAction::None
            }
            _ => KeyAction::None,
        };
    }

    match key.code {
        KeyCode::Enter => {
            if key
                .modifiers
                .intersects(KeyModifiers::SHIFT | KeyModifiers::ALT)
            {
                insert_newline(session);
                KeyAction::None
            } else if session.input.is_empty() && !session.gate {
                KeyAction::None
            } else {
                let task = mem::take(&mut session.input);
                session.input_cursor = 0;
                if !task.is_empty() {
                    session.history.push(task.clone());
                    if session.label.is_empty() {
                        session.label = task
                            .lines()
                            .find(|line| !line.trim().is_empty())
                            .map(|line| line.trim().chars().take(24).collect())
                            .unwrap_or_default();
                    }
                }
                session.history_index = None;
                session.scroller.set_following(true);
                KeyAction::Submit(task)
            }
        }
        KeyCode::Backspace => {
            if session.input_cursor > 0 {
                session.input_cursor -= 1;
                let mut chars: Vec<char> = session.input.chars().collect();
                chars.remove(session.input_cursor);
                session.input = chars.into_iter().collect();
                session.history_index = None;
                session.error = None;
            }
            KeyAction::None
        }
        KeyCode::Left => {
            session.input_cursor = session.input_cursor.saturating_sub(1);
            KeyAction::None
        }
        KeyCode::Right => {
            session.input_cursor = (session.input_cursor + 1).min(session.input.chars().count());
            KeyAction::None
        }
        KeyCode::Up | KeyCode::Down => {
            let chars: Vec<char> = session.input.chars().collect();
            if input_visual_ranges(&chars, pane_width).len() > 1 && session.history_index.is_none() {
                move_cursor_visual_line(
                    &session.input,
                    &mut session.input_cursor,
                    pane_width,
                    key.code == KeyCode::Up,
                );
            } else if key.code == KeyCode::Up {
                match session.history_index {
                    Some(0) => {}
                    Some(i) => {
                        let entry = session.history[i - 1].clone();
                        session.history_index = Some(i - 1);
                        session.input = entry;
                        session.input_cursor = session.input.chars().count();
                        session.error = None;
                    }
                    None => {
                        if session.input.is_empty() && !session.history.is_empty() {
                            let entry = session.history.last().unwrap().clone();
                            session.history_index = Some(session.history.len() - 1);
                            session.input = entry;
                            session.input_cursor = session.input.chars().count();
                            session.error = None;
                        }
                    }
                }
            } else {
                match session.history_index {
                    Some(i) if i + 1 < session.history.len() => {
                        let entry = session.history[i + 1].clone();
                        session.history_index = Some(i + 1);
                        session.input = entry;
                        session.input_cursor = session.input.chars().count();
                        session.error = None;
                    }
                    Some(_) => {
                        session.history_index = None;
                        session.input.clear();
                        session.input_cursor = 0;
                        session.error = None;
                    }
                    None => {}
                }
            }
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
            let mut chars: Vec<char> = session.input.chars().collect();
            chars.insert(session.input_cursor, c);
            session.input = chars.into_iter().collect();
            session.input_cursor += 1;
            session.history_index = None;
            session.error = None;
            KeyAction::None
        }
        _ => KeyAction::None,
    }
}

fn word_left(input: &str, cursor: usize) -> usize {
    let chars: Vec<char> = input.chars().collect();
    let mut pos = cursor;
    while pos > 0 && chars[pos - 1].is_whitespace() {
        pos -= 1;
    }
    while pos > 0 && !chars[pos - 1].is_whitespace() {
        pos -= 1;
    }
    pos
}

fn word_right(input: &str, cursor: usize) -> usize {
    let chars: Vec<char> = input.chars().collect();
    let len = chars.len();
    let mut pos = cursor;
    while pos < len && !chars[pos].is_whitespace() {
        pos += 1;
    }
    while pos < len && chars[pos].is_whitespace() {
        pos += 1;
    }
    pos
}

fn insert_newline(session: &mut Session) {
    let mut chars: Vec<char> = session.input.chars().collect();
    chars.insert(session.input_cursor, '\n');
    session.input = chars.into_iter().collect();
    session.input_cursor += 1;
    session.history_index = None;
    session.error = None;
}

fn finish_question(state: &mut TuiState) {
    let session = &mut state.sessions[state.active];
    let Some(mut q) = session.question.take() else {
        return;
    };
    while q.answers.len() < q.questions.len() {
        q.answers.push("cancelled".to_string());
    }
    session.question = None;
    if let Some(tx) = q.reply.take() {
        let _ = tx.send(Some(q.answers));
    }
}

fn handle_question_key(state: &mut TuiState, key: &KeyEvent) -> KeyAction {
    let session = &mut state.sessions[state.active];
    let Some(q) = session.question.as_mut() else {
        return KeyAction::None;
    };
    let kind = q.questions[q.step].kind.clone();
    let option_count = q.questions[q.step].options.len();
    match key.code {
        KeyCode::Up => {
            q.cursor = q.cursor.saturating_sub(1);
        }
        KeyCode::Down => {
            q.cursor = (q.cursor + 1).min(option_count.saturating_sub(1));
        }
        KeyCode::Char(' ') if matches!(kind, crate::tool::QuestionKind::MultiChoice) => {
            if q.cursor < q.selected.len() {
                q.selected[q.cursor] = !q.selected[q.cursor];
            }
        }
        KeyCode::Char(c) => {
            let mut chars: Vec<char> = q.draft.chars().collect();
            chars.insert(q.draft_cursor, c);
            q.draft = chars.into_iter().collect();
            q.draft_cursor += 1;
        }
        KeyCode::Backspace => {
            if q.draft_cursor > 0 {
                let mut chars: Vec<char> = q.draft.chars().collect();
                chars.remove(q.draft_cursor - 1);
                q.draft = chars.into_iter().collect();
                q.draft_cursor -= 1;
            }
        }
        KeyCode::Left => {
            q.draft_cursor = q.draft_cursor.saturating_sub(1);
        }
        KeyCode::Right => {
            q.draft_cursor = (q.draft_cursor + 1).min(q.draft.chars().count());
        }
        KeyCode::Home => {
            q.draft_cursor = 0;
        }
        KeyCode::End => {
            q.draft_cursor = q.draft.chars().count();
        }
        KeyCode::Esc => {
            finish_question(state);
        }
        KeyCode::Enter => {
            let answer = match kind {
                crate::tool::QuestionKind::Free => {
                    if q.draft.is_empty() {
                        return KeyAction::None;
                    }
                    q.draft.clone()
                }
                crate::tool::QuestionKind::SingleChoice => {
                    if !q.draft.is_empty() {
                        q.draft.clone()
                    } else {
                        q.questions[q.step].options[q.cursor].clone()
                    }
                }
                crate::tool::QuestionKind::MultiChoice => {
                    let picked: Vec<String> = q.selected
                        .iter()
                        .enumerate()
                        .filter(|(_, selected)| **selected)
                        .map(|(i, _)| q.questions[q.step].options[i].clone())
                        .collect();
                    if picked.is_empty() && q.draft.is_empty() {
                        return KeyAction::None;
                    }
                    let mut parts = picked;
                    if !q.draft.is_empty() {
                        parts.push(q.draft.clone());
                    }
                    parts.join(", ")
                }
            };
            q.answers.push(answer);
            if q.step + 1 < q.questions.len() {
                q.step += 1;
                q.reset_step();
            } else {
                finish_question(state);
            }
        }
        _ => {}
    }
    KeyAction::None
}

#[cfg(test)]
struct InputParser {
    pending: Vec<u8>,
}

#[cfg(test)]
impl InputParser {
    fn new() -> Self {
        Self { pending: Vec::new() }
    }

    fn feed(&mut self, chunk: &[u8]) -> Vec<TermEvent> {
        self.pending.extend_from_slice(chunk);
        let mut events = Vec::new();
        while let Some(event) = self.next_event() {
            events.push(event);
        }
        events
    }

    fn flush_escape(&mut self) -> Vec<TermEvent> {
        if self.pending == [0x1Bu8] {
            self.pending.clear();
            vec![TermEvent::Key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE))]
        } else {
            Vec::new()
        }
    }

    fn next_event(&mut self) -> Option<TermEvent> {
        if self.pending.is_empty() {
            return None;
        }
        if self.pending[0] == 0x1B {
            self.parse_escape()
        } else {
            self.parse_byte()
        }
    }

    fn control_byte(&self, b: u8) -> Option<(KeyCode, KeyModifiers)> {
        Some(match b {
            0x0D => (KeyCode::Enter, KeyModifiers::NONE),
            0x0A => (KeyCode::Char('j'), KeyModifiers::CONTROL),
            0x09 => (KeyCode::Tab, KeyModifiers::NONE),
            0x7F => (KeyCode::Backspace, KeyModifiers::NONE),
            0x00 => (KeyCode::Char(' '), KeyModifiers::CONTROL),
            0x01..=0x1A => (
                KeyCode::Char((b - 0x01 + b'a') as char),
                KeyModifiers::CONTROL,
            ),
            0x1C..=0x1F => (
                KeyCode::Char((b - 0x1C + b'4') as char),
                KeyModifiers::CONTROL,
            ),
            _ => return None,
        })
    }

    fn parse_byte(&mut self) -> Option<TermEvent> {
        let b = self.pending[0];
        if b >= 0x80 {
            let len = match b & 0xF8 {
                0xF0 => 4,
                0xE0 => 3,
                0xC0 => 2,
                _ => 1,
            };
            if self.pending.len() < len {
                return None;
            }
            let bytes = self.pending[..len].to_vec();
            self.pending.drain(..len);
            let c = String::from_utf8(bytes).ok()?.chars().next()?;
            let modifiers = if c.is_uppercase() {
                KeyModifiers::SHIFT
            } else {
                KeyModifiers::NONE
            };
            return Some(TermEvent::Key(KeyEvent::new(KeyCode::Char(c), modifiers)));
        }
        self.pending.remove(0);
        if let Some((code, modifiers)) = self.control_byte(b) {
            return Some(TermEvent::Key(KeyEvent::new(code, modifiers)));
        }
        let c = b as char;
        let modifiers = if c.is_uppercase() {
            KeyModifiers::SHIFT
        } else {
            KeyModifiers::NONE
        };
        Some(TermEvent::Key(KeyEvent::new(KeyCode::Char(c), modifiers)))
    }

    fn parse_escape(&mut self) -> Option<TermEvent> {
        if self.pending.len() == 1 {
            return None;
        }
        match self.pending[1] {
            0x1B => {
                self.pending.drain(..2);
                Some(TermEvent::Key(KeyEvent::new(KeyCode::Esc, KeyModifiers::ALT)))
            }
            b'[' => self.parse_csi(),
            b'O' => self.parse_ss3(),
            c @ 0x20..=0x7E => {
                self.pending.drain(..2);
                let c = c as char;
                let modifiers = if c.is_uppercase() {
                    KeyModifiers::ALT | KeyModifiers::SHIFT
                } else {
                    KeyModifiers::ALT
                };
                Some(TermEvent::Key(KeyEvent::new(KeyCode::Char(c), modifiers)))
            }
            c @ 0x00..=0x1F => {
                self.pending.drain(..2);
                let (code, modifiers) = self.control_byte(c)?;
                Some(TermEvent::Key(KeyEvent::new(code, modifiers | KeyModifiers::ALT)))
            }
            _ => {
                self.pending.remove(0);
                self.next_event()
            }
        }
    }

    fn parse_ss3(&mut self) -> Option<TermEvent> {
        if self.pending.len() < 3 {
            return None;
        }
        let b = self.pending[2];
        self.pending.drain(..3);
        Some(match b {
            b'A' => TermEvent::Key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE)),
            b'B' => TermEvent::Key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE)),
            b'C' => TermEvent::Key(KeyEvent::new(KeyCode::Right, KeyModifiers::NONE)),
            b'D' => TermEvent::Key(KeyEvent::new(KeyCode::Left, KeyModifiers::NONE)),
            b'H' => TermEvent::Key(KeyEvent::new(KeyCode::Home, KeyModifiers::NONE)),
            b'F' => TermEvent::Key(KeyEvent::new(KeyCode::End, KeyModifiers::NONE)),
            b'P'..=b'S' => TermEvent::Key(KeyEvent::new(KeyCode::F(1 + (b - b'P')), KeyModifiers::NONE)),
            _ => return None,
        })
    }

    fn parse_csi(&mut self) -> Option<TermEvent> {
        if self.pending.len() > 32 {
            self.pending.clear();
            return None;
        }
        let final_idx = self.pending[2..]
            .iter()
            .position(|&b| (0x40..=0x7E).contains(&b))?;
        let total = 2 + final_idx + 1;
        if self.pending.len() < total {
            return None;
        }
        let seq = String::from_utf8_lossy(&self.pending[2..total]).into_owned();
        self.pending.drain(..total);
        self.parse_csi_seq(&seq)
    }

    fn parse_csi_seq(&self, seq: &str) -> Option<TermEvent> {
        let bytes = seq.as_bytes();
        let final_byte = bytes[bytes.len() - 1];
        let params = &seq[..seq.len() - 1];
        match final_byte {
            b'~' => {
                if let Some(rest) = params.strip_prefix("27;") {
                    let (mod_s, key_s) = rest.split_once(';')?;
                    let modifier = mod_to_flags_xterm(mod_s.parse().ok()?);
                    let code = codepoint_to_keycode(key_s.parse().ok()?)?;
                    Some(TermEvent::Key(KeyEvent::new(code, modifier)))
                } else {
                    let n: u32 = params.parse().ok()?;
                    let code = match n {
                        1 | 7 => KeyCode::Home,
                        2 => KeyCode::Insert,
                        3 => KeyCode::Delete,
                        4 | 8 => KeyCode::End,
                        5 => KeyCode::PageUp,
                        6 => KeyCode::PageDown,
                        11..=15 => KeyCode::F((n - 10) as u8),
                        17..=21 => KeyCode::F((n - 11) as u8),
                        23..=26 => KeyCode::F((n - 12) as u8),
                        28..=29 => KeyCode::F((n - 15) as u8),
                        31..=34 => KeyCode::F((n - 17) as u8),
                        _ => return None,
                    };
                    Some(TermEvent::Key(KeyEvent::new(code, KeyModifiers::NONE)))
                }
            }
            b'u' => {
                if params.starts_with('?') {
                    return None;
                }
                let mut parts = params.split(';');
                let code = codepoint_to_keycode(parts.next()?.parse().ok()?)?;
                let modifier = parts
                    .next()
                    .and_then(|s| s.parse::<u8>().ok())
                    .unwrap_or(1);
                Some(TermEvent::Key(KeyEvent::new(code, mod_to_flags(modifier))))
            }
            b'A' | b'B' | b'C' | b'D' => {
                let code = match final_byte {
                    b'A' => KeyCode::Up,
                    b'B' => KeyCode::Down,
                    b'C' => KeyCode::Right,
                    _ => KeyCode::Left,
                };
                let modifier = params
                    .split(';')
                    .nth(1)
                    .and_then(|s| s.parse::<u8>().ok())
                    .unwrap_or(1);
                Some(TermEvent::Key(KeyEvent::new(code, mod_to_flags_xterm(modifier))))
            }
            b'M' | b'm' => {
                let button = params
                    .trim_start_matches('<')
                    .split(';')
                    .next()?
                    .parse::<u32>()
                    .ok()?;
                let kind = match button {
                    64 => MouseEventKind::ScrollUp,
                    65 => MouseEventKind::ScrollDown,
                    _ => return None,
                };
                Some(TermEvent::Mouse(MouseEvent { kind }))
            }
            _ => None,
        }
    }
}

#[cfg(test)]
fn mod_to_flags(modifier: u8) -> KeyModifiers {
    match modifier {
        2 => KeyModifiers::SHIFT,
        3 => KeyModifiers::ALT,
        4 => KeyModifiers::SHIFT | KeyModifiers::ALT,
        5 => KeyModifiers::CONTROL,
        6 => KeyModifiers::SHIFT | KeyModifiers::CONTROL,
        7 => KeyModifiers::ALT | KeyModifiers::CONTROL,
        8 => KeyModifiers::SHIFT | KeyModifiers::ALT | KeyModifiers::CONTROL,
        _ => KeyModifiers::NONE,
    }
}

#[cfg(test)]
fn mod_to_flags_xterm(modifier: u8) -> KeyModifiers {
    match modifier {
        2 => KeyModifiers::SHIFT,
        3 => KeyModifiers::ALT,
        4 => KeyModifiers::CONTROL,
        5 => KeyModifiers::SHIFT | KeyModifiers::ALT,
        6 => KeyModifiers::SHIFT | KeyModifiers::CONTROL,
        7 => KeyModifiers::ALT | KeyModifiers::CONTROL,
        8 => KeyModifiers::SHIFT | KeyModifiers::ALT | KeyModifiers::CONTROL,
        _ => KeyModifiers::NONE,
    }
}

#[cfg(test)]
fn codepoint_to_keycode(codepoint: u32) -> Option<KeyCode> {
    Some(match codepoint {
        3 | 13 | 57414 => KeyCode::Enter,
        9 => KeyCode::Tab,
        27 => KeyCode::Esc,
        127 => KeyCode::Backspace,
        32..=126 => KeyCode::Char(codepoint as u8 as char),
        57417 => KeyCode::Left,
        57418 => KeyCode::Right,
        57419 => KeyCode::Up,
        57420 => KeyCode::Down,
        57421 => KeyCode::PageUp,
        57422 => KeyCode::PageDown,
        57423 => KeyCode::Home,
        57424 => KeyCode::End,
        57425 => KeyCode::Insert,
        57426 => KeyCode::Delete,
        57376..=57395 => KeyCode::F((codepoint - 57376 + 13) as u8),
        _ => return None,
    })
}

fn input_visual_ranges(chars: &[char], width: usize) -> Vec<(usize, usize)> {
    let width = width.max(1);
    let len = chars.len();
    let mut ranges = Vec::new();
    let mut i = 0;
    while i <= len {
        let end = (i..len).find(|&j| chars[j] == '\n').unwrap_or(len);
        let mut start = i;
        loop {
            let take = (start + width).min(end);
            ranges.push((start, take - start));
            if take == end {
                break;
            }
            start = take;
        }
        i = end + 1;
    }
    ranges
}

fn input_box_lines(input: &str, width: usize) -> usize {
    let chars: Vec<char> = input.chars().collect();
    input_visual_ranges(&chars, width).len().clamp(1, 12)
}

fn wrapped_text(text: &str, width: usize) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    input_visual_ranges(&chars, width)
        .iter()
        .map(|(start, count)| chars[*start..*start + *count].iter().collect())
        .collect()
}

fn question_box_lines(q: &QuestionState, width: usize) -> usize {
    let current = &q.questions[q.step];
    let prompt_chars: Vec<char> = current.prompt.chars().collect();
    let prompt_lines = input_visual_ranges(&prompt_chars, width).len().max(1);
    let option_lines = if current.kind == crate::tool::QuestionKind::Free {
        0
    } else {
        current.options.len()
    };
    1 + prompt_lines + option_lines + 1 + 1
}

fn active_box_lines(session: &Session, width: usize) -> usize {
    if let Some(q) = &session.question {
        question_box_lines(q, width)
    } else {
        input_box_lines(&session.input, width)
    }
}

fn cursor_visual_line(chars: &[char], cursor: usize, width: usize) -> usize {
    input_visual_ranges(chars, width)
        .iter()
        .position(|(start, count)| cursor >= *start && cursor <= start + count)
        .unwrap_or(0)
}

fn clamp_input_scroll(session: &mut Session, width: usize) {
    let chars: Vec<char> = session.input.chars().collect();
    let total = input_visual_ranges(&chars, width).len();
    let visible = total.min(12);
    let max_scroll = total - visible;
    session.input_scroll = session.input_scroll.min(max_scroll);
    let line = cursor_visual_line(&chars, session.input_cursor, width);
    if line < session.input_scroll {
        session.input_scroll = line;
    } else if line >= session.input_scroll + visible {
        session.input_scroll = line - visible + 1;
    }
}

fn input_visual_lines(
    input: &str,
    cursor: usize,
    width: usize,
    offset: usize,
) -> Vec<Line<'static>> {
    let width = width.max(1);
    let chars: Vec<char> = input.chars().collect();
    let len = chars.len();
    let cursor_style = Style::default()
        .bg(Color::Rgb(0xd4, 0xd4, 0xd4))
        .fg(Color::Rgb(0x28, 0x28, 0x32));
    let mut lines: Vec<Line<'static>> = Vec::new();
    let mut visual = 0;
    let mut i = 0;
    while i <= len && lines.len() < 12 {
        let end = (i..len).find(|&j| chars[j] == '\n').unwrap_or(len);
        let mut start = i;
        loop {
            let take = (start + width).min(end);
            if visual >= offset && lines.len() < 12 {
                let count = take - start;
                let mut spans: Vec<Span<'static>> = Vec::new();
                for gi in start..take {
                    let span = if gi == cursor {
                        Span::styled(chars[gi].to_string(), cursor_style)
                    } else {
                        Span::raw(chars[gi].to_string())
                    };
                    spans.push(span);
                }
                let full = count == width;
                let newline_marker = take == end && cursor == end && end < len;
                let trailing_block = cursor == len && end == len && take == end;
                if (newline_marker || trailing_block) && full && count > 0 {
                    if let Some(span) = spans.last_mut() {
                        *span = Span::styled(span.content.to_string(), cursor_style);
                    }
                } else if newline_marker {
                    spans.push(Span::styled("⏎".to_string(), cursor_style));
                } else if trailing_block {
                    spans.push(Span::styled(
                        "█".to_string(),
                        Style::default()
                            .bg(Color::Rgb(0xd4, 0xd4, 0xd4))
                            .fg(Color::Rgb(0xd4, 0xd4, 0xd4)),
                    ));
                }
                lines.push(Line::from(spans));
            }
            visual += 1;
            if take == end {
                break;
            }
            start = take;
        }
        i = end + 1;
    }
    lines
}

fn question_wizard_lines(q: &QuestionState, width: usize) -> Vec<Line<'static>> {
    use crate::tool::QuestionKind;
    let current = &q.questions[q.step];
    let total = q.questions.len();
    let cursor_style = Style::default()
        .bg(Color::Rgb(0xd4, 0xd4, 0xd4))
        .fg(Color::Rgb(0x28, 0x28, 0x32));
    let mut lines = vec![Line::from(Span::styled(
        format!(" ❓ question {} of {}", q.step + 1, total),
        Style::default().bold().fg(Color::Rgb(0xe5, 0xb5, 0x67)),
    ))];
    for wrapped in wrapped_text(&current.prompt, width) {
        lines.push(Line::from(Span::raw(wrapped)));
    }
    if current.kind != QuestionKind::Free {
        for (i, option) in current.options.iter().enumerate() {
            let hovered = i == q.cursor;
            let (marker, style) = if hovered {
                ("▸", cursor_style)
            } else {
                (" ", Style::default())
            };
            match current.kind {
                QuestionKind::SingleChoice => {
                    lines.push(Line::from(vec![
                        Span::styled(marker.to_string(), style),
                        Span::raw(format!(" {option}")),
                    ]));
                }
                QuestionKind::MultiChoice => {
                    let (check, check_style) = if q.selected[i] {
                        ("[x]", cursor_style)
                    } else {
                        ("[ ]", Style::default())
                    };
                    lines.push(Line::from(vec![
                        Span::styled(marker.to_string(), style),
                        Span::styled(check.to_string(), check_style),
                        Span::raw(format!(" {option}")),
                    ]));
                }
                QuestionKind::Free => {}
            }
        }
    }
    let show_draft = matches!(
        current.kind,
        QuestionKind::Free | QuestionKind::MultiChoice
    ) || !q.draft.is_empty();
    if show_draft {
        for line in input_visual_lines(&q.draft, q.draft_cursor, width, 0) {
            lines.push(line);
        }
    }
    let hint = match current.kind {
        QuestionKind::SingleChoice => "↑↓ pick · enter confirm · type for own answer · esc cancel",
        QuestionKind::MultiChoice => "↑↓ move · space toggle · enter confirm · type for own answer · esc cancel",
        QuestionKind::Free => "enter confirm · esc cancel",
    };
    lines.push(Line::from(Span::styled(hint.to_string(), Style::default().dim())));
    lines
}

fn move_cursor_visual_line(input: &str, cursor: &mut usize, width: usize, up: bool) {
    let chars: Vec<char> = input.chars().collect();
    let ranges = input_visual_ranges(&chars, width);
    let cur = ranges
        .iter()
        .position(|(start, count)| *cursor >= *start && *cursor <= start + count)
        .unwrap_or(0);
    let target = if up {
        cur.saturating_sub(1)
    } else {
        (cur + 1).min(ranges.len() - 1)
    };
    if target == cur {
        return;
    }
    let (cur_start, _) = ranges[cur];
    let (target_start, target_count) = ranges[target];
    let col = (*cursor - cur_start).min(target_count);
    *cursor = target_start + col;
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
                    spans.push(Span::styled(" ".repeat(width - current), Style::default()));
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
    let input_lines = active_box_lines(&state.sessions[state.active], state.pane_width);
    let chunks = Layout::new(
        Direction::Vertical,
        [
            Constraint::Length(3),
            Constraint::Min(1),
            Constraint::Length(2 + input_lines as u16),
        ],
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

    let display = display_lines(
        session.renderer.scrollback(),
        session.renderer.blocks(),
        state.pane_width,
    );
    let display = &display[start.min(display.len())..];
    let main = Paragraph::new(display).wrap(Wrap { trim: false }).block(
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
                    let status = if session.running {
                        "working…"
                    } else {
                        "idle"
                    };
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
                            format!(
                                "  {} · {} / {}",
                                status, session.prompt_tokens, session.completion_tokens
                            ),
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
                        crate::bg::BgStatus::Finished(Some(code)) => format!("exit {code:<4}"),
                    };
                    let duration = task
                        .finished_at
                        .map(|finished| finished.duration_since(task.started_at))
                        .unwrap_or_else(|| {
                            std::time::Instant::now().duration_since(task.started_at)
                        });
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
                        Span::styled(format!("{}  ", crate::bg::format_duration(duration)), style),
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
                Span::styled(
                    format!("  ·  {}", status),
                    Style::default().fg(Color::Rgb(0x81, 0xa2, 0xbe)),
                ),
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
                body.extend(lines[start..].iter().take(visible).map(|line| {
                    Line::from(Span::styled(
                        *line,
                        Style::default().fg(Color::Rgb(0x80, 0x80, 0x80)),
                    ))
                }));
            }
            let hint = Line::from(Span::styled(
                " jk scroll · q close · C-q close all",
                Style::default().fg(Color::Rgb(102, 102, 102)),
            ));
            let output_box = Paragraph::new(
                body.into_iter()
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

    let input = if session.question.is_some() {
        Paragraph::new(question_wizard_lines(
            session.question.as_ref().unwrap(),
            state.pane_width,
        ))
    } else if session.running {
        Paragraph::new(Line::from(Span::styled(
            "working…".to_string(),
            Style::default().dim(),
        )))
    } else if let Some(error) = &session.error {
        Paragraph::new(Line::from(Span::styled(error.clone(), Style::default().red())))
    } else {
        Paragraph::new(input_visual_lines(
            &session.input,
            session.input_cursor,
            state.pane_width,
            session.input_scroll,
        ))
    };
    let shared = *state.mode.lock().unwrap();
    let effective = session.stage.unwrap_or(shared);
    let mode_title = if session.gate {
        Line::from(Span::styled(
            format!(" {} ", session.gate_message),
            Style::default().bold().fg(Color::Rgb(0xb5, 0xbd, 0x68)),
        ))
    } else {
        Line::from(Span::styled(
            format!(
                " {} {} ",
                match effective {
                    Mode::Plan => "📋",
                    Mode::Implement => "🔧",
                    Mode::Yolo => "⚒",
                },
                effective.label()
            ),
            Style::default().bold().fg(match effective {
                Mode::Plan => Color::Rgb(0xb5, 0xbd, 0x68),
                Mode::Implement => Color::Rgb(0xe5, 0xb5, 0x67),
                Mode::Yolo => Color::Rgb(0xd4, 0xd4, 0xd4),
            }),
        ))
    };
    let mut input_block = Block::default().borders(Borders::ALL).title(mode_title);
    let total_lines = {
        let chars: Vec<char> = session.input.chars().collect();
        input_visual_ranges(&chars, state.pane_width).len()
    };
    let visible = total_lines.min(12);
    let scroll = session.input_scroll.min(total_lines.saturating_sub(visible));
    let hidden_top = scroll;
    let hidden_bottom = total_lines - scroll - visible;
    if session.question.is_none() && !session.running && hidden_top > 0 {
        input_block = input_block.title_top(
            Line::from(format!("↑ {} more", hidden_top)).right_aligned(),
        );
    }
    if session.question.is_none() && !session.running && hidden_bottom > 0 {
        input_block = input_block.title_bottom(
            Line::from(format!("↓ {} more", hidden_bottom)).right_aligned(),
        );
    }
    frame.render_widget(
        input
            .wrap(Wrap { trim: false })
            .block(input_block),
        chunks[2],
    );
}

#[derive(serde::Deserialize)]
struct SubmitPlanPayload {
    #[serde(default)]
    stages: Vec<crate::tool::PlanStage>,
    #[serde(default)]
    plan: Option<String>,
}

fn parse_stages(arguments: &str) -> Vec<crate::tool::PlanStage> {
    let payload: SubmitPlanPayload =
        serde_json::from_str(arguments).unwrap_or(SubmitPlanPayload {
            stages: Vec::new(),
            plan: None,
        });
    if !payload.stages.is_empty() {
        return payload.stages;
    }
    let text = payload
        .plan
        .unwrap_or_else(|| crate::agent::terminator_payload(arguments, "plan"));
    vec![crate::tool::PlanStage {
        title: "Plan".to_string(),
        tasks: vec![text],
    }]
}

fn stage_implement_message(
    stages: &Option<Vec<crate::tool::PlanStage>>,
    index: usize,
) -> String {
    let Some(stages) = stages.as_ref() else {
        return "Execute the plan.".to_string();
    };
    let stage = &stages[index];
    let done = if index > 0 {
        let done = stages[..index]
            .iter()
            .enumerate()
            .map(|(i, s)| format!("Step {}: {}", i + 1, s.title))
            .collect::<Vec<_>>()
            .join("; ");
        format!("Stages already done: {done}.\n")
    } else {
        String::new()
    };
    let tasks = stage
        .tasks
        .iter()
        .map(|task| format!("- {task}"))
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "Execute Step {} of {}: {}\nTasks:\n{tasks}\n{}Do not start later stages.",
        index + 1,
        stages.len(),
        stage.title,
        done
    )
}

fn stage_implementation_feedback_message(
    stages: &Option<Vec<crate::tool::PlanStage>>,
    index: usize,
    feedback: &str,
) -> String {
    let Some(stages) = stages.as_ref() else {
        return format!(
            "The implementation was rejected. Feedback: {feedback}\n\nRevise the plan and call submit_plan."
        );
    };
    format!(
        "Step {} of {} was implemented, but the user's feedback: {feedback}\n\nCurrent stage:\n{}\n\nRevise the stages from Step {} on (redo this stage if needed), keeping what the earlier stages did, and call submit_plan with all of them.",
        index + 1,
        stages.len(),
        crate::tool::plan_text(&stages[index..index + 1]),
        index + 1
    )
}

fn stage_replan_message(
    stages: &Option<Vec<crate::tool::PlanStage>>,
    index: usize,
    feedback: &str,
) -> String {
    let Some(stages) = stages.as_ref() else {
        return format!(
            "The plan was rejected. Feedback: {feedback}\n\nRevise the plan and call submit_plan."
        );
    };
    if index == 0 {
        format!(
            "The plan was rejected. Feedback: {feedback}\n\nOriginal plan:\n{}\n\nRevise the plan and call submit_plan.",
            crate::tool::plan_text(stages)
        )
    } else {
        format!(
            "Step {} of {} is done. Feedback on the remaining stages: {feedback}\n\nRemaining stages:\n{}\n\nRevise the remaining stages and call submit_plan with all of them.",
            index,
            stages.len(),
            crate::tool::plan_text(&stages[index..])
        )
    }
}

#[derive(Clone, Copy, PartialEq)]
enum GatePhase {
    ReviewImplementation,
    ReviewPlan,
}

async fn handle_outcome(
    agent: &mut Agent,
    new_agent: &std::sync::Arc<dyn Fn() -> Agent + Send + Sync>,
    event_tx: &tokio::sync::mpsc::UnboundedSender<TuiEvent>,
    id: u64,
    gate: &mut Option<(String, GatePhase)>,
    stages: &mut Option<Vec<crate::tool::PlanStage>>,
    stage_index: &mut usize,
    mut outcome: crate::agent::ChatOutcome,
) {
    loop {
        match &outcome {
            crate::agent::ChatOutcome::Terminated { tool, arguments } if tool == "submit_plan" => {
                *stages = Some(parse_stages(arguments));
                *stage_index = 0;
                let (count, title) = stages
                    .as_ref()
                    .map(|s| {
                        (
                            s.len(),
                            s.first().map(|stage| stage.title.clone()).unwrap_or_default(),
                        )
                    })
                    .unwrap_or((0, String::new()));
                *gate = Some((
                    format!(
                        "📋 plan ready ({count} stages) — Enter to implement Step 1: {title}, type feedback to re-plan"
                    ),
                    GatePhase::ReviewPlan,
                ));
                let context = agent.context.clone();
                let _ = event_tx.send(TuiEvent::TurnDone {
                    session: id,
                    context,
                });
                let _ = event_tx.send(TuiEvent::GatePending {
                    session: id,
                    message: gate.clone().unwrap().0,
                });
                return;
            }
            crate::agent::ChatOutcome::Terminated { tool, arguments } if tool == "escalate" => {
                let findings = crate::agent::terminator_payload(arguments, "findings");
                *agent = new_agent().with_pinned_mode(Mode::Plan);
                let _ = event_tx.send(TuiEvent::StageChanged {
                    session: id,
                    mode: Some(Mode::Plan),
                });
                let message = if let Some(s) = stages.as_ref() {
                    format!(
                        "The implementation hit a blocker at Step {} of {}: {}\n\nRevise the remaining stages (from the current one on), keeping what is already done, and call submit_plan with all of them.",
                        *stage_index + 1,
                        s.len(),
                        findings
                    )
                } else {
                    format!(
                        "The implementation hit a blocker: {findings}\n\nRevise the plan, keeping what is already done, and call submit_plan."
                    )
                };
                let tx = event_tx.clone();
                match agent
                    .chat(&message, &mut |event| {
                        let _ = tx.send(TuiEvent::Agent { session: id, event });
                    })
                    .await
                {
                    Ok(next) => {
                        outcome = next;
                        continue;
                    }
                    Err(error) => {
                        let _ = event_tx.send(TuiEvent::TurnError {
                            session: id,
                            message: error.to_string(),
                        });
                        return;
                    }
                }
            }
            _ => {
                let review = agent.stage_mode() == Some(Mode::Implement)
                    && stages
                        .as_ref()
                        .map(|s| *stage_index + 1 < s.len())
                        .unwrap_or(false);
                if review {
                    *stage_index += 1;
                    *gate = Some((
                        format!(
                            "📋 Step {} implemented — review the implementation — Enter to continue, type feedback to re-plan",
                            *stage_index
                        ),
                        GatePhase::ReviewImplementation,
                    ));
                    let context = agent.context.clone();
                    let _ = event_tx.send(TuiEvent::TurnDone {
                        session: id,
                        context,
                    });
                    let _ = event_tx.send(TuiEvent::GatePending {
                        session: id,
                        message: gate.clone().unwrap().0,
                    });
                    return;
                }
                if agent.stage_mode() == Some(Mode::Implement) {
                    *agent = new_agent();
                    let _ = event_tx.send(TuiEvent::StageChanged {
                        session: id,
                        mode: None,
                    });
                }
                let context = agent.context.clone();
                let _ = event_tx.send(TuiEvent::TurnDone {
                    session: id,
                    context,
                });
                return;
            }
        }
    }
}

fn spawn_agent(
    id: u64,
    new_agent: std::sync::Arc<dyn Fn() -> Agent + Send + Sync>,
    restored: Option<Context>,
    event_tx: tokio::sync::mpsc::UnboundedSender<TuiEvent>,
) -> (
    tokio::sync::mpsc::UnboundedSender<String>,
    tokio::task::AbortHandle,
) {
    let (input_tx, mut input_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    let task = tokio::spawn(async move {
        let mut agent = new_agent();
        if let Some(context) = restored {
            agent.context = context;
        }
        let mut gate: Option<(String, GatePhase)> = None;
        let mut stages: Option<Vec<crate::tool::PlanStage>> = None;
        let mut stage_index: usize = 0;
        let mut bg_rx = crate::bg::REGISTRY.subscribe();
        loop {
            tokio::select! {
                input = input_rx.recv() => {
                    let Some(mut input) = input else { break; };
                    if let Some((_, phase)) = gate.take() {
                        match (phase, input.is_empty()) {
                            (GatePhase::ReviewImplementation, true) => {
                                let title = stages
                                    .as_ref()
                                    .map(|s| s[stage_index].title.clone())
                                    .unwrap_or_default();
                                let message = format!(
                                    "📋 review Step {}: {title} — Enter to implement, type feedback to re-plan",
                                    stage_index + 1
                                );
                                gate = Some((message.clone(), GatePhase::ReviewPlan));
                                let _ = event_tx.send(TuiEvent::StageChanged { session: id, mode: Some(Mode::Plan) });
                                let _ = event_tx.send(TuiEvent::GatePending { session: id, message });
                                continue;
                            }
                            (GatePhase::ReviewImplementation, false) => {
                                agent = new_agent().with_pinned_mode(Mode::Plan);
                                let _ = event_tx.send(TuiEvent::StageChanged { session: id, mode: Some(Mode::Plan) });
                                input = stage_implementation_feedback_message(
                                    &stages,
                                    stage_index.saturating_sub(1),
                                    &input,
                                );
                            }
                            (GatePhase::ReviewPlan, true) => {
                                agent = new_agent().with_pinned_mode(Mode::Implement);
                                let _ = event_tx.send(TuiEvent::StageChanged { session: id, mode: Some(Mode::Implement) });
                                input = stage_implement_message(&stages, stage_index);
                            }
                            (GatePhase::ReviewPlan, false) => {
                                agent = new_agent().with_pinned_mode(Mode::Plan);
                                let _ = event_tx.send(TuiEvent::StageChanged { session: id, mode: Some(Mode::Plan) });
                                input = stage_replan_message(&stages, stage_index, &input);
                            }
                        }
                    }
                    let tx = event_tx.clone();
                    let result = agent
                        .chat(
                            &input,
                            &mut |event| {
                                let _ = tx.send(TuiEvent::Agent { session: id, event });
                            },
                        )
                        .await
                        .map_err(|error| error.to_string());
                    match result {
                        Ok(outcome) => {
                            handle_outcome(
                                &mut agent,
                                &new_agent,
                                &event_tx,
                                id,
                                &mut gate,
                                &mut stages,
                                &mut stage_index,
                                outcome,
                            )
                            .await;
                        }
                        Err(message) => {
                            let _ = event_tx.send(TuiEvent::TurnError {
                                session: id,
                                message,
                            });
                        }
                    }
                }
                signal = bg_rx.recv() => {
                    if gate.is_none()
                        && let Ok(task_id) = signal
                        && agent.owns_and_unseen(&task_id)
                    {
                        let tx = event_tx.clone();
                        let result = agent
                            .bg_turn(&mut |event| {
                                let _ = tx.send(TuiEvent::Agent { session: id, event });
                            })
                            .await
                            .map_err(|error| error.to_string());
                        match result {
                            Ok(outcome) => {
                                handle_outcome(
                                    &mut agent,
                                    &new_agent,
                                    &event_tx,
                                    id,
                                    &mut gate,
                                    &mut stages,
                                    &mut stage_index,
                                    outcome,
                                )
                                .await;
                            }
                            Err(message) => {
                                let _ = tx.send(TuiEvent::TurnError {
                                    session: id,
                                    message,
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
    let new_agent = std::sync::Arc::new(new_agent);

    std::io::stdout().write_all(b"\x1b[>7u\x1b[?u")?;
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = std::io::stdout().write_all(b"\x1b[?25h");
        let _ = std::io::stdout().write_all(b"\x1b[<1u");
        let _ = std::io::stdout().flush();
        default_hook(info);
    }));

    let backend = TermwizBackend::new()?;
    let mut terminal = Terminal::new(backend)?;
    terminal.clear()?;

    let (key_tx, mut key_rx) = tokio::sync::mpsc::unbounded_channel::<TermEvent>();
    std::thread::spawn(move || {
        let mut stdin = std::io::stdin();
        let mut parser = termwiz::input::InputParser::new();
        let mut buf = [0u8; 512];
        loop {
            match stdin.read(&mut buf) {
                Ok(n) if n > 0 => {
                    let maybe_more = n == buf.len();
                    parser.parse(
                        &buf[..n],
                        |event| {
                            if let Some(term_event) = map_termwiz_event(event) {
                                if key_tx.send(term_event).is_err() {
                                    std::process::exit(1);
                                }
                            }
                        },
                        maybe_more,
                    );
                }
                Ok(_) => continue,
                Err(_) => return,
            }
        }
    });

    let mut state =
        TuiState::with_sessions(model, session_store::load_sessions(Path::new(CONTEXT_DIR)))
            .with_thinking(thinking)
            .with_mode(mode);
    let size = terminal.size();
    state.pane_width = size
        .as_ref()
        .map(|size| size.width.saturating_sub(2) as usize)
        .unwrap_or(118);
    let input_lines = {
        let pw = state.pane_width;
        active_box_lines(state.session(), pw)
    };
    state.viewport = size
        .map(|size| size.height.saturating_sub(7 + input_lines as u16) as usize)
        .unwrap_or(22);
    let mut inputs: HashMap<u64, tokio::sync::mpsc::UnboundedSender<String>> = HashMap::new();
    let mut handles: HashMap<u64, tokio::task::AbortHandle> = HashMap::new();

    for session in &state.sessions {
        let restored = session.context.clone();
        let (input_tx, input_handle) =
            spawn_agent(session.id, new_agent.clone(), restored, agent_tx.clone());
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

    let start = state.sessions[state.active]
        .scroller
        .offset()
        .min(state.sessions[state.active].max_scroll(state.pane_width, state.viewport));
    let pw = state.pane_width;
    clamp_input_scroll(state.session(), pw);
    terminal.draw(|frame| draw(frame, &state, start))?;

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
                            if let AgentEvent::AskUser { questions, reply } = &event {
                                let mut q = QuestionState::new(questions.clone());
                                q.reply = Some(reply.clone());
                                session.question = Some(q);
                            }
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
                                session_store::save_session(
                                    Path::new(CONTEXT_DIR),
                                    id,
                                    &session.label,
                                    saved,
                                    &session.history,
                                );
                            }
                        }
                    }
                    Some(TuiEvent::TurnError { session: id, message }) => {
                        if let Some(session) = state.sessions.iter_mut().find(|s| s.id == id) {
                            session.renderer.finish();
                            session.running = false;
                            session.error = Some(message);
                            if let Some(saved) = &session.context {
                                session_store::save_session(
                                    Path::new(CONTEXT_DIR),
                                    id,
                                    &session.label,
                                    saved,
                                    &session.history,
                                );
                            }
                        }
                    }
                    Some(TuiEvent::GatePending { session: id, message }) => {
                        if let Some(session) = state.sessions.iter_mut().find(|s| s.id == id) {
                            session.gate = true;
                            session.gate_message = message;
                            session.running = false;
                        }
                    }
                    Some(TuiEvent::StageChanged { session: id, mode }) => {
                        if let Some(session) = state.sessions.iter_mut().find(|s| s.id == id) {
                            session.stage = mode;
                            session.gate = false;
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
                        if !task.is_empty() {
                            session.renderer.push_user(&task);
                        }
                        let max = session.max_scroll(state.pane_width, state.viewport);
                        session.scroller.end(max);
                        inputs[&id].send(task)?;
                    }
                    KeyAction::Quit => break,
                    KeyAction::NewSession => {
                        let id = state.sessions[state.active].id;
                        let (input_tx, input_handle) =
                            spawn_agent(id, new_agent.clone(), None, agent_tx.clone());
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
            _ = tokio::time::sleep(std::time::Duration::from_millis(16)) => {}
        }
        let size = terminal.size();
        state.pane_width = size
            .as_ref()
            .map(|size| size.width.saturating_sub(2) as usize)
            .unwrap_or(118);
        let input_lines = {
            let pw = state.pane_width;
            active_box_lines(state.session(), pw)
        };
        state.viewport = size
            .map(|size| size.height.saturating_sub(7 + input_lines as u16) as usize)
            .unwrap_or(22);
        for session in &mut state.sessions {
            if session.scroller.following() {
                let max = session.max_scroll(state.pane_width, state.viewport);
                session.scroller.follow_tail(max);
            }
        }
        let start = state.sessions[state.active]
            .scroller
            .offset()
            .min(state.sessions[state.active].max_scroll(state.pane_width, state.viewport));
        let pw = state.pane_width;
        clamp_input_scroll(state.session(), pw);
        terminal.draw(|frame| draw(frame, &state, start))?;
    }

    for session in &state.sessions {
        if !session.running
            && let Some(context) = &session.context
            && session_store::session_file_exists(Path::new(CONTEXT_DIR), session.id)
        {
            session_store::save_session(
                Path::new(CONTEXT_DIR),
                session.id,
                &session.label,
                context,
                &session.history,
            );
        }
    }
    for handle in handles.values() {
        handle.abort();
    }
    std::io::stdout().write_all(b"\x1b[<1u")?;
    std::io::stdout().flush()?;
    drop(terminal);
    println!(
        "session over: {} prompt / {} completion tokens",
        state.sessions[state.active].prompt_tokens, state.sessions[state.active].completion_tokens
    );
    Ok(())
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::message::Message;
    use crate::stream::ChunkTokens;
    use crate::transcript::TuiRenderer;
    
    use openai_oxide::client::OpenAI;
    use ratatui::backend::TestBackend;
    use ratatui::style::Modifier;
    use std::time::Duration;
    use tokio::io::AsyncReadExt;
    use tokio::io::AsyncWriteExt;

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

    fn mouse(kind: MouseEventKind) -> TermEvent {
        TermEvent::Mouse(MouseEvent { kind })
    }

    #[test]
    fn mouse_wheel_scrolls_the_viewport() {
        let mut state = TuiState::new("model".into());
        for i in 0..30 {
            state
                .session()
                .renderer
                .on_event(text(&format!("line {i}\n")));
        }
        state.session().renderer.finish();
        let (pw, vp) = (state.pane_width, state.viewport);
        let max = state.session().max_scroll(pw, vp);
        assert!(max > 0);

        handle_key(&mut state, &mouse(MouseEventKind::ScrollDown));
        assert_eq!(state.session().scroller.offset(), 3);
        assert!(!state.session().scroller.following());

        handle_key(&mut state, &mouse(MouseEventKind::ScrollUp));
        assert_eq!(state.session().scroller.offset(), 0);
        assert!(!state.session().scroller.following());

        for _ in 0..20 {
            handle_key(&mut state, &mouse(MouseEventKind::ScrollDown));
        }
        assert_eq!(state.session().scroller.offset(), max);
        assert!(state.session().scroller.following());
    }

    fn ctrl(c: char) -> TermEvent {
        TermEvent::Key(KeyEvent::new(KeyCode::Char(c), KeyModifiers::CONTROL))
    }

    fn ctrl_key(code: KeyCode) -> TermEvent {
        TermEvent::Key(KeyEvent::new(code, KeyModifiers::CONTROL))
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
        state.tasks_cursor.set(
            crate::bg::REGISTRY
                .list()
                .iter()
                .position(|t| t.id == id)
                .unwrap(),
        );
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
                crate::bg::REGISTRY
                    .list()
                    .into_iter()
                    .find(|t| t.id == id)
                    .unwrap()
                    .status,
                crate::bg::BgStatus::Finished(_)
            ) {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        assert!(matches!(
            crate::bg::REGISTRY
                .list()
                .into_iter()
                .find(|t| t.id == id)
                .unwrap()
                .status,
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
                crate::bg::REGISTRY
                    .list()
                    .into_iter()
                    .find(|t| t.id == id)
                    .unwrap()
                    .status,
                crate::bg::BgStatus::Finished(_)
            ) {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        let mut state = TuiState::new("model".into());

        handle_key(&mut state, &ctrl('q'));
        state.tasks_cursor.set(
            crate::bg::REGISTRY
                .list()
                .iter()
                .position(|t| t.id == id)
                .unwrap(),
        );
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
                crate::bg::REGISTRY
                    .list()
                    .into_iter()
                    .find(|t| t.id == id)
                    .unwrap()
                    .status,
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
            .flat_map(|y| (0..80).map(move |x| buffer.cell((x, y)).unwrap().symbol().to_string()))
            .collect();
        assert!(joined.contains("popup-visible"));
        assert!(joined.contains("task"));
    }

    #[tokio::test]
    async fn task_output_popup_is_opaque_over_chat() {
        let id = crate::bg::REGISTRY.run("seq 1 30").unwrap();
        for _ in 0..200 {
            if matches!(
                crate::bg::REGISTRY
                    .list()
                    .into_iter()
                    .find(|t| t.id == id)
                    .unwrap()
                    .status,
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
            .flat_map(|y| (0..80).map(move |x| buffer.cell((x, y)).unwrap().symbol().to_string()))
            .collect();
        assert!(!in_popup.contains("CHATLINE"));
        assert!(in_popup.contains("30"));
    }

    #[tokio::test]
    async fn mouse_wheel_scrolls_the_open_output_popup() {
        let id = crate::bg::REGISTRY.run("seq 1 30").unwrap();
        for _ in 0..200 {
            if matches!(
                crate::bg::REGISTRY
                    .list()
                    .into_iter()
                    .find(|t| t.id == id)
                    .unwrap()
                    .status,
                crate::bg::BgStatus::Finished(_)
            ) {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        let mut state = TuiState::new("model".into());
        state.task_output_id = Some(id);
        state.task_output_scroll.offset = 10;

        handle_key(&mut state, &mouse(MouseEventKind::ScrollUp));
        assert_eq!(state.task_output_scroll.offset(), 7);
        handle_key(&mut state, &mouse(MouseEventKind::ScrollDown));
        assert_eq!(state.task_output_scroll.offset(), 10);
    }

    #[tokio::test]
    async fn output_scroll_clamps_at_the_bottom_and_scrolls_up_immediately() {
        let id = crate::bg::REGISTRY.run("seq 1 30").unwrap();
        for _ in 0..200 {
            if matches!(
                crate::bg::REGISTRY
                    .list()
                    .into_iter()
                    .find(|t| t.id == id)
                    .unwrap()
                    .status,
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
                crate::bg::REGISTRY
                    .list()
                    .into_iter()
                    .find(|t| t.id == id)
                    .unwrap()
                    .status,
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
            .flat_map(|y| (0..80).map(move |x| buffer.cell((x, y)).unwrap().symbol().to_string()))
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
            .flat_map(|y| (0..80).map(move |x| buffer.cell((x, y)).unwrap().symbol().to_string()))
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
            .flat_map(|y| (0..80).map(move |x| buffer.cell((x, y)).unwrap().symbol().to_string()))
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

        assert!(
            described
                .iter()
                .any(|(c, col)| c == "    1   one" && col == &Some(Color::Rgb(128, 128, 128)))
        );
        assert!(
            described
                .iter()
                .any(|(c, col)| c == "-   2  two" && col == &Some(Color::Rgb(0xcc, 0x66, 0x66)))
        );
        assert!(
            described
                .iter()
                .any(|(c, col)| c == "+   2  TWO" && col == &Some(Color::Rgb(0xb5, 0xbd, 0x68)))
        );
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

        assert_eq!(
            handle_key(&mut state, &key(KeyCode::Enter)),
            KeyAction::None
        );
    }

    #[test]
    fn backspace_pops_the_input() {
        let mut state = TuiState::new("model".into());
        state.session().input = "ab".into();
        state.session().input_cursor = 2;

        assert_eq!(
            handle_key(&mut state, &key(KeyCode::Backspace)),
            KeyAction::None
        );
        assert_eq!(state.session().input, "a");
        assert_eq!(state.session().input_cursor, 1);
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

        assert_eq!(
            *state.thinking.lock().unwrap(),
            crate::thinking::ThinkingLevel::Off
        );
        handle_key(&mut state, &event);
        assert_eq!(
            *state.thinking.lock().unwrap(),
            crate::thinking::ThinkingLevel::Low
        );
        handle_key(&mut state, &event);
        assert_eq!(
            *state.thinking.lock().unwrap(),
            crate::thinking::ThinkingLevel::Medium
        );
        handle_key(&mut state, &event);
        assert_eq!(
            *state.thinking.lock().unwrap(),
            crate::thinking::ThinkingLevel::High
        );
        handle_key(&mut state, &event);
        assert_eq!(
            *state.thinking.lock().unwrap(),
            crate::thinking::ThinkingLevel::XHigh
        );
        handle_key(&mut state, &event);
        assert_eq!(
            *state.thinking.lock().unwrap(),
            crate::thinking::ThinkingLevel::Off
        );
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
        assert_eq!(
            user.spans.last().unwrap().style.bg,
            Some(Color::Rgb(52, 53, 65))
        );
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
        assert!(
            renderer
                .blocks()
                .iter()
                .all(|b| *b == BlockKind::ToolRunning)
        );
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
        assert!(
            renderer
                .blocks()
                .iter()
                .all(|b| *b == BlockKind::ToolRunning)
        );
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
        assert_eq!(
            display[0].spans.last().unwrap().style.bg,
            Some(Color::Rgb(40, 40, 50))
        );
        assert_eq!(
            display[3].spans.last().unwrap().style.bg,
            Some(Color::Rgb(40, 50, 40))
        );
    }

    #[test]
    fn tail_start_accounts_for_wrapped_lines() {
        let mut state = TuiState::new("model".into());
        state.viewport = 4;
        state.pane_width = 20;
        for _ in 0..3 {
            state
                .session()
                .renderer
                .on_event(text(&format!("{}\n", "a".repeat(40))));
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
            state
                .session()
                .renderer
                .on_event(text(&format!("line {} {}\n", i, "x".repeat(width))));
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

        assert_eq!(
            handle_key(&mut state, &key(KeyCode::PageUp)),
            KeyAction::None
        );
        assert_eq!(state.session().scroller.offset(), 2);
        assert!(!state.session().scroller.following());
        assert_eq!(
            handle_key(&mut state, &key(KeyCode::PageUp)),
            KeyAction::None
        );
        assert_eq!(state.session().scroller.offset(), 0);
        assert_eq!(
            handle_key(&mut state, &key(KeyCode::PageDown)),
            KeyAction::None
        );
        assert_eq!(state.session().scroller.offset(), 10);
        assert!(!state.session().scroller.following());
        assert_eq!(
            handle_key(&mut state, &key(KeyCode::PageDown)),
            KeyAction::None
        );
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

        assert_eq!(
            handle_key(&mut state, &key(KeyCode::Char('a'))),
            KeyAction::None
        );
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
    fn enter_with_empty_input_is_a_noop_without_a_gate() {
        let mut state = TuiState::new("model".into());
        assert_eq!(
            handle_key(&mut state, &key(KeyCode::Enter)),
            KeyAction::None
        );
    }

    #[test]
    fn cursor_cell_renders_inverted() {
        let mut state = TuiState::new("model".into());
        state.session().input = "ab".into();
        state.session().input_cursor = 1;
        let backend = TestBackend::new(40, 10);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|frame| draw(frame, &state, 0)).unwrap();
        let buffer = terminal.backend().buffer();
        let mut found = false;
        for y in 0..10 {
            for x in 0..40 {
                let cell = buffer.cell((x, y)).unwrap();
                if cell.symbol() == "b" {
                    found = true;
                    let style = cell.style();
                    assert_eq!(
                        style.bg,
                        Some(Color::Rgb(0xd4, 0xd4, 0xd4)),
                        "cursor cell bg missing"
                    );
                    assert_eq!(
                        style.fg,
                        Some(Color::Rgb(0x28, 0x28, 0x32)),
                        "cursor cell fg missing"
                    );
                }
            }
        }
        assert!(found, "cursor char not found");
    }

    #[test]
    fn typing_inserts_at_the_cursor_and_moves_it() {
        let mut state = TuiState::new("model".into());
        for c in "hello".chars() {
            handle_key(&mut state, &key(KeyCode::Char(c)));
        }
        handle_key(&mut state, &key(KeyCode::Left));
        handle_key(&mut state, &key(KeyCode::Left));
        assert_eq!(state.session().input_cursor, 3);
        handle_key(&mut state, &key(KeyCode::Char('X')));
        assert_eq!(state.session().input, "helXlo");
        assert_eq!(state.session().input_cursor, 4);
    }

    #[test]
    fn backspace_deletes_before_the_cursor() {
        let mut state = TuiState::new("model".into());
        for c in "hello".chars() {
            handle_key(&mut state, &key(KeyCode::Char(c)));
        }
        handle_key(&mut state, &key(KeyCode::Left));
        handle_key(&mut state, &key(KeyCode::Backspace));
        assert_eq!(state.session().input, "helo");
        assert_eq!(state.session().input_cursor, 3);
        handle_key(&mut state, &key(KeyCode::Left));
        handle_key(&mut state, &key(KeyCode::Left));
        handle_key(&mut state, &key(KeyCode::Left));
        handle_key(&mut state, &key(KeyCode::Backspace));
        assert_eq!(state.session().input, "helo");
        assert_eq!(state.session().input_cursor, 0);
    }

    #[test]
    fn arrows_move_the_cursor_by_char_and_clamp() {
        let mut state = TuiState::new("model".into());
        for c in "hello".chars() {
            handle_key(&mut state, &key(KeyCode::Char(c)));
        }
        assert_eq!(state.session().input_cursor, 5);
        handle_key(&mut state, &key(KeyCode::Right));
        assert_eq!(state.session().input_cursor, 5);
        handle_key(&mut state, &key(KeyCode::Left));
        handle_key(&mut state, &key(KeyCode::Left));
        assert_eq!(state.session().input_cursor, 3);
        for _ in 0..10 {
            handle_key(&mut state, &key(KeyCode::Left));
        }
        assert_eq!(state.session().input_cursor, 0);
    }

    #[test]
    fn ctrl_arrows_move_the_cursor_by_word() {
        let mut state = TuiState::new("model".into());
        for c in "hello world foo".chars() {
            handle_key(&mut state, &key(KeyCode::Char(c)));
        }
        assert_eq!(state.session().input_cursor, 15);
        handle_key(&mut state, &ctrl_key(KeyCode::Left));
        assert_eq!(state.session().input_cursor, 12);
        handle_key(&mut state, &ctrl_key(KeyCode::Left));
        assert_eq!(state.session().input_cursor, 6);
        handle_key(&mut state, &ctrl_key(KeyCode::Left));
        assert_eq!(state.session().input_cursor, 0);
        handle_key(&mut state, &ctrl_key(KeyCode::Right));
        assert_eq!(state.session().input_cursor, 6);
        handle_key(&mut state, &ctrl_key(KeyCode::Right));
        assert_eq!(state.session().input_cursor, 12);
        handle_key(&mut state, &ctrl_key(KeyCode::Right));
        assert_eq!(state.session().input_cursor, 15);
    }

    #[test]
    fn ctrl_left_from_inside_a_word_lands_on_its_start() {
        let mut state = TuiState::new("model".into());
        for c in "hello world".chars() {
            handle_key(&mut state, &key(KeyCode::Char(c)));
        }
        for _ in 0..3 {
            handle_key(&mut state, &key(KeyCode::Left));
        }
        assert_eq!(state.session().input_cursor, 8);
        handle_key(&mut state, &ctrl_key(KeyCode::Left));
        assert_eq!(state.session().input_cursor, 6);
        handle_key(&mut state, &ctrl_key(KeyCode::Left));
        assert_eq!(state.session().input_cursor, 0);
    }

    #[test]
    fn ctrl_enter_inserts_a_newline_at_the_cursor() {
        let mut state = TuiState::new("model".into());
        for c in "ab".chars() {
            handle_key(&mut state, &key(KeyCode::Char(c)));
        }
        handle_key(&mut state, &ctrl_key(KeyCode::Enter));
        assert_eq!(state.session().input, "ab\n");
        assert_eq!(state.session().input_cursor, 3);
    }

    #[test]
    fn up_and_down_navigate_wrapped_visual_lines() {
        let mut state = TuiState::new("model".into());
        state.pane_width = 20;
        state.session().input = "a".repeat(50).into();
        state.session().input_cursor = 50;
        handle_key(&mut state, &key(KeyCode::Up));
        assert_eq!(state.session().input_cursor, 30);
        handle_key(&mut state, &key(KeyCode::Up));
        assert_eq!(state.session().input_cursor, 10);
        handle_key(&mut state, &key(KeyCode::Up));
        assert_eq!(state.session().input_cursor, 10);
        handle_key(&mut state, &key(KeyCode::Down));
        assert_eq!(state.session().input_cursor, 30);
        handle_key(&mut state, &key(KeyCode::Down));
        assert_eq!(state.session().input_cursor, 50);
        handle_key(&mut state, &key(KeyCode::Down));
        assert_eq!(state.session().input_cursor, 50);
    }

    #[test]
    fn wrapped_single_line_input_navigates_instead_of_scrolling() {
        let mut state = TuiState::new("model".into());
        for i in 0..30 {
            state
                .session()
                .renderer
                .on_event(text(&format!("line {i}\n")));
        }
        state.session().renderer.finish();
        state.pane_width = 20;
        state.session().input = "a".repeat(50).into();
        state.session().input_cursor = 50;
        let (pw, vp) = (state.pane_width, state.viewport);
        let max = state.session().max_scroll(pw, vp);
        assert!(max > 0);
        handle_key(&mut state, &key(KeyCode::Up));
        assert_eq!(state.session().scroller.offset(), 0);
        assert_eq!(state.session().input_cursor, 30);
    }

    #[test]
    fn up_and_down_move_between_input_lines() {
        let mut state = TuiState::new("model".into());
        state.session().input = "hello\nworld".into();
        state.session().input_cursor = 11;
        handle_key(&mut state, &key(KeyCode::Up));
        assert_eq!(state.session().input_cursor, 5);
        handle_key(&mut state, &key(KeyCode::Down));
        assert_eq!(state.session().input_cursor, 11);
        handle_key(&mut state, &key(KeyCode::Up));
        handle_key(&mut state, &key(KeyCode::Up));
        assert_eq!(state.session().input_cursor, 5);
        handle_key(&mut state, &key(KeyCode::Down));
        handle_key(&mut state, &key(KeyCode::Down));
        assert_eq!(state.session().input_cursor, 11);
    }

    #[test]
    fn up_and_down_clamp_the_column_to_the_target_line() {
        let mut state = TuiState::new("model".into());
        state.session().input = "hi\nhello".into();
        state.session().input_cursor = 5;
        handle_key(&mut state, &key(KeyCode::Up));
        assert_eq!(state.session().input_cursor, 2);
        handle_key(&mut state, &key(KeyCode::Down));
        assert_eq!(state.session().input_cursor, 5);
    }

    #[test]
    fn up_on_empty_input_loads_the_last_message() {
        let mut state = TuiState::new("model".into());
        state.session().history = vec!["first".into(), "second".into(), "third".into()];
        handle_key(&mut state, &key(KeyCode::Up));
        assert_eq!(state.session().input, "third");
        assert_eq!(state.session().history_index, Some(2));
        assert_eq!(state.session().input_cursor, 5);
    }

    #[test]
    fn up_walks_backwards_through_history() {
        let mut state = TuiState::new("model".into());
        state.session().history = vec!["first".into(), "second".into(), "third".into()];
        handle_key(&mut state, &key(KeyCode::Up));
        assert_eq!(state.session().input, "third");
        handle_key(&mut state, &key(KeyCode::Up));
        assert_eq!(state.session().input, "second");
        handle_key(&mut state, &key(KeyCode::Up));
        assert_eq!(state.session().input, "first");
        handle_key(&mut state, &key(KeyCode::Up));
        assert_eq!(state.session().input, "first");
        assert_eq!(state.session().history_index, Some(0));
    }

    #[test]
    fn down_walks_forward_and_blanks_at_the_end() {
        let mut state = TuiState::new("model".into());
        state.session().history = vec!["first".into(), "second".into(), "third".into()];
        handle_key(&mut state, &key(KeyCode::Up));
        assert_eq!(state.session().input, "third");
        handle_key(&mut state, &key(KeyCode::Up));
        assert_eq!(state.session().input, "second");
        handle_key(&mut state, &key(KeyCode::Up));
        assert_eq!(state.session().input, "first");
        handle_key(&mut state, &key(KeyCode::Down));
        assert_eq!(state.session().input, "second");
        handle_key(&mut state, &key(KeyCode::Down));
        assert_eq!(state.session().input, "third");
        handle_key(&mut state, &key(KeyCode::Down));
        assert_eq!(state.session().input, "");
        assert_eq!(state.session().history_index, None);
        assert_eq!(state.session().input_cursor, 0);
    }

    fn question_with_reply(
        questions: Vec<crate::tool::Question>,
    ) -> (
        crate::session::QuestionState,
        tokio::sync::watch::Receiver<Option<Vec<String>>>,
    ) {
        let (reply_tx, reply_rx) = tokio::sync::watch::channel(None);
        let mut state = crate::session::QuestionState::new(questions);
        state.reply = Some(reply_tx);
        (state, reply_rx)
    }

    fn question(prompt: &str, kind: &str, options: &[&str]) -> crate::tool::Question {
        use crate::tool::QuestionKind;
        let kind = match kind {
            "single_choice" => QuestionKind::SingleChoice,
            "multi_choice" => QuestionKind::MultiChoice,
            _ => QuestionKind::Free,
        };
        crate::tool::Question {
            prompt: prompt.into(),
            kind,
            options: options.iter().map(|o| o.to_string()).collect(),
        }
    }

    #[test]
    fn question_single_choice_picks_an_option() {
        let mut state = TuiState::new("model".into());
        let (q, _rx) = question_with_reply(vec![question(
            "which db?",
            "single_choice",
            &["pg", "mysql"],
        )]);
        state.session().question = Some(q);
        handle_key(&mut state, &key(KeyCode::Down));
        assert_eq!(state.session().question.as_ref().unwrap().cursor, 1);
        handle_key(&mut state, &key(KeyCode::Enter));
        assert!(state.session().question.is_none());
    }

    #[test]
    fn question_single_choice_own_answer_overrides_the_option() {
        let mut state = TuiState::new("model".into());
        let (q, _rx) = question_with_reply(vec![question(
            "which db?",
            "single_choice",
            &["pg", "mysql"],
        )]);
        state.session().question = Some(q);
        for c in "sqlite".chars() {
            handle_key(&mut state, &key(KeyCode::Char(c)));
        }
        handle_key(&mut state, &key(KeyCode::Enter));
        assert!(state.session().question.is_none());
    }

    #[test]
    fn question_multi_choice_toggles_and_joins_the_selection() {
        let mut state = TuiState::new("model".into());
        let (q, rx) = question_with_reply(vec![question(
            "pick some",
            "multi_choice",
            &["a", "b", "c"],
        )]);
        state.session().question = Some(q);
        handle_key(&mut state, &key(KeyCode::Char(' ')));
        handle_key(&mut state, &key(KeyCode::Down));
        handle_key(&mut state, &key(KeyCode::Char(' ')));
        handle_key(&mut state, &key(KeyCode::Enter));
        assert!(state.session().question.is_none());
        let answers = rx.borrow().clone().unwrap();
        assert_eq!(answers, vec!["a, b".to_string()]);
    }

    #[test]
    fn question_free_requires_a_non_empty_answer() {
        let mut state = TuiState::new("model".into());
        let (q, _rx) = question_with_reply(vec![question("name?", "free", &[])]);
        state.session().question = Some(q);
        handle_key(&mut state, &key(KeyCode::Enter));
        assert!(state.session().question.is_some());
        handle_key(&mut state, &key(KeyCode::Char('x')));
        handle_key(&mut state, &key(KeyCode::Enter));
        assert!(state.session().question.is_none());
    }

    #[test]
    fn question_multi_step_walks_through_every_question() {
        let mut state = TuiState::new("model".into());
        let (q, rx) = question_with_reply(vec![
            question("which db?", "single_choice", &["pg", "mysql"]),
            question("name?", "free", &[]),
        ]);
        state.session().question = Some(q);
        handle_key(&mut state, &key(KeyCode::Enter));
        assert_eq!(state.session().question.as_ref().unwrap().step, 1);
        for c in "kit".chars() {
            handle_key(&mut state, &key(KeyCode::Char(c)));
        }
        handle_key(&mut state, &key(KeyCode::Enter));
        assert!(state.session().question.is_none());
        let answers = rx.borrow().clone().unwrap();
        assert_eq!(answers, vec!["pg".to_string(), "kit".to_string()]);
    }

    #[test]
    fn question_esc_cancels_and_pads_the_remaining_answers() {
        let mut state = TuiState::new("model".into());
        let (q, rx) = question_with_reply(vec![
            question("which db?", "single_choice", &["pg", "mysql"]),
            question("name?", "free", &[]),
        ]);
        state.session().question = Some(q);
        handle_key(&mut state, &key(KeyCode::Char(' ')));
        handle_key(&mut state, &key(KeyCode::Esc));
        assert!(state.session().question.is_none());
        let answers = rx.borrow().clone().unwrap();
        assert_eq!(answers, vec!["cancelled".to_string(), "cancelled".to_string()]);
    }

    #[test]
    fn typing_exits_history_mode() {
        let mut state = TuiState::new("model".into());
        state.session().history = vec!["abc".into(), "def".into()];
        handle_key(&mut state, &key(KeyCode::Up));
        assert_eq!(state.session().input, "def");
        handle_key(&mut state, &key(KeyCode::Char('x')));
        assert_eq!(state.session().input, "defx");
        assert_eq!(state.session().history_index, None);
        handle_key(&mut state, &key(KeyCode::Up));
        assert_eq!(state.session().input, "defx");
    }

    #[test]
    fn submit_records_the_message_in_history() {
        let mut state = TuiState::new("model".into());
        state.session().input = "hello".into();
        state.session().input_cursor = 5;
        assert_eq!(
            handle_key(&mut state, &key(KeyCode::Enter)),
            KeyAction::Submit("hello".into())
        );
        assert_eq!(state.session().history, vec!["hello".to_string()]);
        assert_eq!(state.session().history_index, None);
    }

    #[test]
    fn multi_line_history_entry_navigates_history_not_visual_lines() {
        let mut state = TuiState::new("model".into());
        state.session().history = vec!["a\nb".into(), "c".into()];
        handle_key(&mut state, &key(KeyCode::Up));
        assert_eq!(state.session().input, "c");
        handle_key(&mut state, &key(KeyCode::Up));
        assert_eq!(state.session().input, "a\nb");
        assert_eq!(state.session().history_index, Some(0));
        handle_key(&mut state, &key(KeyCode::Up));
        assert_eq!(state.session().input, "a\nb");
        assert_eq!(state.session().input_cursor, 3);
    }

    #[test]
    fn input_box_lines_grows_with_wrapped_and_multiline_input() {
        assert_eq!(input_box_lines("", 80), 1);
        assert_eq!(input_box_lines("hello", 80), 1);
        assert_eq!(input_box_lines("hello\nworld", 80), 2);
        assert_eq!(input_box_lines(&"a".repeat(100), 80), 2);
        assert_eq!(input_box_lines(&"a".repeat(300), 80), 4);
        assert_eq!(input_box_lines(&"a".repeat(960), 80), 12);
        assert_eq!(input_box_lines(&"a".repeat(1000), 80), 12);
        assert_eq!(input_box_lines(&format!("\n{}", "a".repeat(20)), 20), 2);
        assert_eq!(input_box_lines(&format!("\n{}", "a".repeat(21)), 20), 3);
    }

    #[test]
    fn input_scroll_keeps_the_cursor_visible() {
        let mut state = TuiState::new("model".into());
        state.pane_width = 20;
        state.session().input = "a".repeat(300).into();
        state.session().input_cursor = 300;
        clamp_input_scroll(state.session(), 20);
        assert_eq!(state.session().input_scroll, 3);
        state.session().input_cursor = 0;
        clamp_input_scroll(state.session(), 20);
        assert_eq!(state.session().input_scroll, 0);
        state.session().input_cursor = 200;
        clamp_input_scroll(state.session(), 20);
        assert_eq!(state.session().input_scroll, 0);
        state.session().input_cursor = 280;
        clamp_input_scroll(state.session(), 20);
        assert_eq!(state.session().input_scroll, 2);
    }

    #[test]
    fn input_scroll_clamps_when_the_input_shrinks() {
        let mut state = TuiState::new("model".into());
        state.pane_width = 20;
        state.session().input = "a".repeat(300).into();
        state.session().input_scroll = 5;
        state.session().input = "a".repeat(100).into();
        state.session().input_cursor = 100;
        clamp_input_scroll(state.session(), 20);
        assert_eq!(state.session().input_scroll, 0);
    }

    #[test]
    fn input_visual_lines_renders_only_the_visible_window() {
        let lines = input_visual_lines(&"a".repeat(300), 300, 20, 3);
        assert_eq!(lines.len(), 12);
        let lines = input_visual_lines(&"a".repeat(300), 300, 20, 0);
        assert_eq!(lines.len(), 12);
        let lines = input_visual_lines(&"a".repeat(300), 300, 20, 13);
        assert_eq!(lines.len(), 2);
    }

    #[test]
    fn input_overflow_shows_more_in_the_top_border() {
        let mut state = TuiState::new("model".into());
        state.pane_width = 20;
        state.session().input = "a".repeat(300).into();
        state.session().input_cursor = 300;
        clamp_input_scroll(state.session(), 20);
        let backend = TestBackend::new(22, 40);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|frame| draw(frame, &state, 0)).unwrap();
        let buffer = terminal.backend().buffer();
        let top: String = (0..22)
            .map(|x| buffer.cell((x, 26)).unwrap().symbol().to_string())
            .collect();
        assert!(top.contains("↑ 3 more"), "top border: {top}");
        let bottom: String = (0..22)
            .map(|x| buffer.cell((x, 39)).unwrap().symbol().to_string())
            .collect();
        assert!(!bottom.contains("more"), "bottom border: {bottom}");
    }

    #[test]
    fn input_overflow_shows_more_in_both_borders() {
        let mut state = TuiState::new("model".into());
        state.pane_width = 20;
        state.session().input = "a".repeat(400).into();
        state.session().input_cursor = 280;
        clamp_input_scroll(state.session(), 20);
        assert_eq!(state.session().input_scroll, 2);
        let backend = TestBackend::new(22, 40);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|frame| draw(frame, &state, 0)).unwrap();
        let buffer = terminal.backend().buffer();
        let top: String = (0..22)
            .map(|x| buffer.cell((x, 26)).unwrap().symbol().to_string())
            .collect();
        assert!(top.contains("↑ 2 more"), "top border: {top}");
        let bottom: String = (0..22)
            .map(|x| buffer.cell((x, 39)).unwrap().symbol().to_string())
            .collect();
        assert!(bottom.contains("↓ 6 more"), "bottom border: {bottom}");
    }

    #[test]
    fn wide_input_wraps_onto_extra_input_lines() {
        let mut state = TuiState::new("model".into());
        state.pane_width = 20;
        state.session().input = "a".repeat(50).into();
        let backend = TestBackend::new(22, 12);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|frame| draw(frame, &state, 0)).unwrap();
        let buffer = terminal.backend().buffer();
        assert_eq!(buffer.cell((1, 8)).unwrap().symbol(), "a");
        assert_eq!(buffer.cell((20, 8)).unwrap().symbol(), "a");
        assert_eq!(buffer.cell((1, 9)).unwrap().symbol(), "a");
        assert_eq!(buffer.cell((20, 9)).unwrap().symbol(), "a");
        assert_eq!(buffer.cell((1, 10)).unwrap().symbol(), "a");
        assert_eq!(buffer.cell((10, 10)).unwrap().symbol(), "a");
        assert_eq!(buffer.cell((11, 10)).unwrap().symbol(), " ");
    }

    #[test]
    fn cursor_on_a_newline_renders_a_visible_marker() {
        let mut state = TuiState::new("model".into());
        state.pane_width = 20;
        state.session().input = "ab\ncd".into();
        state.session().input_cursor = 2;
        let backend = TestBackend::new(22, 12);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|frame| draw(frame, &state, 0)).unwrap();
        let buffer = terminal.backend().buffer();
        let mut found = false;
        for y in 0..12 {
            for x in 0..22 {
                if buffer.cell((x, y)).unwrap().symbol() == "⏎" {
                    found = true;
                }
            }
        }
        assert!(found, "newline cursor marker not found");
    }

    #[test]
    fn multiline_input_grows_the_box_and_shrinks_the_chat() {
        let mut state = TuiState::new("model".into());
        state.pane_width = 20;
        state.session().input = "a".repeat(300).into();
        let backend = TestBackend::new(22, 40);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|frame| draw(frame, &state, 0)).unwrap();
        let buffer = terminal.backend().buffer();
        assert_eq!(buffer.cell((0, 26)).unwrap().symbol(), "┌");
        assert_eq!(buffer.cell((0, 39)).unwrap().symbol(), "└");
        assert_eq!(buffer.cell((0, 3)).unwrap().symbol(), "┌");
        assert_eq!(buffer.cell((0, 25)).unwrap().symbol(), "└");
    }

    #[test]
    fn cursor_at_end_of_full_line_stays_visible() {
        let mut state = TuiState::new("model".into());
        state.pane_width = 20;
        state.session().input = "a".repeat(20).into();
        state.session().input_cursor = 20;
        let backend = TestBackend::new(22, 12);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|frame| draw(frame, &state, 0)).unwrap();
        let buffer = terminal.backend().buffer();
        for x in 1..=20 {
            assert_eq!(buffer.cell((x, 10)).unwrap().symbol(), "a");
        }
        let style = buffer.cell((20, 10)).unwrap().style();
        assert_eq!(style.bg, Some(Color::Rgb(0xd4, 0xd4, 0xd4)));
        assert_eq!(style.fg, Some(Color::Rgb(0x28, 0x28, 0x32)));
    }

    #[test]
    fn cursor_on_newline_of_full_line_stays_visible() {
        let mut state = TuiState::new("model".into());
        state.pane_width = 20;
        state.session().input = format!("{}\n", "a".repeat(20));
        state.session().input_cursor = 20;
        let backend = TestBackend::new(22, 12);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|frame| draw(frame, &state, 0)).unwrap();
        let buffer = terminal.backend().buffer();
        for x in 1..=20 {
            assert_eq!(buffer.cell((x, 9)).unwrap().symbol(), "a");
        }
        let style = buffer.cell((20, 9)).unwrap().style();
        assert_eq!(style.bg, Some(Color::Rgb(0xd4, 0xd4, 0xd4)));
        let joined: String = (0..12)
            .flat_map(|y| (0..22).map(move |x| buffer.cell((x, y)).unwrap().symbol().to_string()))
            .collect();
        assert!(!joined.contains("⏎"));
    }

    #[test]
    fn label_uses_the_first_nonempty_line_of_multiline_input() {
        let mut state = TuiState::new("model".into());
        state.session().input = "\n  second line  \n".into();
        assert_eq!(
            handle_key(&mut state, &key(KeyCode::Enter)),
            KeyAction::Submit("\n  second line  \n".into())
        );
        assert_eq!(state.session().label, "second line");
    }

    #[test]
    fn enter_with_empty_input_submits_when_the_gate_is_pending() {
        let mut state = TuiState::new("model".into());
        state.session().gate = true;
        assert_eq!(
            handle_key(&mut state, &key(KeyCode::Enter)),
            KeyAction::Submit(String::new())
        );
    }

    #[test]
    fn home_and_end_work_while_running() {
        let mut state = TuiState::new("model".into());
        state.session().running = true;
        state.viewport = 2;
        for _ in 0..3 {
            state.session().renderer.on_event(text("x\ny\nz\nw\n"));
        }
        state.session().renderer.finish();
        state.session().scroller.set_following(true);
        let (pw, vp) = (state.pane_width, state.viewport);
        let max = state.session().max_scroll(pw, vp);
        state.session().scroller.end(max);

        assert_eq!(handle_key(&mut state, &key(KeyCode::Home)), KeyAction::None);
        assert_eq!(state.session().scroller.offset(), 0);
        assert!(!state.session().scroller.following());
        assert_eq!(handle_key(&mut state, &key(KeyCode::End)), KeyAction::None);
        assert_eq!(state.session().scroller.offset(), max);
        assert!(state.session().scroller.following());
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
        assert!(
            rendered[1].spans[1]
                .style
                .add_modifier
                .contains(Modifier::BOLD)
        );
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
        assert!(
            rendered[3].spans[0]
                .style
                .add_modifier
                .contains(Modifier::BOLD)
        );
    }

    #[test]
    fn thinking_with_markdown_is_rendered() {
        let rendered = lines(vec![thinking("**bold** thinking"), text("done")]);

        let thinking_line = &rendered[1];
        assert!(thinking_line.spans.iter().any(|s| {
            s.content.as_ref() == "bold" && s.style.add_modifier.contains(Modifier::BOLD)
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

        let one = rendered
            .iter()
            .find(|l| l.spans.iter().any(|s| s.content.as_ref() == "One"))
            .unwrap();
        let one_span = one
            .spans
            .iter()
            .find(|s| s.content.as_ref() == "One")
            .unwrap();
        assert_eq!(one_span.style.fg, Some(Color::Rgb(0xf0, 0xc6, 0x74)));
        assert!(one_span.style.add_modifier.contains(Modifier::BOLD));
        assert!(one_span.style.add_modifier.contains(Modifier::UNDERLINED));
        assert!(!one.to_string().contains('#'));

        let two = rendered
            .iter()
            .find(|l| l.spans.iter().any(|s| s.content.as_ref() == "Two"))
            .unwrap();
        let two_span = two
            .spans
            .iter()
            .find(|s| s.content.as_ref() == "Two")
            .unwrap();
        assert_eq!(two_span.style.fg, Some(Color::Rgb(0xf0, 0xc6, 0x74)));
        assert!(two_span.style.add_modifier.contains(Modifier::BOLD));
        assert!(!two_span.style.add_modifier.contains(Modifier::UNDERLINED));
    }

    #[test]
    fn links_render_pi_style() {
        let rendered = lines(vec![
            text("[label](https://example.com)"),
            AgentEvent::CompletionStarted,
        ]);

        let label = rendered
            .iter()
            .find(|l| l.spans.iter().any(|s| s.content.as_ref() == "label"))
            .unwrap();
        let label_span = label
            .spans
            .iter()
            .find(|s| s.content.as_ref() == "label")
            .unwrap();
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
        let quoted_span = quoted
            .spans
            .iter()
            .find(|s| s.content.as_ref() == "quoted")
            .unwrap();
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
        assert!(
            rendered[1].spans[1]
                .style
                .add_modifier
                .contains(Modifier::BOLD)
        );
        assert_eq!(rendered[1].spans[1].content, "a");
        assert!(
            rendered[4].spans[1]
                .style
                .add_modifier
                .contains(Modifier::BOLD)
        );
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
        let rendered = lines(vec![text("para\n***\nnext"), AgentEvent::CompletionStarted]);
        let described: Vec<String> = rendered.iter().map(|l| l.to_string()).collect();

        assert!(described.contains(&"***".into()), "{described:?}");
        assert!(
            !described
                .iter()
                .any(|l| l.contains("para ***") || l.contains("*** next")),
            "{described:?}"
        );
    }

    #[test]
    fn rule_like_line_inside_code_fence_is_untouched() {
        let rendered = lines(vec![
            text(
                "```bash\n***\n```
",
            ),
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
        context.messages.push(Message::User {
            content: "do the thing".into(),
        });
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
        context.messages.push(Message::User {
            content: "hello".into(),
        });
        context.total_prompt_tokens = 50;
        context.total_completion_tokens = 5;
        let file = SessionFile {
            label: "fix login".into(),
            context: context.clone(),
            history: vec!["hello".to_string()],
        };

        let state = TuiState::with_sessions("model".into(), vec![(3, file)]);
        assert_eq!(state.sessions.len(), 2);
        assert_eq!(state.sessions[0].label, "fix login");
        assert_eq!(
            state.sessions[0].context.as_ref().unwrap().messages.len(),
            1
        );
        assert_eq!(state.sessions[0].prompt_tokens, 50);
        assert_eq!(state.sessions[0].completion_tokens, 5);
        assert_eq!(state.sessions[0].history, vec!["hello".to_string()]);
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
            .flat_map(|y| (0..60).map(move |x| buffer.cell((x, y)).unwrap().symbol().to_string()))
            .collect();
        assert!(joined.contains("llama"));
        assert!(joined.contains("100 prompt / 5 completion tok"));
        assert!(joined.contains("[1/1]"));
        assert!(joined.contains("hello"));
    }

    fn complete_request(data: &[u8]) -> Option<String> {
        let header_end = data.windows(4).position(|w| w == b"\r\n\r\n")?;
        let headers = std::str::from_utf8(&data[..header_end]).unwrap();
        let length = headers.lines().find_map(|line| {
            let (name, value) = line.split_once(':')?;
            if name.trim().eq_ignore_ascii_case("content-length") {
                value.trim().parse::<usize>().ok()
            } else {
                None
            }
        })?;
        let body_start = header_end + 4;
        if data.len() < body_start + length {
            return None;
        }
        Some(String::from_utf8_lossy(&data[body_start..body_start + length]).into_owned())
    }

    async fn wait_for_event(
        rx: &mut tokio::sync::mpsc::UnboundedReceiver<TuiEvent>,
        events: &mut Vec<TuiEvent>,
        predicate: impl Fn(&TuiEvent) -> bool,
    ) -> bool {
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(10);
        while tokio::time::Instant::now() < deadline {
            if let Ok(Some(event)) =
                tokio::time::timeout(std::time::Duration::from_secs(1), rx.recv()).await
            {
                if predicate(&event) {
                    events.push(event);
                    return true;
                }
                events.push(event);
            }
        }
        false
    }

    #[tokio::test]
    async fn plan_gate_approval_runs_the_implement_stage_then_resets() {
        let (tx_req, mut rx_req) = tokio::sync::mpsc::unbounded_channel::<String>();
        let responses = Arc::new(Mutex::new(std::collections::VecDeque::from([
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"c1\",\"type\":\"function\",\"function\":{\"name\":\"submit_plan\",\"arguments\":\"{\\\"stages\\\":[{\\\"title\\\":\\\"step one\\\",\\\"tasks\\\":[\\\"do it\\\"]}]}\"}}]}}]}\n\ndata: [DONE]\n\n".to_string(),
            "data: {\"choices\":[{\"delta\":{\"content\":\"implemented\"}}]}\n\ndata: [DONE]\n\n".to_string(),
        ])));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            while let Ok((mut socket, _)) = listener.accept().await {
                let tx = tx_req.clone();
                let responses = responses.clone();
                tokio::spawn(async move {
                    let mut data = Vec::new();
                    let mut buf = [0u8; 8192];
                    loop {
                        match socket.read(&mut buf).await {
                            Ok(0) | Err(_) => break,
                            Ok(n) => {
                                data.extend_from_slice(&buf[..n]);
                                if let Some(body) = complete_request(&data) {
                                    let _ = tx.send(body);
                                    let sse = match responses.lock().unwrap().pop_front() {
                                        Some(response) => response,
                                        None => "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\ndata: [DONE]\n\n".to_string(),
                                    };
                                    let response = format!(
                                        "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\nconnection: close\r\n\r\n{sse}"
                                    );
                                    if socket.write_all(response.as_bytes()).await.is_err() {
                                        break;
                                    }
                                    break;
                                }
                            }
                        }
                    }
                });
            }
        });

        let client = OpenAI::with_config(
            openai_oxide::ClientConfig::new("local").base_url(format!("http://{addr}")),
        );
        let shared = Arc::new(Mutex::new(Mode::Yolo));
        let factory = Arc::new(move || {
            Agent::new(
                client.clone(),
                "test-model",
                shared.clone(),
                10000,
                Duration::from_secs(30),
            )
            .with_pinned_mode(Mode::Plan)
        });
        let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel::<TuiEvent>();
        let (input_tx, handle) = spawn_agent(1, factory, None, event_tx);

        input_tx.send("plan me a feature".to_string()).unwrap();
        let mut events = vec![];
        assert!(
            wait_for_event(&mut event_rx, &mut events, |e| matches!(
                e,
                TuiEvent::GatePending { .. }
            ))
            .await
        );
        input_tx.send(String::new()).unwrap();
        assert!(
            wait_for_event(&mut event_rx, &mut events, |e| matches!(
                e,
                TuiEvent::StageChanged { mode: None, .. }
            ))
            .await
        );
        handle.abort();

        let mut requests = vec![];
        while let Ok(body) = rx_req.try_recv() {
            requests.push(body);
        }
        let implement_request = requests
            .into_iter()
            .find(|body| body.contains("Execute Step 1 of 1"))
            .expect("the implement stage request carries the plan");
        assert!(implement_request.contains("step one"));
        assert!(events.iter().any(|e| matches!(
            e,
            TuiEvent::StageChanged {
                mode: Some(Mode::Implement),
                ..
            }
        )));
        assert!(
            events
                .iter()
                .any(|e| matches!(e, TuiEvent::TurnDone { .. }))
        );
    }

    async fn staged_mock_server(responses: Vec<String>) -> (String, tokio::sync::mpsc::UnboundedReceiver<String>) {
        let (tx_req, rx_req) = tokio::sync::mpsc::unbounded_channel::<String>();
        let responses = Arc::new(Mutex::new(std::collections::VecDeque::from(responses)));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            while let Ok((mut socket, _)) = listener.accept().await {
                let tx = tx_req.clone();
                let responses = responses.clone();
                tokio::spawn(async move {
                    let mut data = Vec::new();
                    let mut buf = [0u8; 8192];
                    loop {
                        match socket.read(&mut buf).await {
                            Ok(0) | Err(_) => break,
                            Ok(n) => {
                                data.extend_from_slice(&buf[..n]);
                                if let Some(body) = complete_request(&data) {
                                    let _ = tx.send(body);
                                    let sse = match responses.lock().unwrap().pop_front() {
                                        Some(response) => response,
                                        None => "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\ndata: [DONE]\n\n".to_string(),
                                    };
                                    let response = format!(
                                        "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\nconnection: close\r\n\r\n{sse}"
                                    );
                                    if socket.write_all(response.as_bytes()).await.is_err() {
                                        break;
                                    }
                                    break;
                                }
                            }
                        }
                    }
                });
            }
        });
        (format!("http://{addr}"), rx_req)
    }

    fn plan_sse(id: &str, stages: &str) -> String {
        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"__ID__\",\"type\":\"function\",\"function\":{\"name\":\"submit_plan\",\"arguments\":\"__STAGES__\"}}]}}]}\n\ndata: [DONE]\n\n"
            .replace("__ID__", id)
            .replace("__STAGES__", stages)
    }

    #[tokio::test]
    async fn staged_plan_walks_through_every_stage_with_a_review_gate() {
        let stages = r#"{\"stages\":[{\"title\":\"data\",\"tasks\":[\"entity\"]},{\"title\":\"api\",\"tasks\":[\"controller\"]}]}"#;
        let (base_url, mut rx_req) = staged_mock_server(vec![
            plan_sse("c1", stages),
            "data: {\"choices\":[{\"delta\":{\"content\":\"stage one done\"}}]}\n\ndata: [DONE]\n\n".to_string(),
            "data: {\"choices\":[{\"delta\":{\"content\":\"stage two done\"}}]}\n\ndata: [DONE]\n\n".to_string(),
        ]).await;
        let client = OpenAI::with_config(
            openai_oxide::ClientConfig::new("local").base_url(base_url),
        );
        let shared = Arc::new(Mutex::new(Mode::Yolo));
        let factory = Arc::new(move || {
            Agent::new(
                client.clone(),
                "test-model",
                shared.clone(),
                10000,
                Duration::from_secs(30),
            )
            .with_pinned_mode(Mode::Plan)
        });
        let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel::<TuiEvent>();
        let (input_tx, handle) = spawn_agent(1, factory, None, event_tx);
        let mut events = vec![];

        input_tx.send("plan me".to_string()).unwrap();
        assert!(
            wait_for_event(&mut event_rx, &mut events, |e| matches!(
                e,
                TuiEvent::GatePending { message, .. } if message.contains("plan ready (2 stages)")
                    && message.contains("Step 1: data")
            ))
            .await
        );

        input_tx.send(String::new()).unwrap();
        assert!(
            wait_for_event(&mut event_rx, &mut events, |e| matches!(
                e,
                TuiEvent::GatePending { message, .. } if message.contains("Step 1 implemented")
                    && message.contains("review the implementation")
            ))
            .await
        );
        assert!(
            !events
                .iter()
                .rev()
                .take(3)
                .any(|e| matches!(e, TuiEvent::StageChanged { mode: Some(Mode::Plan), .. }))
        );

        input_tx.send(String::new()).unwrap();
        assert!(
            wait_for_event(&mut event_rx, &mut events, |e| matches!(
                e,
                TuiEvent::GatePending { message, .. } if message.contains("review Step 2: api")
            ))
            .await
        );

        input_tx.send(String::new()).unwrap();
        assert!(
            wait_for_event(&mut event_rx, &mut events, |e| matches!(
                e,
                TuiEvent::StageChanged { mode: None, .. }
            ))
            .await
        );
        handle.abort();

        let mut requests = vec![];
        while let Ok(body) = rx_req.try_recv() {
            requests.push(body);
        }
        assert!(requests.iter().any(|body| body.contains("Execute Step 1 of 2: data")));
        assert!(requests.iter().any(|body| body.contains("Do not start later stages")));
        assert!(requests.iter().any(|body| body.contains("Execute Step 2 of 2: api")));
        assert!(requests.iter().any(|body| body.contains("Stages already done: Step 1: data")));
    }

    #[tokio::test]
    async fn review_gate_feedback_replans_the_remaining_stages() {
        let stages = r#"{\"stages\":[{\"title\":\"data\",\"tasks\":[\"entity\"]},{\"title\":\"api\",\"tasks\":[\"controller\"]}]}"#;
        let (base_url, mut rx_req) = staged_mock_server(vec![
            plan_sse("c1", stages),
            "data: {\"choices\":[{\"delta\":{\"content\":\"stage one done\"}}]}\n\ndata: [DONE]\n\n".to_string(),
        ]).await;
        let client = OpenAI::with_config(
            openai_oxide::ClientConfig::new("local").base_url(base_url),
        );
        let shared = Arc::new(Mutex::new(Mode::Yolo));
        let factory = Arc::new(move || {
            Agent::new(
                client.clone(),
                "test-model",
                shared.clone(),
                10000,
                Duration::from_secs(30),
            )
            .with_pinned_mode(Mode::Plan)
        });
        let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel::<TuiEvent>();
        let (input_tx, handle) = spawn_agent(1, factory, None, event_tx);
        let mut events = vec![];

        input_tx.send("plan me".to_string()).unwrap();
        assert!(
            wait_for_event(&mut event_rx, &mut events, |e| matches!(
                e,
                TuiEvent::GatePending { .. }
            ))
            .await
        );
        input_tx.send(String::new()).unwrap();
        assert!(
            wait_for_event(&mut event_rx, &mut events, |e| matches!(
                e,
                TuiEvent::GatePending { message, .. } if message.contains("Step 1 implemented")
            ))
            .await
        );
        input_tx.send(String::new()).unwrap();
        assert!(
            wait_for_event(&mut event_rx, &mut events, |e| matches!(
                e,
                TuiEvent::GatePending { message, .. } if message.contains("review Step 2: api")
            ))
            .await
        );
        input_tx.send("api should be rest".to_string()).unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;
        handle.abort();

        let mut requests = vec![];
        while let Ok(body) = rx_req.try_recv() {
            requests.push(body);
        }
        let replan = requests
            .iter()
            .find(|body| body.contains("Feedback on the remaining stages"))
            .expect("the review gate feedback re-plans the remaining stages");
        assert!(replan.contains("api should be rest"));
        assert!(replan.contains("Step 1 of 2 is done"));
        assert!(replan.contains("Revise the remaining stages"));
    }

    #[tokio::test]
    async fn implementation_gate_feedback_replans_from_the_current_stage() {
        let stages = r#"{\"stages\":[{\"title\":\"data\",\"tasks\":[\"entity\"]},{\"title\":\"api\",\"tasks\":[\"controller\"]}]}"#;
        let (base_url, mut rx_req) = staged_mock_server(vec![
            plan_sse("c1", stages),
            "data: {\"choices\":[{\"delta\":{\"content\":\"stage one done\"}}]}\n\ndata: [DONE]\n\n".to_string(),
        ]).await;
        let client = OpenAI::with_config(
            openai_oxide::ClientConfig::new("local").base_url(base_url),
        );
        let shared = Arc::new(Mutex::new(Mode::Yolo));
        let factory = Arc::new(move || {
            Agent::new(
                client.clone(),
                "test-model",
                shared.clone(),
                10000,
                Duration::from_secs(30),
            )
            .with_pinned_mode(Mode::Plan)
        });
        let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel::<TuiEvent>();
        let (input_tx, handle) = spawn_agent(1, factory, None, event_tx);
        let mut events = vec![];

        input_tx.send("plan me".to_string()).unwrap();
        assert!(
            wait_for_event(&mut event_rx, &mut events, |e| matches!(
                e,
                TuiEvent::GatePending { .. }
            ))
            .await
        );
        input_tx.send(String::new()).unwrap();
        assert!(
            wait_for_event(&mut event_rx, &mut events, |e| matches!(
                e,
                TuiEvent::GatePending { message, .. } if message.contains("review the implementation")
            ))
            .await
        );
        input_tx.send("the entity is wrong".to_string()).unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;
        handle.abort();

        let mut requests = vec![];
        while let Ok(body) = rx_req.try_recv() {
            requests.push(body);
        }
        let replan = requests
            .iter()
            .find(|body| body.contains("the user's feedback"))
            .expect("the implementation gate feedback re-plans from the current stage");
        assert!(replan.contains("the entity is wrong"));
        assert!(replan.contains("Step 1 of 2 was implemented"));
        assert!(replan.contains("redo this stage if needed"));
    }

    fn parse(bytes: &[u8]) -> Vec<TermEvent> {
        let mut parser = InputParser::new();
        parser.feed(bytes)
    }

    fn parse_split(bytes: &[u8], at: usize) -> Vec<TermEvent> {
        let mut parser = InputParser::new();
        let mut events = parser.feed(&bytes[..at]);
        events.extend(parser.feed(&bytes[at..]));
        events
    }

    fn key_parts(event: &TermEvent) -> (KeyCode, KeyModifiers) {
        match event {
            TermEvent::Key(k) => (k.code, k.modifiers),
            _ => panic!("expected key"),
        }
    }

    #[test]
    fn parser_modify_other_keys_shift_enter() {
        let events = parse(b"\x1b[27;2;13~");
        assert_eq!(events.len(), 1);
        assert_eq!(key_parts(&events[0]), (KeyCode::Enter, KeyModifiers::SHIFT));
    }

    #[test]
    fn parser_modify_other_keys_ctrl_backspace() {
        let events = parse(b"\x1b[27;4;127~");
        assert_eq!(events.len(), 1);
        assert_eq!(key_parts(&events[0]), (KeyCode::Backspace, KeyModifiers::CONTROL));
    }

    #[test]
    fn parser_modify_other_keys_shift_letter() {
        let events = parse(b"\x1b[27;2;97~");
        assert_eq!(events.len(), 1);
        assert_eq!(key_parts(&events[0]), (KeyCode::Char('a'), KeyModifiers::SHIFT));
    }

    #[test]
    fn parser_kitty_shift_enter() {
        let events = parse(b"\x1b[13;2u");
        assert_eq!(events.len(), 1);
        assert_eq!(key_parts(&events[0]), (KeyCode::Enter, KeyModifiers::SHIFT));
    }

    #[test]
    fn parser_kitty_ctrl_c() {
        let events = parse(b"\x1b[99;5u");
        assert_eq!(events.len(), 1);
        assert_eq!(key_parts(&events[0]), (KeyCode::Char('c'), KeyModifiers::CONTROL));
    }

    #[test]
    fn parser_plain_enter_and_lf() {
        let events = parse(b"\r\n");
        assert_eq!(events.len(), 2);
        assert_eq!(key_parts(&events[0]), (KeyCode::Enter, KeyModifiers::NONE));
        assert_eq!(key_parts(&events[1]), (KeyCode::Char('j'), KeyModifiers::CONTROL));
    }

    #[test]
    fn parser_ctrl_byte() {
        let events = parse(b"\x03");
        assert_eq!(events.len(), 1);
        assert_eq!(key_parts(&events[0]), (KeyCode::Char('c'), KeyModifiers::CONTROL));
    }

    #[test]
    fn parser_arrow_with_modifier() {
        let events = parse(b"\x1b[1;2A");
        assert_eq!(events.len(), 1);
        assert_eq!(key_parts(&events[0]), (KeyCode::Up, KeyModifiers::SHIFT));
    }

    #[test]
    fn parser_alt_enter() {
        let events = parse(b"\x1b\r");
        assert_eq!(events.len(), 1);
        assert_eq!(key_parts(&events[0]), (KeyCode::Enter, KeyModifiers::ALT));
    }

    #[test]
    fn parser_split_sequence() {
        let events = parse_split(b"\x1b[27;2;13~", 3);
        assert_eq!(events.len(), 1);
        assert_eq!(key_parts(&events[0]), (KeyCode::Enter, KeyModifiers::SHIFT));
    }

    #[test]
    fn parser_bare_esc_flush() {
        let mut parser = InputParser::new();
        assert!(parser.feed(b"\x1b").is_empty());
        let events = parser.flush_escape();
        assert_eq!(events.len(), 1);
        assert_eq!(key_parts(&events[0]), (KeyCode::Esc, KeyModifiers::NONE));
    }

    #[test]
    fn parser_protocol_responses_dropped() {
        assert!(parse(b"\x1b[?1;2;4c\x1b[?7u").is_empty());
    }

    #[test]
    fn parser_mouse_scroll() {
        let events = parse(b"\x1b[<64;1;1M");
        assert_eq!(events.len(), 1);
        assert!(matches!(
            events[0],
            TermEvent::Mouse(MouseEvent { kind: MouseEventKind::ScrollUp })
        ));
    }

    #[test]
    fn parser_utf8_char() {
        let events = parse("é".as_bytes());
        assert_eq!(events.len(), 1);
        assert_eq!(key_parts(&events[0]), (KeyCode::Char('é'), KeyModifiers::NONE));
    }

    #[test]
    fn termwiz_parsers_shift_enter() {
        let mut parser = termwiz::input::InputParser::new();
        let events = parser.parse_as_vec(b"\x1b[27;2;13~", false);
        assert_eq!(events.len(), 1);
        let termwiz::input::InputEvent::Key(key_event) = &events[0] else {
            panic!("expected key event");
        };
        assert_eq!(key_event.key, termwiz::input::KeyCode::Enter);
        assert!(key_event.modifiers.contains(termwiz::input::Modifiers::SHIFT));
        let mapped = map_termwiz_event(events[0].clone()).unwrap();
        assert_eq!(
            key_parts(&mapped),
            (KeyCode::Enter, KeyModifiers::SHIFT)
        );
    }

    #[test]
    fn up_recalls_after_error() {
        let mut state = TuiState::new("model".into());
        state.session().error = Some("request error".into());
        state.session().running = false;
        state.session().history = vec!["hello".into()];
        handle_key(&mut state, &key(KeyCode::Up));
        assert_eq!(state.session().input, "hello");
        assert_eq!(state.session().error, None);
    }

    #[test]
    fn termwiz_parsers_kitty_shift_enter() {
        let mut parser = termwiz::input::InputParser::new();
        let events = parser.parse_as_vec(b"\x1b[13;2u", false);
        assert_eq!(events.len(), 1);
        let mapped = map_termwiz_event(events[0].clone()).unwrap();
        assert_eq!(key_parts(&mapped), (KeyCode::Enter, KeyModifiers::SHIFT));
    }
}
