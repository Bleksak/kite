use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};

use crate::session::{Cursor, Session};
use crate::tui::{KeyAction, KeyCode, KeyEvent, KeyModifiers, TuiState};

pub struct State {
    pub open: bool,
    pub cursor: Cursor,
    pub query: String,
    pub rename: Option<String>,
}

impl Default for State {
    fn default() -> Self {
        Self {
            open: false,
            cursor: Cursor::default(),
            query: String::new(),
            rename: None,
        }
    }
}

pub fn handle_key(state: &mut TuiState, key: &KeyEvent) -> Option<KeyAction> {
    if !state.picker.open {
        return None;
    }
    if state.picker.rename.is_some() {
        return Some(match key.code {
            KeyCode::Enter => {
                let buf = state.picker.rename.take().unwrap();
                if let Some(i) = state.filtered().get(state.picker.cursor.pos).copied() {
                    state.sessions[i].label = buf;
                }
                KeyAction::None
            }
            KeyCode::Esc => {
                state.picker.rename = None;
                KeyAction::None
            }
            KeyCode::Backspace => {
                if let Some(buf) = &mut state.picker.rename {
                    buf.pop();
                }
                KeyAction::None
            }
            KeyCode::Char(c) => {
                if let Some(buf) = &mut state.picker.rename {
                    buf.push(c);
                }
                KeyAction::None
            }
            _ => KeyAction::None,
        });
    }
    if key.modifiers.contains(KeyModifiers::CONTROL) {
        return Some(match key.code {
            KeyCode::Char('j') => {
                state.picker.cursor.down(state.filtered().len());
                KeyAction::None
            }
            KeyCode::Char('k') => {
                state.picker.cursor.up(state.filtered().len());
                KeyAction::None
            }
            KeyCode::Char('n') => {
                state.sessions.push(Session::new(state.next_id));
                state.active = state.sessions.len() - 1;
                state.next_id += 1;
                state.picker.open = false;
                KeyAction::NewSession
            }
            KeyCode::Char('x') => {
                let filtered = state.filtered();
                if let Some(i) = filtered.get(state.picker.cursor.pos).copied()
                    && state.sessions.len() > 1
                    && !state.sessions[i].running
                {
                    state.sessions.remove(i);
                    if state.active >= state.sessions.len() {
                        state.active = state.sessions.len() - 1;
                    }
                }
                state.picker.cursor.clamp(state.filtered().len());
                KeyAction::CloseSession
            }
            KeyCode::Char('r') => {
                state.picker.rename = Some(String::new());
                KeyAction::None
            }
            KeyCode::Char('q') => {
                state.task_list.open = true;
                state.picker.open = false;
                KeyAction::None
            }
            KeyCode::Char('s') => {
                state.picker.open = false;
                KeyAction::None
            }
            _ => KeyAction::None,
        });
    }
    Some(match key.code {
        KeyCode::Up => {
            state.picker.cursor.up(state.filtered().len());
            KeyAction::None
        }
        KeyCode::Down => {
            state.picker.cursor.down(state.filtered().len());
            KeyAction::None
        }
        KeyCode::Enter => {
            let filtered = state.filtered();
            state.picker.open = false;
            match filtered.get(state.picker.cursor.pos).copied() {
                Some(i) => KeyAction::PickerSelected(i),
                None => KeyAction::None,
            }
        }
        KeyCode::Esc => {
            state.picker.open = false;
            KeyAction::None
        }
        KeyCode::Backspace => {
            state.picker.query.pop();
            state.picker.cursor.clamp(state.filtered().len());
            KeyAction::None
        }
        KeyCode::Char(c) => {
            state.picker.query.push(c);
            state.picker.cursor.clamp(state.filtered().len());
            KeyAction::None
        }
        _ => KeyAction::None,
    })
}

pub fn draw(frame: &mut Frame, state: &TuiState, area: Rect) {
    let filtered = state.filtered();
    let renaming = state.picker.rename.clone();
    let height = area.height;
    let visible = height.saturating_sub(3) as usize;
    let start = if filtered.len() <= visible {
        0
    } else {
        let mut s = state.picker.cursor.pos.saturating_sub(visible / 2);
        if s + visible > filtered.len() {
            s = filtered.len() - visible;
        }
        s
    };
    let lines: Vec<Line> = if filtered.is_empty() {
        vec![Line::from(Span::styled(
            " no matches",
            Style::default().fg(Color::Rgb(102, 102, 102)),
        ))]
    } else {
        filtered
            .iter()
            .enumerate()
            .skip(start)
            .take(visible)
            .map(|(i, session_idx)| {
                let selected = i + start == state.picker.cursor.pos;
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
                            buf.chars().take(60).collect()
                        }
                    }
                    _ => {
                        if session.label.is_empty() {
                            "—".to_string()
                        } else {
                            session.label.chars().take(60).collect()
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
    } else if state.picker.query.is_empty() {
        " sessions".to_string()
    } else {
        format!(" sessions · {}", state.picker.query)
    };
    crate::tui::render_panel(
        frame,
        area,
        format!(" {} ", title),
        lines
            .into_iter()
            .chain(std::iter::once(hint))
            .collect::<Vec<Line>>(),
        false,
    );
}
