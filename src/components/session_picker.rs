use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};

use crate::event::keyboard::{KeyCode, KeyEvent, KeyModifiers};
use crate::screen::Screen;
use crate::session::{Cursor, Session};
use crate::tui::{filtered, KeyAction, TuiState};

pub struct State {
    pub cursor: Cursor,
    pub query: String,
    pub rename: Option<String>,
}

impl Default for State {
    fn default() -> Self {
        Self {
            cursor: Cursor::default(),
            query: String::new(),
            rename: None,
        }
    }
}

pub fn handle_key(state: &mut TuiState, key: &KeyEvent) -> Option<KeyAction> {
    let TuiState {
        screen,
        sessions,
        active,
        next_id,
        ..
    } = state;
    let (picker, list) = match screen {
        Screen::SessionPicker { picker, list } => (picker, list),
        _ => return None,
    };
    if picker.rename.is_some() {
        return Some(match key.code {
            KeyCode::Enter => {
                let buf = picker.rename.take().unwrap();
                if let Some(i) = filtered(sessions, &picker.query).get(picker.cursor.pos).copied()
                {
                    sessions[i].label = buf;
                }
                KeyAction::None
            }
            KeyCode::Esc => {
                picker.rename = None;
                KeyAction::None
            }
            KeyCode::Backspace => {
                if let Some(buf) = &mut picker.rename {
                    buf.pop();
                }
                KeyAction::None
            }
            KeyCode::Char(c) => {
                if let Some(buf) = &mut picker.rename {
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
                picker.cursor.down(filtered(sessions, &picker.query).len());
                KeyAction::None
            }
            KeyCode::Char('k') => {
                picker.cursor.up(filtered(sessions, &picker.query).len());
                KeyAction::None
            }
            KeyCode::Char('n') => {
                sessions.push(Session::new(*next_id));
                *active = sessions.len() - 1;
                *next_id += 1;
                *screen = Screen::Chat;
                KeyAction::NewSession
            }
            KeyCode::Char('x') => {
                let filtered_list = filtered(sessions, &picker.query);
                if let Some(i) = filtered_list.get(picker.cursor.pos).copied()
                    && sessions.len() > 1
                    && !sessions[i].running
                {
                    sessions.remove(i);
                    if *active >= sessions.len() {
                        *active = sessions.len() - 1;
                    }
                }
                picker.cursor.clamp(filtered(sessions, &picker.query).len());
                KeyAction::CloseSession
            }
            KeyCode::Char('r') => {
                picker.rename = Some(String::new());
                KeyAction::None
            }
            KeyCode::Char('q') => {
                *screen = Screen::TaskList(list.clone());
                KeyAction::None
            }
            KeyCode::Char('s') => {
                *screen = Screen::Chat;
                KeyAction::None
            }
            _ => KeyAction::None,
        });
    }
    Some(match key.code {
        KeyCode::Up => {
            picker.cursor.up(filtered(sessions, &picker.query).len());
            KeyAction::None
        }
        KeyCode::Down => {
            picker.cursor.down(filtered(sessions, &picker.query).len());
            KeyAction::None
        }
        KeyCode::Enter => {
            let filtered = filtered(sessions, &picker.query);
            let selected = filtered.get(picker.cursor.pos).copied();
            *screen = Screen::Chat;
            match selected {
                Some(i) => KeyAction::PickerSelected(i),
                None => KeyAction::None,
            }
        }
        KeyCode::Esc => {
            *screen = Screen::Chat;
            KeyAction::None
        }
        KeyCode::Backspace => {
            picker.query.pop();
            picker.cursor.clamp(filtered(sessions, &picker.query).len());
            KeyAction::None
        }
        KeyCode::Char(c) => {
            picker.query.push(c);
            picker.cursor.clamp(filtered(sessions, &picker.query).len());
            KeyAction::None
        }
        _ => KeyAction::None,
    })
}

pub fn draw(frame: &mut Frame, state: &TuiState, area: Rect) {
    let TuiState { screen, sessions, .. } = state;
    let picker = match screen {
        Screen::SessionPicker { picker, .. } => picker,
        _ => return,
    };
    let filtered = filtered(sessions, &picker.query);
    let renaming = picker.rename.clone();
    let height = area.height;
    let visible = height.saturating_sub(3) as usize;
    let start = if filtered.len() <= visible {
        0
    } else {
        let mut s = picker.cursor.pos.saturating_sub(visible / 2);
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
                let selected = i + start == picker.cursor.pos;
                let session = &sessions[*session_idx];
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
    } else if picker.query.is_empty() {
        " sessions".to_string()
    } else {
        format!(" sessions · {}", picker.query)
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
