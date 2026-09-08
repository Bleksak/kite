use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};

use crate::event::keyboard::{KeyCode, KeyEvent, KeyModifiers};
use crate::screen::Screen;
use crate::session::{Cursor, Scroller};
use crate::tui::{KeyAction, TuiState};

#[derive(Clone)]
pub struct State {
    pub cursor: Cursor,
}

impl Default for State {
    fn default() -> Self {
        Self {
            cursor: Cursor::default(),
        }
    }
}

pub fn handle_key(state: &mut TuiState, key: &KeyEvent) -> Option<KeyAction> {
    let TuiState { screen, active, .. } = state;
    let list = match screen {
        Screen::TaskList(list) => list,
        _ => return None,
    };
    let tasks = crate::bg::REGISTRY.list();
    let len = tasks.len();
    if key.modifiers.contains(KeyModifiers::CONTROL) {
        return Some(match key.code {
            KeyCode::Char('j') => {
                list.cursor.down(len);
                KeyAction::None
            }
            KeyCode::Char('k') => {
                list.cursor.up(len);
                KeyAction::None
            }
            KeyCode::Char('s') => {
                let mut picker = crate::components::session_picker::State::default();
                picker.cursor.set(*active);
                let carried = list.clone();
                *screen = Screen::SessionPicker {
                    picker,
                    list: carried,
                };
                KeyAction::None
            }
            KeyCode::Char('q') => {
                *screen = Screen::Chat;
                KeyAction::None
            }
            _ => KeyAction::None,
        });
    }
    Some(match key.code {
        KeyCode::Up => {
            list.cursor.up(len);
            KeyAction::None
        }
        KeyCode::Down => {
            list.cursor.down(len);
            KeyAction::None
        }
        KeyCode::Char('j') => {
            list.cursor.down(len);
            KeyAction::None
        }
        KeyCode::Char('k') => {
            list.cursor.up(len);
            KeyAction::None
        }
        KeyCode::Char('x') => {
            if let Some(task) = tasks.get(list.cursor.pos) {
                let _ = crate::bg::REGISTRY.kill(&task.id);
            }
            KeyAction::None
        }
        KeyCode::Enter => {
            let pos = list.cursor.pos;
            if let Some(task) = tasks.get(pos) {
                let id = task.id.clone();
                let carried = list.clone();
                *screen = Screen::TaskDetail {
                    detail: crate::components::task_detail::State {
                        task_id: id,
                        scroll: Scroller::at_tail(),
                    },
                    list: carried,
                };
            }
            KeyAction::None
        }
        KeyCode::Char('q') | KeyCode::Esc => {
            *screen = Screen::Chat;
            KeyAction::None
        }
        _ => KeyAction::None,
    })
}

pub fn draw(frame: &mut Frame, state: &TuiState, area: Rect) {
    let list = match &state.screen {
        Screen::TaskList(list) => list,
        _ => return,
    };
    let tasks = crate::bg::REGISTRY.list();
    let height = area.height;
    let visible = height.saturating_sub(3) as usize;
    let start = if tasks.len() <= visible {
        0
    } else {
        let mut s = list.cursor.pos.saturating_sub(visible / 2);
        if s + visible > tasks.len() {
            s = tasks.len() - visible;
        }
        s
    };
    let lines: Vec<Line> = if tasks.is_empty() {
        vec![Line::from(Span::styled(
            " no tasks",
            Style::default().fg(Color::Rgb(102, 102, 102)),
        ))]
    } else {
        tasks
            .iter()
            .enumerate()
            .skip(start)
            .take(visible)
            .map(|(i, task)| {
                let selected = i + start == list.cursor.pos;
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
                let command: String = task.command.chars().take(60).collect();
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
    crate::tui::render_panel(
        frame,
        area,
        " background tasks ".to_string(),
        lines
            .into_iter()
            .chain(std::iter::once(hint))
            .collect::<Vec<Line>>(),
        false,
    );
}
