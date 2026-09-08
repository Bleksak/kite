use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};

use crate::event::keyboard::{KeyCode, KeyEvent, KeyModifiers, MouseEvent, MouseEventKind};
use crate::screen::Screen;
use crate::session::Scroller;
use crate::tui::{KeyAction, TuiState};

pub struct State {
    pub task_id: String,
    pub scroll: Scroller,
}

fn scroll_max(task_id: &str, viewport: usize) -> usize {
    let Some(task) = crate::bg::REGISTRY.list().into_iter().find(|t| t.id == task_id) else {
        return 0;
    };
    let text = std::fs::read_to_string(&task.output_path).unwrap_or_default();
    let total = text.lines().count();
    let visible = viewport.saturating_sub(10);
    total.saturating_sub(visible)
}

pub fn mouse(state: &mut TuiState, mouse: &MouseEvent) -> Option<KeyAction> {
    let viewport = state.viewport;
    let detail = match &mut state.screen {
        Screen::TaskDetail { detail, .. } => detail,
        _ => return None,
    };
    let max = scroll_max(&detail.task_id, viewport);
    Some(match mouse.kind {
        MouseEventKind::ScrollUp => {
            detail.scroll.toward_top(3);
            KeyAction::None
        }
        MouseEventKind::ScrollDown => {
            detail.scroll.toward_bottom(3, max);
            KeyAction::None
        }
    })
}

pub fn handle_key(state: &mut TuiState, key: &KeyEvent) -> Option<KeyAction> {
    let viewport = state.viewport;
    let detail = match &mut state.screen {
        Screen::TaskDetail { detail, .. } => detail,
        _ => return None,
    };
    if key.modifiers.contains(KeyModifiers::CONTROL) {
        return Some(match key.code {
            KeyCode::Char('q') => {
                state.screen = Screen::Chat;
                KeyAction::None
            }
            _ => KeyAction::None,
        });
    }
    let scroll_max = scroll_max(&detail.task_id, viewport);
    Some(match key.code {
        KeyCode::Esc | KeyCode::Char('q') => {
            let list = match std::mem::replace(&mut state.screen, Screen::Chat) {
                Screen::TaskDetail { list, .. } => list,
                _ => unreachable!(),
            };
            state.screen = Screen::TaskList(list);
            KeyAction::None
        }
        KeyCode::Char('j') | KeyCode::Down => {
            detail.scroll.toward_bottom(3, scroll_max);
            KeyAction::None
        }
        KeyCode::Char('k') | KeyCode::Up => {
            detail.scroll.toward_top(3);
            KeyAction::None
        }
        KeyCode::PageDown => {
            detail.scroll.toward_bottom(8, scroll_max);
            KeyAction::None
        }
        KeyCode::PageUp => {
            detail.scroll.toward_top(8);
            KeyAction::None
        }
        KeyCode::Home => {
            detail.scroll.home();
            KeyAction::None
        }
        KeyCode::End => {
            detail.scroll.end(scroll_max);
            KeyAction::None
        }
        _ => KeyAction::None,
    })
}

pub fn draw(frame: &mut Frame, state: &TuiState, area: Rect) {
    let detail = match &state.screen {
        Screen::TaskDetail { detail, .. } => detail,
        _ => return,
    };
    let tasks = crate::bg::REGISTRY.list();
    let Some(task) = tasks.iter().find(|t| t.id == detail.task_id) else {
        return;
    };
    let text = std::fs::read_to_string(&task.output_path).unwrap_or_default();
    let lines: Vec<&str> = text.lines().collect();
    let visible = area.height as usize - 4;
    let total = lines.len();
    let max = total.saturating_sub(visible);
    let start = if detail.scroll.following() {
        max
    } else {
        detail.scroll.offset().min(max)
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
    crate::tui::render_panel(
        frame,
        area,
        format!(" task {} output ", task.id),
        body
            .into_iter()
            .chain(std::iter::once(hint))
            .collect::<Vec<Line>>(),
        true,
    );
}
