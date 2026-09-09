use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};

use crate::event::keyboard::{KeyCode, KeyEvent, KeyModifiers, MouseEvent, MouseEventKind};
use crate::screen::Screen;
use crate::tui::{KeyAction, TuiState};

fn scroll_max(body: &str, viewport: usize) -> usize {
    let total = body.lines().count();
    let visible = viewport.saturating_sub(10);
    total.saturating_sub(visible)
}

pub fn mouse(state: &mut TuiState, mouse: &MouseEvent) -> Option<KeyAction> {
    let viewport = state.viewport;
    let max = {
        let body = match &state.screen {
            Screen::ToolDetail { body, .. } => body,
            _ => return None,
        };
        scroll_max(body, viewport)
    };
    let scroll = match &mut state.screen {
        Screen::ToolDetail { scroll, .. } => scroll,
        _ => return None,
    };
    Some(match mouse.kind {
        MouseEventKind::ScrollUp => {
            scroll.toward_top(3);
            KeyAction::None
        }
        MouseEventKind::ScrollDown => {
            scroll.toward_bottom(3, max);
            KeyAction::None
        }
        _ => KeyAction::None,
    })
}

pub fn handle_key(state: &mut TuiState, key: &KeyEvent) -> Option<KeyAction> {
    if !matches!(state.screen, Screen::ToolDetail { .. }) {
        return None;
    }
    if key.modifiers.contains(KeyModifiers::CONTROL) {
        return Some(match key.code {
            KeyCode::Char('q') => {
                state.screen = Screen::Chat;
                KeyAction::None
            }
            _ => KeyAction::None,
        });
    }
    if matches!(key.code, KeyCode::Esc | KeyCode::Char('q')) {
        state.screen = Screen::Chat;
        return Some(KeyAction::None);
    }
    let viewport = state.viewport;
    let max = {
        let body = match &state.screen {
            Screen::ToolDetail { body, .. } => body,
            _ => return None,
        };
        scroll_max(body, viewport)
    };
    let scroll = match &mut state.screen {
        Screen::ToolDetail { scroll, .. } => scroll,
        _ => return None,
    };
    Some(match key.code {
        KeyCode::Char('j') | KeyCode::Down => {
            scroll.toward_bottom(3, max);
            KeyAction::None
        }
        KeyCode::Char('k') | KeyCode::Up => {
            scroll.toward_top(3);
            KeyAction::None
        }
        KeyCode::PageDown => {
            scroll.toward_bottom(8, max);
            KeyAction::None
        }
        KeyCode::PageUp => {
            scroll.toward_top(8);
            KeyAction::None
        }
        KeyCode::Home => {
            scroll.home();
            KeyAction::None
        }
        KeyCode::End => {
            scroll.end(max);
            KeyAction::None
        }
        _ => KeyAction::None,
    })
}

pub fn draw(frame: &mut Frame, state: &TuiState, area: Rect) {
    let (body, scroll) = match &state.screen {
        Screen::ToolDetail { body, scroll } => (body, scroll),
        _ => return,
    };
    let lines: Vec<&str> = body.lines().collect();
    let visible = area.height as usize - 4;
    let total = lines.len();
    let max = total.saturating_sub(visible);
    let start = if scroll.following() {
        max
    } else {
        scroll.offset().min(max)
    };
    let mut body_lines: Vec<Line> = Vec::new();
    body_lines.extend(lines[start..].iter().take(visible).map(|line| {
        Line::from(Span::styled(
            *line,
            Style::default().fg(Color::Rgb(0x80, 0x80, 0x80)),
        ))
    }));
    if total == 0 {
        body_lines.push(Line::from(Span::styled(
            " (no output) ",
            Style::default().fg(Color::Rgb(102, 102, 102)),
        )));
    }
    let hint = Line::from(Span::styled(
        " jk scroll · PgUp/PgDn page · q close",
        Style::default().fg(Color::Rgb(102, 102, 102)),
    ));
    crate::tui::render_panel(
        frame,
        area,
        " full output ".to_string(),
        body_lines
            .into_iter()
            .chain(std::iter::once(hint))
            .collect::<Vec<Line>>(),
        true,
    );
}
