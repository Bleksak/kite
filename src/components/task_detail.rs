use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};

use crate::tui::{
    KeyAction, KeyCode, KeyEvent, KeyModifiers, MouseEvent, MouseEventKind, TuiState,
};

fn scroll_max(state: &TuiState) -> usize {
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

pub fn mouse(state: &mut TuiState, mouse: &MouseEvent) -> Option<KeyAction> {
    if state.task_output_id.is_none() {
        return None;
    }
    let max = scroll_max(state);
    Some(match mouse.kind {
        MouseEventKind::ScrollUp => {
            state.task_output_scroll.toward_top(3);
            KeyAction::None
        }
        MouseEventKind::ScrollDown => {
            state.task_output_scroll.toward_bottom(3, max);
            KeyAction::None
        }
    })
}

pub fn handle_key(state: &mut TuiState, key: &KeyEvent) -> Option<KeyAction> {
    if state.task_output_id.is_none() {
        return None;
    }
    if key.modifiers.contains(KeyModifiers::CONTROL) {
        return Some(match key.code {
            KeyCode::Char('q') => {
                state.task_list.open = false;
                state.task_output_id = None;
                KeyAction::None
            }
            _ => KeyAction::None,
        });
    }
    let scroll_max = scroll_max(state);
    Some(match key.code {
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
    })
}

pub fn draw(frame: &mut Frame, state: &TuiState, area: Rect) {
    let Some(id) = &state.task_output_id else {
        return;
    };
    let tasks = crate::bg::REGISTRY.list();
    let Some(task) = tasks.iter().find(|t| t.id == *id) else {
        return;
    };
    let text = std::fs::read_to_string(&task.output_path).unwrap_or_default();
    let lines: Vec<&str> = text.lines().collect();
    let visible = area.height as usize - 4;
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
