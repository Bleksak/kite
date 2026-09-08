use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};

use crate::session::Scroller;
use crate::tui::{
    KeyAction, KeyCode, KeyEvent, MouseEvent, MouseEventKind, TuiState,
};

pub struct State {
    pub open: bool,
    pub view: usize,
    pub scroll: Scroller,
}

impl Default for State {
    fn default() -> Self {
        Self {
            open: false,
            view: 0,
            scroll: Scroller::default(),
        }
    }
}

fn scroll_max(state: &TuiState) -> usize {
    let Some((stages, _)) = state
        .sessions
        .get(state.active)
        .and_then(|s| s.plan.clone())
    else {
        return 0;
    };
    let Some(stage) = stages.get(state.plan_detail.view) else {
        return 0;
    };
    let total =
        1 + 1 + if stage.tasks.is_empty() { 1 } else { stage.tasks.len() } + 1;
    let visible = state.viewport.saturating_sub(4);
    total.saturating_sub(visible)
}

pub fn mouse(state: &mut TuiState, mouse: &MouseEvent) -> Option<KeyAction> {
    if !state.plan_detail.open {
        return None;
    }
    let max = scroll_max(state);
    Some(match mouse.kind {
        MouseEventKind::ScrollUp => {
            state.plan_detail.scroll.toward_top(3);
            KeyAction::None
        }
        MouseEventKind::ScrollDown => {
            state.plan_detail.scroll.toward_bottom(3, max);
            KeyAction::None
        }
    })
}

pub fn handle_key(state: &mut TuiState, key: &KeyEvent) -> Option<KeyAction> {
    if !state.plan_detail.open {
        return None;
    }
    let len = state
        .session()
        .plan
        .as_ref()
        .map(|(stages, _)| stages.len())
        .unwrap_or(0);
    let scroll_max = scroll_max(state);
    Some(match key.code {
        KeyCode::Char('p') | KeyCode::Char('q') | KeyCode::Esc => {
            state.plan_detail.open = false;
            KeyAction::None
        }
        KeyCode::Left | KeyCode::Char('h') => {
            if state.plan_detail.view > 0 {
                state.plan_detail.view -= 1;
                state.plan_detail.scroll = Scroller::at_tail();
            }
            KeyAction::None
        }
        KeyCode::Right | KeyCode::Char('l') => {
            if state.plan_detail.view + 1 < len {
                state.plan_detail.view += 1;
                state.plan_detail.scroll = Scroller::at_tail();
            }
            KeyAction::None
        }
        KeyCode::Char('j') | KeyCode::Down => {
            state.plan_detail.scroll.toward_bottom(3, scroll_max);
            KeyAction::None
        }
        KeyCode::Char('k') | KeyCode::Up => {
            state.plan_detail.scroll.toward_top(3);
            KeyAction::None
        }
        KeyCode::PageDown => {
            state.plan_detail.scroll.toward_bottom(8, scroll_max);
            KeyAction::None
        }
        KeyCode::PageUp => {
            state.plan_detail.scroll.toward_top(8);
            KeyAction::None
        }
        KeyCode::Home => {
            state.plan_detail.scroll.home();
            KeyAction::None
        }
        KeyCode::End => {
            state.plan_detail.scroll.end(scroll_max);
            KeyAction::None
        }
        _ => KeyAction::None,
    })
}

pub fn draw(frame: &mut Frame, state: &TuiState, area: Rect) {
    let (stages, _) = state
        .sessions
        .get(state.active)
        .and_then(|s| s.plan.clone())
        .unwrap_or((Vec::new(), 0));
    let scroll_max = scroll_max(state);
    let mut lines: Vec<Line> = Vec::new();
    let mut title_suffix = String::new();
    if stages.is_empty() {
        lines.push(Line::from(Span::styled(
            " no plan",
            Style::default().fg(Color::Rgb(102, 102, 102)),
        )));
    } else if let Some(stage) = stages.get(state.plan_detail.view) {
        title_suffix = format!(" · Step {} of {}", state.plan_detail.view + 1, stages.len());
        let mut md = String::new();
        md.push_str(&stage.title);
        md.push_str("\n\n");
        if stage.tasks.is_empty() {
            md.push_str("_no tasks_");
        } else {
            for (i, task) in stage.tasks.iter().enumerate() {
                md.push_str(&format!("{}. {}\n", i + 1, task));
            }
            if md.ends_with('\n') {
                md.pop();
            }
        }
        lines = crate::transcript::render_markdown_lines(&md);
    }
    let hint = Line::from(Span::styled(
        " ←→ step · jk/ PgUp PgDn scroll · ctrl+p close",
        Style::default().fg(Color::Rgb(102, 102, 102)),
    ));
    let all: Vec<Line> = lines.into_iter().chain(std::iter::once(hint)).collect();
    let visible = state.viewport.saturating_sub(4);
    let start = if state.plan_detail.scroll.following() {
        scroll_max
    } else {
        state.plan_detail.scroll.offset().min(scroll_max)
    };
    crate::tui::render_panel(
        frame,
        area,
        format!(" plan {title_suffix} "),
        all[start..].iter().take(visible).cloned().collect::<Vec<Line>>(),
        true,
    );
}
