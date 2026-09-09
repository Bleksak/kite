use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};

use crate::event::keyboard::{KeyCode, KeyEvent, MouseEvent, MouseEventKind};
use crate::screen::Screen;
use crate::session::Scroller;
use crate::tui::{KeyAction, TuiState};

pub struct State {
    pub view: usize,
    pub scroll: Scroller,
}

impl Default for State {
    fn default() -> Self {
        Self {
            view: 0,
            scroll: Scroller::default(),
        }
    }
}

fn scroll_max(stages: &[crate::tool::PlanStage], view: usize, viewport: usize) -> usize {
    let Some(stage) = stages.get(view) else {
        return 0;
    };
    let total =
        1 + 1 + stage.description.lines().count().max(1) + 1;
    let visible = viewport.saturating_sub(4);
    total.saturating_sub(visible)
}

pub fn mouse(state: &mut TuiState, mouse: &MouseEvent) -> Option<KeyAction> {
    let viewport = state.viewport;
    let (stages, _) = state
        .sessions
        .get(state.active)
        .and_then(|s| s.plan.clone())
        .unwrap_or((Vec::new(), 0));
    let plan = match &mut state.screen {
        Screen::PlanDetail(plan) => plan,
        _ => return None,
    };
    let max = scroll_max(&stages, plan.view, viewport);
    Some(match mouse.kind {
        MouseEventKind::ScrollUp => {
            plan.scroll.toward_top(3);
            KeyAction::None
        }
        MouseEventKind::ScrollDown => {
            plan.scroll.toward_bottom(3, max);
            KeyAction::None
        }
        _ => KeyAction::None,
    })
}

pub fn handle_key(state: &mut TuiState, key: &KeyEvent) -> Option<KeyAction> {
    let viewport = state.viewport;
    let (stages, _) = state
        .sessions
        .get(state.active)
        .and_then(|s| s.plan.clone())
        .unwrap_or((Vec::new(), 0));
    let len = stages.len();
    let plan = match &mut state.screen {
        Screen::PlanDetail(plan) => plan,
        _ => return None,
    };
    let scroll_max = scroll_max(&stages, plan.view, viewport);
    Some(match key.code {
        KeyCode::Char('p') | KeyCode::Char('q') | KeyCode::Esc => {
            state.screen = Screen::Chat;
            KeyAction::None
        }
        KeyCode::Left | KeyCode::Char('h') => {
            if plan.view > 0 {
                plan.view -= 1;
                plan.scroll = Scroller::at_tail();
            }
            KeyAction::None
        }
        KeyCode::Right | KeyCode::Char('l') => {
            if plan.view + 1 < len {
                plan.view += 1;
                plan.scroll = Scroller::at_tail();
            }
            KeyAction::None
        }
        KeyCode::Char('j') | KeyCode::Down => {
            plan.scroll.toward_bottom(3, scroll_max);
            KeyAction::None
        }
        KeyCode::Char('k') | KeyCode::Up => {
            plan.scroll.toward_top(3);
            KeyAction::None
        }
        KeyCode::PageDown => {
            plan.scroll.toward_bottom(8, scroll_max);
            KeyAction::None
        }
        KeyCode::PageUp => {
            plan.scroll.toward_top(8);
            KeyAction::None
        }
        KeyCode::Home => {
            plan.scroll.home();
            KeyAction::None
        }
        KeyCode::End => {
            plan.scroll.end(scroll_max);
            KeyAction::None
        }
        _ => KeyAction::None,
    })
}

pub fn draw(frame: &mut Frame, state: &TuiState, area: Rect) {
    let plan = match &state.screen {
        Screen::PlanDetail(plan) => plan,
        _ => return,
    };
    let (stages, _) = state
        .sessions
        .get(state.active)
        .and_then(|s| s.plan.clone())
        .unwrap_or((Vec::new(), 0));
    let scroll_max = scroll_max(&stages, plan.view, state.viewport);
    let mut lines: Vec<Line> = Vec::new();
    let mut title_suffix = String::new();
    if stages.is_empty() {
        lines.push(Line::from(Span::styled(
            " no plan",
            Style::default().fg(Color::Rgb(102, 102, 102)),
        )));
    } else if let Some(stage) = stages.get(plan.view) {
        title_suffix = format!(" · Step {} of {}", plan.view + 1, stages.len());
        let mut md = String::new();
        md.push_str(&stage.title);
        md.push_str("\n\n");
        if stage.description.trim().is_empty() {
            md.push_str("_no description_");
        } else {
            md.push_str(&stage.description);
        }
        lines = crate::transcript::render_markdown_lines(&md);
    }
    let hint = Line::from(Span::styled(
        " ←→ step · jk/ PgUp PgDn scroll · ctrl+p close",
        Style::default().fg(Color::Rgb(102, 102, 102)),
    ));
    let all: Vec<Line> = lines.into_iter().chain(std::iter::once(hint)).collect();
    let visible = state.viewport.saturating_sub(4);
    let start = if plan.scroll.following() {
        scroll_max
    } else {
        plan.scroll.offset().min(scroll_max)
    };
    crate::tui::render_panel(
        frame,
        area,
        format!(" plan {title_suffix} "),
        all[start..].iter().take(visible).cloned().collect::<Vec<Line>>(),
        true,
    );
}
