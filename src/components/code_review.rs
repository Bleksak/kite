use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};

use crate::screen::Screen;
use crate::session::ReviewComment;
use crate::tui::{
    word_left, word_right, KeyAction, KeyCode, KeyEvent, KeyModifiers, MouseEvent,
    MouseEventKind, TuiState,
};

pub struct State {
    pub start: usize,
    pub text: String,
    pub file_index: usize,
    pub cursor: usize,
    pub anchor: Option<usize>,
    pub comments: Vec<ReviewComment>,
    pub commenting: bool,
    pub comment_draft: String,
    pub comment_cursor: usize,
    pub comment_whole_file: bool,
}

impl Default for State {
    fn default() -> Self {
        Self {
            start: 0,
            text: String::new(),
            file_index: 0,
            cursor: 0,
            anchor: None,
            comments: Vec::new(),
            commenting: false,
            comment_draft: String::new(),
            comment_cursor: 0,
            comment_whole_file: false,
        }
    }
}

pub fn open(state: &mut TuiState) -> State {
    let mut s = State::default();
    s.comments = std::mem::take(&mut state.session().review_comments);
    s.text = review_diff_text(state.session().review_baseline.as_deref());
    s
}

fn git_head_tree() -> String {
    let output = std::process::Command::new("git")
        .args(["rev-parse", "--verify", "HEAD^{tree}"])
        .output();
    match output {
        Ok(output) if output.status.success() => {
            String::from_utf8_lossy(&output.stdout).trim().to_string()
        }
        _ => {
            let empty = std::process::Command::new("git")
                .arg("mktree")
                .stdin(std::process::Stdio::null())
                .output();
            match empty {
                Ok(output) if output.status.success() => {
                    String::from_utf8_lossy(&output.stdout).trim().to_string()
                }
                _ => String::new(),
            }
        }
    }
}

fn review_diff_text(baseline: Option<&str>) -> String {
    let Some(current) = crate::plan_gate::git_snapshot_tree(None) else {
        return "not a git repository".to_string();
    };
    let baseline = baseline.map(|b| b.to_string()).unwrap_or_else(git_head_tree);
    if baseline.is_empty() {
        return "git has no baseline".to_string();
    }
    match std::process::Command::new("git")
        .args(["diff", &baseline, &current])
        .output()
    {
        Ok(output) if output.status.success() => {
            String::from_utf8_lossy(&output.stdout).into_owned()
        }
        Ok(output) => String::from_utf8_lossy(&output.stderr).into_owned(),
        Err(source) => format!("git failed: {source}"),
    }
}

fn diff_file_sections(text: &str) -> Vec<(String, String)> {
    let mut sections: Vec<(String, Vec<String>)> = Vec::new();
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("diff --git ") {
            let parts: Vec<&str> = rest.split(' ').collect();
            let path = parts
                .iter()
                .find(|p| p.starts_with("b/"))
                .or_else(|| parts.iter().find(|p| p.starts_with("a/")))
                .map(|p| p[2..].to_string())
                .unwrap_or_else(|| rest.to_string());
            sections.push((path, vec![line.to_string()]));
        } else if let Some(entry) = sections.last_mut() {
            entry.1.push(line.to_string());
        }
    }
    sections
        .into_iter()
        .map(|(path, lines)| (path, lines.join("\n")))
        .collect()
}

pub(crate) fn review_comment_range_label(comment: &ReviewComment) -> String {
    match &comment.range {
        Some(range) if range.end == range.start + 1 => range.start.to_string(),
        Some(range) => format!("{}-{}", range.start, range.end - 1),
        None => String::new(),
    }
}

fn review_comment_line(comment: &ReviewComment) -> Line<'static> {
    let label = review_comment_range_label(comment);
    let text = if label.is_empty() {
        format!("💬 {}", comment.text)
    } else {
        format!("💬 {label}: {}", comment.text)
    };
    Line::from(Span::styled(
        text,
        Style::default().fg(Color::Rgb(0xd2, 0xa2, 0x6c)),
    ))
}

fn comment_draft_line(cr: &State) -> Line<'static> {
    let cursor = cr.comment_cursor.min(cr.comment_draft.chars().count());
    let cursor_style = Style::default()
        .bg(Color::Rgb(0xd4, 0xd4, 0xd4))
        .fg(Color::Rgb(0x28, 0x28, 0x32));
    let mut spans = vec![Span::styled(
        " 💬 ".to_string(),
        Style::default().fg(Color::Rgb(0xd2, 0xa2, 0x6c)),
    )];
    let chars: Vec<char> = cr.comment_draft.chars().collect();
    for (i, c) in chars.iter().enumerate() {
        if i == cursor {
            spans.push(Span::styled(c.to_string(), cursor_style));
        } else {
            spans.push(Span::raw(c.to_string()));
        }
    }
    if cursor == chars.len() {
        spans.push(Span::styled(" ".to_string(), cursor_style));
    }
    Line::from(spans)
}

fn draft_after_index(cr: &State) -> Option<usize> {
    if !cr.commenting {
        return None;
    }
    Some(if cr.comment_whole_file {
        0
    } else {
        cr.anchor
            .map(|anchor| anchor.max(cr.cursor))
            .unwrap_or(cr.cursor)
    })
}

fn draft_mixed_index(cr: &State) -> Option<usize> {
    let sections = diff_file_sections(&cr.text);
    let (path, section) = sections.get(cr.file_index)?;
    let numbered = crate::transcript::diff_lines_numbered(section);
    let after = draft_after_index(cr)?;
    if after >= numbered.len() {
        return None;
    }
    let comments: Vec<&ReviewComment> = cr
        .comments
        .iter()
        .filter(|c| c.file == *path)
        .collect();
    let mut placed = vec![false; comments.len()];
    let mut index = 0;
    for (i, d) in numbered.iter().enumerate() {
        index += 1;
        if i == 0 {
            for (ci, comment) in comments.iter().enumerate() {
                if !placed[ci] && comment.range.is_none() {
                    index += 1;
                    placed[ci] = true;
                }
            }
        }
        if let Some(number) = d.new.or(d.old) {
            for (ci, comment) in comments.iter().enumerate() {
                if !placed[ci]
                    && comment
                        .range
                        .as_ref()
                        .is_some_and(|r| r.end - 1 == number)
                {
                    index += 1;
                    placed[ci] = true;
                }
            }
        }
        if i == after {
            return Some(index);
        }
    }
    None
}

pub(crate) fn review_file_view(cr: &State) -> (Vec<Line<'static>>, usize) {
    let sections = diff_file_sections(&cr.text);
    let Some((path, section)) = sections.get(cr.file_index) else {
        return (Vec::new(), 0);
    };
    let numbered = crate::transcript::diff_lines_numbered(section);
    let diff_count = numbered.len();
    let sel_range = cr.anchor.map(|anchor| {
        (anchor.min(cr.cursor), anchor.max(cr.cursor))
    });
    let comments: Vec<&ReviewComment> = cr
        .comments
        .iter()
        .filter(|c| &c.file == path)
        .collect();
    let mut placed = vec![false; comments.len()];
    let draft_after = draft_after_index(cr);
    let mut lines = Vec::new();
    for (index, d) in numbered.iter().enumerate() {
        let selected = sel_range
            .is_some_and(|(lo, hi)| index >= lo && index <= hi);
        let is_cursor = index == cr.cursor;
        let mut line = d.line.clone();
        if selected || is_cursor {
            let bg = if is_cursor {
                Color::Rgb(0x4a, 0x4a, 0x5e)
            } else {
                Color::Rgb(0x38, 0x38, 0x48)
            };
            for span in line.spans.iter_mut() {
                span.style = span.style.patch(Style::default().bg(bg));
            }
        }
        lines.push(line);
        if index == 0 {
            for (i, comment) in comments.iter().enumerate() {
                if !placed[i] && comment.range.is_none() {
                    lines.push(review_comment_line(comment));
                    placed[i] = true;
                }
            }
        }
        if let Some(number) = d.new.or(d.old) {
            for (i, comment) in comments.iter().enumerate() {
                if !placed[i]
                    && comment
                        .range
                        .as_ref()
                        .is_some_and(|r| r.end - 1 == number)
                {
                    lines.push(review_comment_line(comment));
                    placed[i] = true;
                }
            }
        }
        if draft_after == Some(index) {
            lines.push(comment_draft_line(cr));
        }
    }
    for (i, comment) in comments.iter().enumerate() {
        if !placed[i] {
            lines.push(review_comment_line(comment));
        }
    }
    (lines, diff_count)
}

fn review_comment_range(cr: &State) -> (usize, usize) {
    match cr.anchor {
        Some(anchor) => (anchor.min(cr.cursor), anchor.max(cr.cursor)),
        None => (cr.cursor, cr.cursor),
    }
}

fn commit_review_comment(cr: &mut State) {
    let text = cr.comment_draft.trim().to_string();
    let whole_file = cr.comment_whole_file;
    cr.commenting = false;
    cr.comment_draft.clear();
    cr.comment_cursor = 0;
    cr.comment_whole_file = false;
    let sections = diff_file_sections(&cr.text);
    let Some((path, section)) = sections.get(cr.file_index) else {
        return;
    };
    let text = text.trim();
    if text.is_empty() {
        return;
    }
    if whole_file {
        if let Some(existing) = cr
            .comments
            .iter_mut()
            .find(|c| c.file == *path && c.range.is_none())
        {
            existing.text = text.to_string();
        } else {
            cr.comments.push(ReviewComment {
                file: path.clone(),
                range: None,
                text: text.to_string(),
            });
        }
        return;
    }
    let numbered = crate::transcript::diff_lines_numbered(section);
    let (lo, hi) = review_comment_range(cr);
    cr.anchor = None;
    let numbers: Vec<usize> = (lo..=hi)
        .filter_map(|i| numbered.get(i).and_then(|d| d.new.or(d.old)))
        .collect();
    if numbers.is_empty() {
        return;
    }
    let start = numbers.first().copied().unwrap();
    let end = numbers.last().copied().unwrap();
    if let Some(existing) = cr
        .comments
        .iter_mut()
        .find(|c| {
            c.file == *path
                && c.range
                    .as_ref()
                    .is_some_and(|r| r.start <= hi && lo < r.end)
        })
    {
        existing.text = text.to_string();
        existing.range = Some(start..end + 1);
    } else {
        cr.comments.push(ReviewComment {
            file: path.clone(),
            range: Some(start..end + 1),
            text: text.to_string(),
        });
    }
}

fn diff_scroll_max(cr: &State, viewport: usize) -> usize {
    let (lines, _) = review_file_view(cr);
    let total = if lines.is_empty() { 1 } else { lines.len() };
    let visible = viewport.saturating_sub(4);
    total.saturating_sub(visible)
}

pub fn mouse(state: &mut TuiState, mouse: &MouseEvent) -> Option<KeyAction> {
    let viewport = state.viewport;
    let cr = match &mut state.screen {
        Screen::CodeReview(cr) => cr,
        _ => return None,
    };
    let max = diff_scroll_max(cr, viewport);
    Some(match mouse.kind {
        MouseEventKind::ScrollUp => {
            cr.start = cr.start.saturating_sub(3);
            KeyAction::None
        }
        MouseEventKind::ScrollDown => {
            cr.start = (cr.start + 3).min(max);
            KeyAction::None
        }
    })
}

pub fn handle_key(state: &mut TuiState, key: &KeyEvent) -> Option<KeyAction> {
    let viewport = state.viewport;
    let cr = match &mut state.screen {
        Screen::CodeReview(cr) => cr,
        _ => return None,
    };
    let scroll_max = diff_scroll_max(cr, viewport);
    let visible = viewport.saturating_sub(4);
    let file_count = diff_file_sections(&cr.text).len();
    if cr.commenting {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        return Some(match key.code {
            KeyCode::Enter => {
                commit_review_comment(cr);
                KeyAction::None
            }
            KeyCode::Esc => {
                cr.commenting = false;
                cr.comment_draft.clear();
                cr.comment_cursor = 0;
                KeyAction::None
            }
            KeyCode::Backspace => {
                if cr.comment_cursor > 0 {
                    cr.comment_cursor -= 1;
                    let mut chars: Vec<char> = cr.comment_draft.chars().collect();
                    chars.remove(cr.comment_cursor);
                    cr.comment_draft = chars.into_iter().collect();
                }
                KeyAction::None
            }
            KeyCode::Left => {
                if ctrl {
                    cr.comment_cursor = word_left(&cr.comment_draft, cr.comment_cursor);
                } else {
                    cr.comment_cursor = cr.comment_cursor.saturating_sub(1);
                }
                KeyAction::None
            }
            KeyCode::Right => {
                if ctrl {
                    cr.comment_cursor = word_right(&cr.comment_draft, cr.comment_cursor);
                } else {
                    cr.comment_cursor =
                        (cr.comment_cursor + 1).min(cr.comment_draft.chars().count());
                }
                KeyAction::None
            }
            KeyCode::Char(c) => {
                let mut chars: Vec<char> = cr.comment_draft.chars().collect();
                chars.insert(cr.comment_cursor.min(chars.len()), c);
                cr.comment_draft = chars.into_iter().collect();
                cr.comment_cursor += 1;
                KeyAction::None
            }
            _ => KeyAction::None,
        });
    }
    Some(match key.code {
        KeyCode::Char('g') | KeyCode::Char('q') | KeyCode::Esc => {
            let comments = std::mem::take(&mut cr.comments);
            state.screen = Screen::Chat;
            KeyAction::ReviewClosed(comments)
        }
        KeyCode::Left | KeyCode::Char('h') => {
            cr.file_index = cr.file_index.saturating_sub(1);
            cr.cursor = 0;
            cr.anchor = None;
            cr.start = 0;
            KeyAction::None
        }
        KeyCode::Right | KeyCode::Char('l') => {
            if cr.file_index + 1 < file_count {
                cr.file_index += 1;
                cr.cursor = 0;
                cr.anchor = None;
                cr.start = 0;
            }
            KeyAction::None
        }
        KeyCode::Char('j') | KeyCode::Down => {
            let (_, diff_count) = review_file_view(cr);
            cr.cursor = (cr.cursor + 1).min(diff_count.saturating_sub(1));
            if cr.cursor < cr.start {
                cr.start = cr.cursor;
            } else if cr.cursor >= cr.start + visible {
                cr.start = cr.cursor - visible + 1;
            }
            KeyAction::None
        }
        KeyCode::Char('k') | KeyCode::Up => {
            cr.cursor = cr.cursor.saturating_sub(1);
            if cr.cursor < cr.start {
                cr.start = cr.cursor;
            }
            KeyAction::None
        }
        KeyCode::PageDown => {
            cr.start = (cr.start + 8).min(scroll_max);
            KeyAction::None
        }
        KeyCode::PageUp => {
            cr.start = cr.start.saturating_sub(8);
            KeyAction::None
        }
        KeyCode::Home => {
            cr.start = 0;
            KeyAction::None
        }
        KeyCode::End => {
            cr.start = scroll_max;
            KeyAction::None
        }
        KeyCode::Char('v') => {
            cr.anchor = if cr.anchor.is_some() {
                None
            } else {
                Some(cr.cursor)
            };
            KeyAction::None
        }
        KeyCode::Char('c') | KeyCode::Char('C') => {
            let whole_file = key.code == KeyCode::Char('C')
                || key.modifiers.contains(KeyModifiers::SHIFT);
            let (lo, hi) = review_comment_range(cr);
            let sections = diff_file_sections(&cr.text);
            let mut draft = String::new();
            if let Some((path, _)) = sections.get(cr.file_index) {
                let existing = if whole_file {
                    cr.comments
                        .iter()
                        .find(|c| c.file == *path && c.range.is_none())
                } else {
                    cr.comments.iter().find(|c| {
                        c.file == *path
                            && c.range
                                .as_ref()
                                .is_some_and(|r| r.start <= hi && lo < r.end)
                    })
                };
                if let Some(existing) = existing {
                    draft = existing.text.clone();
                }
            }
            cr.comment_draft = draft;
            cr.comment_cursor = cr.comment_draft.chars().count();
            cr.comment_whole_file = whole_file;
            cr.commenting = true;
            if whole_file {
                cr.start = 0;
            } else if let Some(draft_index) = draft_mixed_index(cr)
                && draft_index >= cr.start + visible
            {
                cr.start = draft_index - visible + 1;
            }
            KeyAction::None
        }
        KeyCode::Char('r') => {
            let path = diff_file_sections(&cr.text)
                .get(cr.file_index)
                .map(|(path, _)| path.clone());
            if let Some(path) = path {
                let session = state.session();
                if let Some(pos) = session.review_reviewed.iter().position(|p| p == &path) {
                    session.review_reviewed.remove(pos);
                } else {
                    session.review_reviewed.push(path);
                }
            }
            KeyAction::None
        }
        KeyCode::Char('x') => {
            let sections = diff_file_sections(&cr.text);
            if let Some((path, section)) = sections.get(cr.file_index) {
                let numbered = crate::transcript::diff_lines_numbered(section);
                let cursor_number = numbered
                    .get(cr.cursor)
                    .and_then(|d| d.new.or(d.old));
                let on_title = cr.cursor == 0;
                let pos = match cursor_number {
                    Some(number) => cr.comments.iter().position(|c| {
                        c.file == *path
                            && c.range
                                .as_ref()
                                .is_some_and(|r| r.contains(&number))
                    }),
                    None if on_title => cr.comments.iter().position(
                        |c| c.file == *path && c.range.is_none(),
                    ),
                    None => None,
                };
                if let Some(pos) = pos {
                    cr.comments.remove(pos);
                }
            }
            KeyAction::None
        }
        _ => KeyAction::None,
    })
}

pub fn draw(frame: &mut Frame, state: &TuiState, area: Rect) {
    let cr = match &state.screen {
        Screen::CodeReview(cr) => cr,
        _ => return,
    };
    let scroll_max = diff_scroll_max(cr, state.viewport);
    let (mut lines, _) = review_file_view(cr);
    if lines.is_empty() {
        lines = vec![Line::from(Span::styled(
            " no changes",
            Style::default().fg(Color::Rgb(102, 102, 102)),
        ))];
    }
    let title = {
        let sections = diff_file_sections(&cr.text);
        if sections.is_empty() {
            " review ".to_string()
        } else {
            let (path, _) = &sections[cr.file_index];
            let reviewed = state
                .sessions
                .get(state.active)
                .map(|s| s.review_reviewed.iter().any(|p| p == path))
                .unwrap_or(false);
            format!(
                " review {} / {} {}{}",
                cr.file_index + 1,
                sections.len(),
                path,
                if reviewed { " · ✓" } else { "" }
            )
        }
    };
    let hint = if cr.commenting {
        Line::from(Span::styled(
            " enter submit · esc cancel · backspace delete",
            Style::default().fg(Color::Rgb(102, 102, 102)),
        ))
    } else {
        Line::from(Span::styled(
            " ←→ file · jk cursor · v select · c comment · shift+c file comment · x remove comment · r reviewed · ctrl+g close",
            Style::default().fg(Color::Rgb(102, 102, 102)),
        ))
    };
    lines.push(hint);
    let visible = state.viewport.saturating_sub(4);
    let start = cr.start.min(scroll_max);
    crate::tui::render_panel(
        frame,
        area,
        title,
        lines[start..].iter().take(visible).cloned().collect::<Vec<Line>>(),
        true,
    );
}
