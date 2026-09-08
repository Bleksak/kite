use ratatui::buffer::{Buffer, Cell};
use ratatui::layout::Rect;
use ratatui::text::Line;
use ratatui::widgets::{Paragraph, Widget, Wrap};

use crate::context::Context;
use crate::mode::Mode;
use crate::tool::Question;
use crate::transcript::TuiRenderer;

#[derive(Debug, Clone)]
pub struct QuestionState {
    pub questions: Vec<Question>,
    pub step: usize,
    pub answers: Vec<String>,
    pub cursor: usize,
    pub selected: Vec<bool>,
    pub draft: String,
    pub draft_cursor: usize,
    pub reply: Option<tokio::sync::watch::Sender<Option<Vec<String>>>>,
}

impl QuestionState {
    pub fn new(questions: Vec<Question>) -> QuestionState {
        let mut state = QuestionState {
            questions,
            step: 0,
            answers: Vec::new(),
            cursor: 0,
            selected: Vec::new(),
            draft: String::new(),
            draft_cursor: 0,
            reply: None,
        };
        state.reset_step();
        state
    }

    pub fn reset_step(&mut self) {
        let options = &self.questions[self.step].options;
        self.cursor = 0;
        self.selected = options.iter().map(|_| false).collect();
        self.draft.clear();
        self.draft_cursor = 0;
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct ReviewComment {
    pub file: String,
    pub range: Option<std::ops::Range<usize>>,
    pub text: String,
}

pub struct Session {
    pub id: u64,
    pub renderer: TuiRenderer,
    pub input: String,
    pub running: bool,
    pub error: Option<String>,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub scroller: Scroller,
    pub label: String,
    pub context: Option<Context>,
    pub review_baseline: Option<String>,
    pub review_comments: Vec<ReviewComment>,
    pub review_reviewed: Vec<String>,
    pub gate: bool,
    pub gate_message: String,
    pub stage: Option<Mode>,
    pub input_cursor: usize,
    pub input_scroll: usize,
    pub history: Vec<String>,
    pub history_index: Option<usize>,
    pub question: Option<QuestionState>,
    pub suggest: Option<Vec<crate::auto_complete::Suggestion>>,
    pub suggest_sel: usize,
    pub plan: Option<(Vec<crate::tool::PlanStage>, usize)>,
    tail_cache: (usize, usize, usize, usize),
}

impl Session {
    pub fn new(id: u64) -> Session {
        Session {
            id,
            renderer: TuiRenderer::new(),
            input: String::new(),
            running: false,
            error: None,
            prompt_tokens: 0,
            completion_tokens: 0,
            scroller: Scroller::at_tail(),
            label: String::new(),
            context: None,
            review_baseline: None,
            review_comments: Vec::new(),
            review_reviewed: Vec::new(),
            gate: false,
            gate_message: String::new(),
            stage: None,
            input_cursor: 0,
            input_scroll: 0,
            history: Vec::new(),
            history_index: None,
            question: None,
            suggest: None,
            suggest_sel: 0,
            plan: None,
            tail_cache: (0, 0, 0, 0),
        }
    }

    pub fn max_scroll(&mut self, pane_width: usize, viewport: usize) -> usize {
        let len = self.renderer.scrollback().len();
        let (cl, cw, cv, cs) = self.tail_cache;
        if cl == len && cw == pane_width && cv == viewport {
            return cs;
        }
        let result = tail_start(len, self.renderer.scrollback(), pane_width as u16, viewport);
        self.tail_cache = (len, pane_width, viewport, result);
        result
    }
}

#[derive(Default)]
pub struct Cursor {
    pub pos: usize,
}

impl Cursor {
    pub fn down(&mut self, len: usize) {
        if len > 0 {
            self.pos = (self.pos + 1) % len;
        }
    }

    pub fn up(&mut self, len: usize) {
        if len > 0 {
            self.pos = if self.pos == 0 { len - 1 } else { self.pos - 1 };
        }
    }

    pub fn clamp(&mut self, len: usize) {
        self.pos = if len == 0 { 0 } else { self.pos.min(len - 1) };
    }

    pub fn set(&mut self, pos: usize) {
        self.pos = pos;
    }
}

#[derive(Default)]
pub struct Scroller {
    pub offset: usize,
    pub following: bool,
}

impl Scroller {
    pub fn at_tail() -> Self {
        Scroller {
            offset: 0,
            following: true,
        }
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn following(&self) -> bool {
        self.following
    }

    pub fn set_following(&mut self, following: bool) {
        self.following = following;
    }

    pub fn toward_top(&mut self, n: usize) {
        self.offset = self.offset.saturating_sub(n);
        self.following = false;
    }

    pub fn toward_bottom(&mut self, n: usize, max: usize) {
        self.offset = (self.offset + n).min(max);
        self.following = self.offset == max;
    }

    pub fn home(&mut self) {
        self.offset = 0;
        self.following = false;
    }

    pub fn end(&mut self, max: usize) {
        self.offset = max;
        self.following = true;
    }

    pub fn follow_tail(&mut self, max: usize) {
        if self.following {
            self.offset = max;
        }
    }
}

fn tail_start(len: usize, lines: &[Line<'static>], width: u16, viewport: usize) -> usize {
    let first = len.saturating_sub(viewport);
    if fits_viewport(&lines[first..], width, viewport) {
        return first;
    }
    let mut lo = first;
    let mut hi = len;
    while lo < hi {
        let mid = (lo + hi) / 2;
        if fits_viewport(&lines[mid..], width, viewport) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    lo
}

fn fits_viewport(lines: &[Line<'static>], width: u16, viewport: usize) -> bool {
    if lines.is_empty() {
        return true;
    }
    let area = Rect::new(0, 0, width, (viewport + 1) as u16);
    let mut buffer = Buffer::empty(area);
    Paragraph::new(lines)
        .wrap(Wrap { trim: false })
        .render(area, &mut buffer);
    (0..width).all(|x| {
        buffer
            .cell((x, viewport as u16))
            .is_none_or(|cell| *cell == Cell::default())
    })
}
