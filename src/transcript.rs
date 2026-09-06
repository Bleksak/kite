use std::collections::HashMap;
use std::mem;
use std::sync::LazyLock;

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use tui_markdown::{from_str_with_options, Options, StyleSheet};

use crate::context::Context;
use crate::message::Message;
use crate::stream::{AgentEvent, AnswerGate, ChunkTokens, ThinkingMode};
use crate::tool::{Tool, ToolOutput};

#[derive(Clone)]
struct TuiStyleSheet;



impl StyleSheet for TuiStyleSheet {

    fn code_block_fence(&self) -> &str {

        ""

    }



    fn heading(&self, level: u8) -> Style {

        let style = Style::default()

            .fg(Color::Rgb(0xf0, 0xc6, 0x74))

            .bold();

        if level == 1 {

            style.underlined()

        } else {

            style

        }

    }



    fn heading_marker(&self, _level: u8) -> &str {

        ""

    }



    fn link(&self) -> Style {

        Style::default()

            .fg(Color::Rgb(0x81, 0xa2, 0xbe))

            .underlined()

    }



    fn code(&self) -> Style {

        Style::default().fg(Color::Rgb(138, 190, 183))

    }



    fn blockquote(&self) -> Style {

        Style::default()

            .fg(Color::Rgb(128, 128, 128))

            .italic()

    }

}



static MD_OPTIONS: LazyLock<Options<TuiStyleSheet>> = LazyLock::new(|| Options::new(TuiStyleSheet));

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BlockKind {

    User,

    Thinking,

    Answer,

    ToolRunning,

    ToolDone,

}



pub struct TuiRenderer {

    gate: AnswerGate,

    scrollback: Vec<Line<'static>>,

    blocks: Vec<BlockKind>,

    thinking: String,

    answer: String,

    turn_answer: String,

    answer_start: Option<usize>,

    tool_blocks: std::collections::HashMap<String, (usize, usize)>,

}



impl TuiRenderer {

    pub fn new() -> TuiRenderer {

        TuiRenderer {

            gate: AnswerGate::new(),

            scrollback: Vec::new(),

            blocks: Vec::new(),

            thinking: String::new(),

            answer: String::new(),

            turn_answer: String::new(),

            answer_start: None,

            tool_blocks: std::collections::HashMap::new(),

        }

    }



    pub fn scrollback(&self) -> &[Line<'static>] {

        &self.scrollback

    }



    pub fn blocks(&self) -> &[BlockKind] {

        &self.blocks

    }



    fn push_line(&mut self, line: Line<'static>, block: BlockKind) {

        self.scrollback.push(line);

        self.blocks.push(block);

    }



    pub fn push_user(&mut self, text: &str) {

        self.close_thinking();

        self.push_line(padding_line(), BlockKind::User);

        if has_markdown(text) {

            self.push_markdown(text, BlockKind::User);

        } else {

            for line in text.trim_end_matches('\n').split('\n') {

                if line.is_empty() {

                    self.push_line(Line::default(), BlockKind::User);

                } else {

                    self.push_line(

                        Line::from(Span::styled(

                            strip_vs16(line),

                            Style::default()

                                .bold()

                                .fg(Color::Rgb(212, 212, 212)),

                        )),

                        BlockKind::User,

                    );

                }

            }

        }

        self.push_line(padding_line(), BlockKind::User);

    }



    pub fn replay_context(&mut self, context: &Context) {

        let mut tool_headers: HashMap<String, String> = HashMap::new();

        for message in &context.messages {

            match message {

                Message::System { .. } => {}

                Message::User { content } => self.push_user(content),

                Message::Assistant { content, tool_calls } => {

                    if let Some(text) = content {

                        self.on_event(AgentEvent::Tokens(ChunkTokens {

                            thinking: None,

                            text: Some(text.clone()),

                        }));

                        self.on_event(AgentEvent::CompletionStarted);

                    }

                    for call in tool_calls {

                        if let Ok(tool) = Tool::try_from(call.clone()) {

                            let header = tool.header();

                            tool_headers.insert(call.id.clone(), header.clone());

                            let body = match tool.output() {

                                ToolOutput::Before(body) => Some(body.to_string()),

                                _ => None,

                            };

                            self.on_event(AgentEvent::ToolStarted { header, body });

                        }

                    }

                }

                Message::Tool { tool_call_id, content } => {

                    if let Some(header) = tool_headers.get(tool_call_id).cloned() {

                        self.on_event(AgentEvent::ToolResult {

                            header,

                            body: content.clone(),

                        });

                    }

                }

            }

        }

    }



    pub fn on_event(&mut self, event: AgentEvent) {

        match event {

            AgentEvent::CompletionStarted => {

                self.close_thinking();

                self.end_answer_block();

                self.gate = AnswerGate::new();

            }

            AgentEvent::Tokens(chunk) => {

                let (mode, text) = self.gate.on_chunk(&chunk);

                if mode == ThinkingMode::Live

                    && let Some(thinking) = &chunk.thinking

                {

                    if self.thinking.is_empty() && !thinking.is_empty() {

                        self.push_line(padding_line(), BlockKind::Thinking);

                    }

                    self.thinking.push_str(thinking);

                }

                if let Some(text) = text {

                    self.close_thinking();

                    self.push_answer_text(&text);

                    self.turn_answer.push_str(&strip_vs16(&text));

                }

            }

            AgentEvent::ToolStarted { header, body } => {

                self.end_answer_block();

                self.close_thinking();

                self.tool_blocks.insert(header.clone(), (self.scrollback.len(), self.scrollback.len()));

                self.push_line(padding_line(), BlockKind::ToolRunning);

                self.push_line(

                    Line::from(Span::styled(format!("⚙ {header}"), Style::default().bold().fg(Color::Rgb(212, 212, 212)))),

                    BlockKind::ToolRunning,

                );

                if let Some(body) = body {

                    self.push_indented(&body, BlockKind::ToolRunning);

                }

                if let Some(entry) = self.tool_blocks.get_mut(&header) {

                    entry.1 = self.scrollback.len();

                }

            }

            AgentEvent::ToolResult { header, body } => {

                self.flush_answer();

                match self.tool_blocks.remove(&header) {

                    Some((start, end)) => {

                        for block in &mut self.blocks[start..end] {

                            if *block == BlockKind::ToolRunning {

                                *block = BlockKind::ToolDone;

                            }

                        }

                    }

                    None => {

                        self.push_line(

                            Line::from(Span::styled(format!("⚙ {header}"), Style::default().bold().fg(Color::Rgb(212, 212, 212)))),

                            BlockKind::ToolDone,

                        );

                    }

                }

                if header.starts_with("edit_file") {

                    self.push_diff(&body, BlockKind::ToolDone);

                } else {

                    self.push_indented(&body, BlockKind::ToolDone);

                }

                self.push_line(padding_line(), BlockKind::ToolDone);

            }

            AgentEvent::BgTaskDone { id, command, code } => {

                let status = match code {

                    Some(0) => "exit 0".to_string(),

                    Some(code) => format!("exit {code}"),

                    None => "killed".to_string(),

                };

                self.push_line(

                    Line::from(Span::styled(

                        format!("⏺ task {id} ({command}) finished — {status}"),

                        Style::default().fg(Color::Rgb(0x81, 0xa2, 0xbe)),

                    )),

                    BlockKind::ToolDone,

                );

            }

        }

    }



    pub fn finish(&mut self) {

        self.close_thinking();

        self.end_answer_block();

    }



    fn end_answer_block(&mut self) {

        self.flush_answer();

        let had_answer = self.answer_start.is_some();

        self.render_answer_markdown();

        if had_answer {

            self.push_line(padding_line(), BlockKind::Answer);

        }

    }



    fn close_thinking(&mut self) {

        if self.thinking.is_empty() {

            return;

        }

        self.flush_answer();

        let thinking = mem::take(&mut self.thinking);

        let trimmed = thinking.trim_start_matches('\n').trim_end_matches('\n');

        if has_markdown(trimmed) {

            self.push_markdown(trimmed, BlockKind::Thinking);

        } else {

            for line in trimmed.split('\n') {

                if line.is_empty() {

                    self.push_line(Line::default(), BlockKind::Thinking);

                } else {

                    self.push_line(

                        Line::from(Span::styled(strip_vs16(line), Style::default().fg(Color::Rgb(128, 128, 128)))),

                        BlockKind::Thinking,

                    );

                }

            }

        }

    }



    fn push_answer_text(&mut self, text: &str) {

        self.answer.push_str(text);

        while let Some(index) = self.answer.find('\n') {

            let line = strip_vs16(&self.answer[..index]);

            self.answer.drain(..=index);

            if line.trim().is_empty() && self.answer_start.is_none() {

                continue;

            }

            if self.answer_start.is_none() {

                self.push_line(padding_line(), BlockKind::Answer);

            }

            self.mark_answer_start();

            if line.is_empty() {

                self.push_line(Line::default(), BlockKind::Answer);

            } else {

                self.push_line(Line::from(line), BlockKind::Answer);

            }

        }

    }



    fn flush_answer(&mut self) {

        if self.answer.is_empty() {

            return;

        }

        let line = strip_vs16(&mem::take(&mut self.answer));

        if line.trim().is_empty() && self.answer_start.is_none() {

            return;

        }

        if self.answer_start.is_none() {

            self.push_line(padding_line(), BlockKind::Answer);

        }

        self.mark_answer_start();

        self.push_line(Line::from(line), BlockKind::Answer);

    }



    fn mark_answer_start(&mut self) {

        if self.answer_start.is_none() {

            self.answer_start = Some(self.scrollback.len());

        }

    }



    fn render_answer_markdown(&mut self) {

        if self.turn_answer.trim().is_empty() {

            return;

        }

        let Some(start) = self.answer_start else {

            return;

        };

        if !has_markdown(&self.turn_answer) {

            self.turn_answer.clear();

            self.answer_start = None;

            return;

        }

        let lines = style_markdown_lines(

            render_markdown_lines(&self.turn_answer),

            None,

            None,

        );

        let count = lines.len();

        self.scrollback.splice(start.., lines);

        self.blocks.splice(

            start..,

            std::iter::repeat_n(BlockKind::Answer, count).collect::<Vec<_>>(),

        );

        self.turn_answer.clear();

        self.answer_start = None;

    }



    fn push_markdown(&mut self, text: &str, block: BlockKind) {

        let (base, bold) = match block {

            BlockKind::User => (Some(Color::Rgb(212, 212, 212)), None),

            BlockKind::Thinking => (Some(Color::Rgb(128, 128, 128)), None),

            _ => (None, None),

        };

        let lines = style_markdown_lines(render_markdown_lines(text), base, bold);

        for line in lines {

            self.push_line(line, block);

        }

    }



    fn push_indented(&mut self, body: &str, block: BlockKind) {

        for line in body.trim_end_matches('\n').split('\n') {

            if line.is_empty() {

                self.push_line(Line::default(), block);

            } else {

                self.push_line(

                    Line::from(Span::styled(

                        format!("  {}", strip_vs16(line)),

                        Style::default().fg(Color::Rgb(128, 128, 128)),

                    )),

                    block,

                );

            }

        }

    }



    fn push_diff(&mut self, body: &str, block: BlockKind) {

        let gray = Color::Rgb(128, 128, 128);

        let mut old_line: Option<usize> = None;

        let mut new_line: Option<usize> = None;

        for raw in body.trim_end_matches('\n').split('\n') {

            let line = strip_vs16(raw);

            if let Some(hunk) = line.strip_prefix("@@ ") {

                let range = hunk.strip_suffix(" @@").unwrap_or(hunk);

                let mut parts = range.split(' ');

                old_line = parts.next().and_then(|p| hunk_start(p, '-'));

                new_line = parts.next().and_then(|p| hunk_start(p, '+'));

                continue;

            }

            if line.starts_with("--- ") || line.starts_with("+++ ") {

                continue;

            }

            let is_marker = line.starts_with('\\');

            let (style, bump_old, bump_new, sign, content) = match line.chars().next() {

                Some('+') => (Color::Rgb(0xb5, 0xbd, 0x68), false, true, "+", &line[1..]),

                Some('-') => (Color::Rgb(0xcc, 0x66, 0x66), true, false, "-", &line[1..]),

                _ if is_marker => (gray, false, false, "", line.as_str()),

                _ => (gray, true, true, "", line.as_str()),

            };

            let number = if is_marker {

                String::new()

            } else if bump_old {

                num(old_line)

            } else if bump_new {

                num(new_line)

            } else {

                String::new()

            };

            let field = if sign.is_empty() {

                format!("{:>5}", number)

            } else {

                format!("{sign}{:>4}", number)

            };

            let rendered = format!("{}  {}", field, content);

            self.push_line(

                Line::from(Span::styled(rendered, Style::default().fg(style))),

                block,

            );

            if bump_old {

                old_line = old_line.map(|n| n + 1);

            }

            if bump_new {

                new_line = new_line.map(|n| n + 1);

            }

        }

    }

}



fn padding_line() -> Line<'static> {

    Line::from(Span::styled(

        " ".to_string(),

        Style::default().fg(Color::Red),

    ))

}



fn style_markdown_lines(

    lines: Vec<Line<'static>>,

    base: Option<Color>,

    bold: Option<Color>,

) -> Vec<Line<'static>> {

    lines

        .into_iter()

        .map(|line| {

            let spans: Vec<Span> = line

                .spans

                .iter()

                .map(|span| {

                    let style = if span.style.fg.is_none() {

                        let color = if span.style.add_modifier.contains(Modifier::BOLD) {

                            bold.or(base)

                        } else {

                            base

                        };

                        match color {

                            Some(color) => span.style.patch(Style::default().fg(color)),

                            None => span.style,

                        }

                    } else {

                        span.style

                    };

                    Span {

                        content: std::borrow::Cow::Owned(span.content.to_string()),

                        style,

                    }

                })

                .collect();

            Line {

                style: line.style,

                alignment: line.alignment,

                spans,

            }

        })

        .collect()

}



fn render_markdown_lines(markdown: &str) -> Vec<Line<'static>> {

    let (prepared, restore) = prepare_markdown(markdown);

    let rendered = from_str_with_options(&prepared, &MD_OPTIONS);

    rendered

        .lines

        .iter()

        .map(|line| {

            let original = line.to_string();

            let spans: Vec<Span> = match restore.get(&original) {

                Some(rule) => {

                    let style = line

                        .spans

                        .first()

                        .map(|s| s.style.patch(line.style))

                        .unwrap_or(line.style);

                    vec![Span {

                        content: std::borrow::Cow::Owned(strip_vs16(rule)),

                        style,

                    }]

                }

                None => line

                    .spans

                    .iter()

                    .map(|span| Span {

                        content: std::borrow::Cow::Owned(strip_vs16(

                            span.content.as_ref(),

                        )),

                        style: span.style.patch(line.style),

                    })

                    .collect(),

            };

            Line {

                style: Style::default(),

                alignment: line.alignment,

                spans,

            }

        })

        .collect()

}



fn strip_vs16(text: &str) -> String {

    if text.contains('\u{FE0F}') {

        text.replace('\u{FE0F}', "")

    } else {

        text.to_string()

    }

}



fn num(n: Option<usize>) -> String {

    n.map(|n| n.to_string()).unwrap_or_default()

}



fn hunk_start(part: &str, sign: char) -> Option<usize> {

    part.strip_prefix(sign)?.split(',').next()?.parse().ok()

}



fn has_markdown(text: &str) -> bool {

    const MARKERS: [&str; 7] = ["**", "`", "# ", "```", "- ", "[", "> "];

    MARKERS.iter().any(|marker| text.contains(marker))

}



fn prepare_markdown(text: &str) -> (String, HashMap<String, String>) {

    let mut out: Vec<String> = Vec::new();

    let mut restore: HashMap<String, String> = HashMap::new();

    let mut in_fence = false;

    let mut fence_marker = "";

    for line in text.lines() {

        let trimmed = line.trim_end();

        let body = trimmed.trim_start();

        let indent = trimmed.len() - body.len();

        let fence = if body.starts_with("```") {

            "```"

        } else if body.starts_with("~~~") {

            "~~~"

        } else {

            ""

        };

        if !fence.is_empty() {

            if in_fence {

                if fence == fence_marker {

                    in_fence = false;

                }

            } else {

                in_fence = true;

                fence_marker = fence;

            }

            out.push(trimmed.to_string());

            continue;

        }

        if in_fence {

            out.push(trimmed.to_string());

            continue;

        }

        let compact = body.replace(' ', "");

        let is_rule = compact.len() >= 3

            && indent < 4

            && (compact.chars().all(|c| c == '*') || compact.chars().all(|c| c == '_'));

        if is_rule {

            if out.last().is_some_and(|last| !last.trim().is_empty()) {

                out.push(String::new());

            }

            let token = format!("kiterule{}", restore.len());

            restore.insert(token.clone(), body.to_string());

            out.push(token);

        } else {

            out.push(trimmed.to_string());

        }

    }

    let mut joined: Vec<String> = Vec::with_capacity(out.len() + 1);

    for (i, line) in out.iter().enumerate() {

        joined.push(line.clone());

        if line.starts_with("kiterule") && i + 1 < out.len() && !out[i + 1].trim().is_empty() {

            joined.push(String::new());

        }

    }

    (joined.join("\n"), restore)

}



#[cfg(test)]
mod test {
    use super::*;
    use ratatui::style::Color;

    fn lines(events: Vec<AgentEvent>) -> Vec<Line<'static>> {
        let mut renderer = TuiRenderer::new();
        for event in events {
            renderer.on_event(event);
        }
        renderer.finish();
        renderer.scrollback().to_vec()
    }

    fn thinking(text: &str) -> AgentEvent {
        AgentEvent::Tokens(ChunkTokens {
            thinking: Some(text.into()),
            text: None,
        })
    }

    fn text(content: &str) -> AgentEvent {
        AgentEvent::Tokens(ChunkTokens {
            thinking: None,
            text: Some(content.into()),
        })
    }

    fn describe(lines: &[Line<'static>]) -> Vec<(String, Modifier)> {
        lines
            .iter()
            .map(|line| {
                let span = line.spans.first().cloned().unwrap_or_default();
                (span.content.to_string(), span.style.add_modifier)
            })
            .collect()
    }

    fn colored(lines: &[Line<'static>]) -> Vec<(String, Option<Color>)> {
        lines
            .iter()
            .map(|line| {
                let span = line.spans.first().cloned().unwrap_or_default();
                (span.content.to_string(), span.style.fg)
            })
            .collect()
    }

    #[test]
    fn thinking_then_answer() {
        let described = describe(&lines(vec![thinking("Let me think. "), text("42")]));

        assert_eq!(
            described,
            vec![
                (" ".into(), Modifier::empty()),
                ("Let me think. ".into(), Modifier::empty()),
                (" ".into(), Modifier::empty()),
                ("42".into(), Modifier::empty()),
                (" ".into(), Modifier::empty()),
            ]
        );
    }

    #[test]
    fn straggler_thinking_is_hidden() {
        let described = describe(&lines(vec![
            thinking("reasoning "),
            text("answer"),
            thinking("straggler"),
        ]));

        assert_eq!(
            described,
            vec![
                (" ".into(), Modifier::empty()),
                ("reasoning ".into(), Modifier::empty()),
                (" ".into(), Modifier::empty()),
                ("answer".into(), Modifier::empty()),
                (" ".into(), Modifier::empty()),
            ]
        );
    }

    #[test]
    fn tool_event_closes_thinking() {
        let described = describe(&lines(vec![
            thinking("thinking"),
            AgentEvent::ToolStarted {
                header: "bash".into(),
                body: Some("ls".into()),
            },
        ]));

        assert_eq!(
            described,
            vec![
                (" ".into(), Modifier::empty()),
                ("thinking".into(), Modifier::empty()),
                (" ".into(), Modifier::empty()),
                ("⚙ bash".into(), Modifier::BOLD),
                ("  ls".into(), Modifier::empty()),
            ]
        );
    }

    #[test]
    fn edit_file_result_renders_colored_diff_with_line_numbers() {
        let diff = "--- a.txt\n+++ a.txt\n@@ -1,3 +1,3 @@\n one\n-two\n+TWO\n three\n";
        let described = colored(&lines(vec![
            AgentEvent::ToolStarted {
                header: "edit_file: a.txt".into(),
                body: None,
            },
            AgentEvent::ToolResult {
                header: "edit_file: a.txt".into(),
                body: diff.into(),
            },
        ]));

        assert!(described.iter().any(|(c, col)| c == "    1   one" && col == &Some(Color::Rgb(128, 128, 128))));
        assert!(described.iter().any(|(c, col)| c == "-   2  two" && col == &Some(Color::Rgb(0xcc, 0x66, 0x66))));
        assert!(described.iter().any(|(c, col)| c == "+   2  TWO" && col == &Some(Color::Rgb(0xb5, 0xbd, 0x68))));
        assert!(described.iter().any(|(c, _)| c == "    3   three"));
        assert!(!described.iter().any(|(c, _)| c.starts_with("--- a.txt")));
        assert!(!described.iter().any(|(c, _)| c.starts_with("@@")));
    }

    #[test]
    fn edit_file_diff_line_numbers_reset_per_hunk() {
        let diff = "--- a.txt\n+++ a.txt\n@@ -1,2 +1,2 @@\n-a\n+b\n@@ -10,2 +10,2 @@\n-x\n+y\n";
        let described = colored(&lines(vec![
            AgentEvent::ToolStarted {
                header: "edit_file: a.txt".into(),
                body: None,
            },
            AgentEvent::ToolResult {
                header: "edit_file: a.txt".into(),
                body: diff.into(),
            },
        ]));

        assert!(described.iter().any(|(c, _)| c == "-   1  a"));
        assert!(described.iter().any(|(c, _)| c == "+   1  b"));
        assert!(described.iter().any(|(c, _)| c == "-  10  x"));
        assert!(described.iter().any(|(c, _)| c == "+  10  y"));
    }

    #[test]
    fn buffered_text_flushes_at_the_completion_boundary() {
        let described = describe(&lines(vec![
            AgentEvent::Tokens(ChunkTokens {
                thinking: Some("glued".into()),
                text: Some("narration".into()),
            }),
            AgentEvent::CompletionStarted,
            thinking("next round"),
        ]));

        assert_eq!(
            described,
            vec![
                (" ".into(), Modifier::empty()),
                ("glued".into(), Modifier::empty()),
                (" ".into(), Modifier::empty()),
                ("narration".into(), Modifier::empty()),
                (" ".into(), Modifier::empty()),
                (" ".into(), Modifier::empty()),
                ("next round".into(), Modifier::empty()),
            ]
        );
    }

    #[test]
    fn thinking_is_live_again_on_the_next_completion() {
        let described = describe(&lines(vec![
            thinking("first "),
            text("narration"),
            AgentEvent::CompletionStarted,
            thinking("second "),
            text("answer"),
        ]));

        assert_eq!(
            described,
            vec![
                (" ".into(), Modifier::empty()),
                ("first ".into(), Modifier::empty()),
                (" ".into(), Modifier::empty()),
                ("narration".into(), Modifier::empty()),
                (" ".into(), Modifier::empty()),
                (" ".into(), Modifier::empty()),
                ("second ".into(), Modifier::empty()),
                (" ".into(), Modifier::empty()),
                ("answer".into(), Modifier::empty()),
                (" ".into(), Modifier::empty()),
            ]
        );
    }

    #[test]
    fn tool_result_with_matching_header_skips_the_header() {
        let described = describe(&lines(vec![
            AgentEvent::ToolStarted {
                header: "read_file: a.txt".into(),
                body: None,
            },
            AgentEvent::ToolResult {
                header: "read_file: a.txt".into(),
                body: "line one".into(),
            },
        ]));

        assert_eq!(
            described,
            vec![
                (" ".into(), Modifier::empty()),
                ("⚙ read_file: a.txt".into(), Modifier::BOLD),
                ("  line one".into(), Modifier::empty()),
                (" ".into(), Modifier::empty()),
            ]
        );
    }

    #[test]
    fn parallel_tool_results_get_their_own_blocks() {
        let described = describe(&lines(vec![
            AgentEvent::ToolStarted {
                header: "read_file: a.txt".into(),
                body: None,
            },
            AgentEvent::ToolStarted {
                header: "read_file: b.txt".into(),
                body: None,
            },
            AgentEvent::ToolResult {
                header: "read_file: a.txt".into(),
                body: "alpha".into(),
            },
            AgentEvent::ToolResult {
                header: "read_file: b.txt".into(),
                body: "beta".into(),
            },
        ]));

        assert_eq!(
            described,
            vec![
                (" ".into(), Modifier::empty()),
                ("⚙ read_file: a.txt".into(), Modifier::BOLD),
                (" ".into(), Modifier::empty()),
                ("⚙ read_file: b.txt".into(), Modifier::BOLD),
                ("  alpha".into(), Modifier::empty()),
                (" ".into(), Modifier::empty()),
                ("  beta".into(), Modifier::empty()),
                (" ".into(), Modifier::empty()),
            ]
        );
    }

    #[test]
    fn answer_chunks_flow_into_one_line() {
        let described = describe(&lines(vec![text("Hello "), text("world")]));

        assert_eq!(
            described,
            vec![
                (" ".into(), Modifier::empty()),
                ("Hello world".into(), Modifier::empty()),
                (" ".into(), Modifier::empty()),
            ]
        );
    }

    #[test]
    fn answer_newlines_split_lines() {
        let described = describe(&lines(vec![text("first\nsecond")]));

        assert_eq!(
            described,
            vec![
                (" ".into(), Modifier::empty()),
                ("first".into(), Modifier::empty()),
                ("second".into(), Modifier::empty()),
                (" ".into(), Modifier::empty()),
            ]
        );
    }

    #[test]
    fn user_question_is_pushed_to_the_scrollback() {
        let mut renderer = TuiRenderer::new();
        renderer.push_user("do the thing");
        renderer.finish();

        let described = describe(renderer.scrollback());
        assert_eq!(
            described,
            vec![
                (" ".into(), Modifier::empty()),
                ("do the thing".into(), Modifier::BOLD),
                (" ".into(), Modifier::empty()),
            ]
        );
    }

    #[test]
    fn user_block_stores_white_text_and_the_block_kind() {
        let mut renderer = TuiRenderer::new();
        renderer.push_user("do the thing");
        renderer.finish();

        let line = &renderer.scrollback()[1];
        let span = &line.spans[0];
        assert_eq!(span.style.fg, Some(Color::Rgb(212, 212, 212)));
        assert_eq!(span.style.bg, None);
        assert!(span.style.add_modifier.contains(Modifier::BOLD));
        assert_eq!(renderer.blocks()[1], BlockKind::User);
    }

    #[test]
    fn a_leading_blank_line_in_the_answer_is_not_pushed() {
        let described = describe(&lines(vec![text("\n\nHello"), text("\nworld")]));

        assert_eq!(
            described,
            vec![
                (" ".into(), Modifier::empty()),
                ("Hello".into(), Modifier::empty()),
                ("world".into(), Modifier::empty()),
                (" ".into(), Modifier::empty()),
            ]
        );
    }

    #[test]
    fn a_leading_newline_in_thinking_is_not_pushed() {
        let described = describe(&lines(vec![thinking("\n\nLet me think."), text("42")]));

        assert_eq!(
            described,
            vec![
                (" ".into(), Modifier::empty()),
                ("Let me think.".into(), Modifier::empty()),
                (" ".into(), Modifier::empty()),
                ("42".into(), Modifier::empty()),
                (" ".into(), Modifier::empty()),
            ]
        );
    }

    #[test]
    fn an_internal_blank_line_in_the_answer_is_kept() {
        let described = describe(&lines(vec![text("one\n\ntwo")]));

        assert_eq!(
            described,
            vec![
                (" ".into(), Modifier::empty()),
                ("one".into(), Modifier::empty()),
                ("".into(), Modifier::empty()),
                ("two".into(), Modifier::empty()),
                (" ".into(), Modifier::empty()),
            ]
        );
    }

    #[test]
    fn a_tool_block_turns_done_when_the_result_arrives() {
        let mut renderer = TuiRenderer::new();
        renderer.on_event(AgentEvent::ToolStarted {
            header: "bash".into(),
            body: Some("ls".into()),
        });
        assert!(renderer.blocks().iter().all(|b| *b == BlockKind::ToolRunning));
        assert_eq!(renderer.blocks().len(), 3);

        renderer.on_event(AgentEvent::ToolResult {
            header: "bash".into(),
            body: "ok".into(),
        });
        assert!(renderer.blocks().iter().all(|b| *b == BlockKind::ToolDone));
        assert_eq!(renderer.blocks().len(), 5);
    }

    #[test]
    fn an_unfinished_tool_block_stays_running() {
        let mut renderer = TuiRenderer::new();
        renderer.on_event(AgentEvent::ToolStarted {
            header: "bash".into(),
            body: None,
        });
        renderer.finish();
        assert!(renderer.blocks().iter().all(|b| *b == BlockKind::ToolRunning));
        assert_eq!(renderer.blocks().len(), 2);
    }

    #[test]
    fn parallel_tool_blocks_each_turn_done_on_their_own_result() {
        let mut renderer = TuiRenderer::new();
        renderer.on_event(AgentEvent::ToolStarted {
            header: "a".into(),
            body: None,
        });
        renderer.on_event(AgentEvent::ToolStarted {
            header: "b".into(),
            body: None,
        });
        renderer.on_event(AgentEvent::ToolResult {
            header: "a".into(),
            body: "alpha".into(),
        });
        assert_eq!(renderer.blocks()[0], BlockKind::ToolDone);
        assert_eq!(renderer.blocks()[2], BlockKind::ToolRunning);
        renderer.on_event(AgentEvent::ToolResult {
            header: "b".into(),
            body: "beta".into(),
        });
        assert!(renderer.blocks().iter().all(|b| *b == BlockKind::ToolDone));
    }

    #[test]
    fn a_trailing_newline_in_a_tool_result_is_not_pushed() {
        let mut renderer = TuiRenderer::new();
        renderer.on_event(AgentEvent::ToolStarted {
            header: "bash".into(),
            body: Some("echo out".into()),
        });
        renderer.on_event(AgentEvent::ToolResult {
            header: "bash".into(),
            body: "out\n".into(),
        });

        assert_eq!(renderer.scrollback().len(), 5);
        assert_eq!(renderer.scrollback()[3].spans[0].content.as_ref(), "  out");
    }

    #[test]
    fn markdown_answer_renders_styled_lines_at_the_boundary() {
        let rendered = lines(vec![
            text("Some **bold** text"),
            AgentEvent::CompletionStarted,
        ]);

        assert_eq!(rendered.len(), 3);
        assert_eq!(rendered[1].spans.len(), 3);
        assert_eq!(rendered[1].spans[1].content, "bold");
        assert!(rendered[1].spans[1].style.add_modifier.contains(Modifier::BOLD));
    }

    #[test]
    fn markdown_heading_and_paragraph_render_styled() {
        let rendered = lines(vec![
            text("# Head\n\n**Bold**"),
            AgentEvent::CompletionStarted,
        ]);

        assert_eq!(rendered.len(), 5);
        let heading = &rendered[1].spans[0];
        assert!(heading.style.add_modifier.contains(Modifier::BOLD));
        assert!(heading.style.add_modifier.contains(Modifier::UNDERLINED));
        assert!(rendered[3].spans[0].style.add_modifier.contains(Modifier::BOLD));
    }

    #[test]
    fn thinking_with_markdown_is_rendered() {
        let rendered = lines(vec![thinking("**bold** thinking"), text("done")]);

        let thinking_line = &rendered[1];
        assert!(thinking_line.spans.iter().any(|s| {
            s.content.as_ref() == "bold"
                && s.style.add_modifier.contains(Modifier::BOLD)
        }));
    }

    #[test]
    fn plain_thinking_is_not_reparsed() {
        let described = describe(&lines(vec![thinking("just thinking"), text("done")]));

        assert_eq!(
            described,
            vec![
                (" ".into(), Modifier::empty()),
                ("just thinking".into(), Modifier::empty()),
                (" ".into(), Modifier::empty()),
                ("done".into(), Modifier::empty()),
                (" ".into(), Modifier::empty()),
            ]
        );
    }

    #[test]
    fn user_input_with_markdown_is_rendered() {
        let mut renderer = TuiRenderer::new();
        renderer.push_user("make it **bold**");
        renderer.finish();

        let line = &renderer.scrollback()[1];
        assert!(line.spans.iter().any(|s| {
            s.content.as_ref() == "bold" && s.style.add_modifier.contains(Modifier::BOLD)
        }));
    }

    #[test]
    fn plain_user_input_is_not_reparsed() {
        let mut renderer = TuiRenderer::new();
        renderer.push_user("just a question");
        renderer.finish();

        let line = &renderer.scrollback()[1];
        assert_eq!(line.spans[0].content.as_ref(), "just a question");
        assert!(line.spans[0].style.add_modifier.contains(Modifier::BOLD));
    }

    #[test]
    fn user_markdown_gets_pi_base_and_plain_bold() {
        let mut renderer = TuiRenderer::new();
        renderer.push_user("plain **bold** and `code`");
        renderer.finish();

        let line = &renderer.scrollback()[1];
        let spans = &line.spans;
        assert_eq!(spans[0].style.fg, Some(Color::Rgb(212, 212, 212)));
        let bold_span = spans.iter().find(|s| s.content.as_ref() == "bold").unwrap();
        assert_eq!(bold_span.style.fg, Some(Color::Rgb(212, 212, 212)));
        assert!(bold_span.style.add_modifier.contains(Modifier::BOLD));
        let code_span = spans.iter().find(|s| s.content.as_ref() == "code").unwrap();
        assert_eq!(code_span.style.fg, Some(Color::Rgb(138, 190, 183)));
        assert_eq!(code_span.style.bg, None);
    }

    #[test]
    fn thinking_markdown_gets_black_base_and_black_bold() {
        let rendered = lines(vec![thinking("plain **bold**"), text("done")]);

        let line = &rendered[1];
        let spans = &line.spans;
        assert_eq!(spans[0].style.fg, Some(Color::Rgb(128, 128, 128)));
        let bold_span = spans.iter().find(|s| s.content.as_ref() == "bold").unwrap();
        assert_eq!(bold_span.style.fg, Some(Color::Rgb(128, 128, 128)));
        assert!(bold_span.style.add_modifier.contains(Modifier::BOLD));
    }

    #[test]
    fn headings_render_pi_style_without_markers() {
        let rendered = lines(vec![text("# One\n\n## Two"), AgentEvent::CompletionStarted]);

        let one = rendered.iter().find(|l| l.spans.iter().any(|s| s.content.as_ref() == "One")).unwrap();
        let one_span = one.spans.iter().find(|s| s.content.as_ref() == "One").unwrap();
        assert_eq!(one_span.style.fg, Some(Color::Rgb(0xf0, 0xc6, 0x74)));
        assert!(one_span.style.add_modifier.contains(Modifier::BOLD));
        assert!(one_span.style.add_modifier.contains(Modifier::UNDERLINED));
        assert!(!one.to_string().contains('#'));

        let two = rendered.iter().find(|l| l.spans.iter().any(|s| s.content.as_ref() == "Two")).unwrap();
        let two_span = two.spans.iter().find(|s| s.content.as_ref() == "Two").unwrap();
        assert_eq!(two_span.style.fg, Some(Color::Rgb(0xf0, 0xc6, 0x74)));
        assert!(two_span.style.add_modifier.contains(Modifier::BOLD));
        assert!(!two_span.style.add_modifier.contains(Modifier::UNDERLINED));
    }

    #[test]
    fn links_render_pi_style() {
        let rendered = lines(vec![text("[label](https://example.com)"), AgentEvent::CompletionStarted]);

        let label = rendered
            .iter()
            .find(|l| l.spans.iter().any(|s| s.content.as_ref() == "label"))
            .unwrap();
        let label_span = label.spans.iter().find(|s| s.content.as_ref() == "label").unwrap();
        assert_eq!(label_span.style.fg, Some(Color::Rgb(0x81, 0xa2, 0xbe)));
        assert!(label_span.style.add_modifier.contains(Modifier::UNDERLINED));
    }

    #[test]
    fn quotes_render_gray_italic() {
        let rendered = lines(vec![text("> quoted"), AgentEvent::CompletionStarted]);

        let quoted = rendered
            .iter()
            .find(|l| l.spans.iter().any(|s| s.content.as_ref() == "quoted"))
            .unwrap();
        let quoted_span = quoted.spans.iter().find(|s| s.content.as_ref() == "quoted").unwrap();
        assert_eq!(quoted_span.style.fg, Some(Color::Rgb(128, 128, 128)));
        assert!(quoted_span.style.add_modifier.contains(Modifier::ITALIC));
    }

    #[test]
    fn markdown_rendering_strips_vs16() {
        let mut renderer = TuiRenderer::new();
        renderer.push_user("icon \u{23F8}\u{FE0F} note");
        renderer.finish();

        for line in renderer.scrollback() {
            for span in &line.spans {
                assert!(!span.content.to_string().contains('\u{FE0F}'));
            }
        }
    }

    #[test]
    fn tool_output_is_not_markdown_rendered() {
        let described = describe(&lines(vec![AgentEvent::ToolStarted {
            header: "bash".into(),
            body: Some("**cmd**".into()),
        }]));

        assert_eq!(
            described,
            vec![
                (" ".into(), Modifier::empty()),
                ("⚙ bash".into(), Modifier::BOLD),
                ("  **cmd**".into(), Modifier::empty()),
            ]
        );
    }

    #[test]
    fn each_completion_renders_its_own_answer() {
        let rendered = lines(vec![
            text("first **a**"),
            AgentEvent::CompletionStarted,
            text("second **b**"),
        ]);

        assert_eq!(rendered.len(), 6);
        assert!(rendered[1].spans[1].style.add_modifier.contains(Modifier::BOLD));
        assert_eq!(rendered[1].spans[1].content, "a");
        assert!(rendered[4].spans[1].style.add_modifier.contains(Modifier::BOLD));
        assert_eq!(rendered[4].spans[1].content, "b");
    }

    #[test]
    fn code_block_renders_without_fences() {
        let rendered = lines(vec![
            text("```python\nx = 1\n```\n"),
            AgentEvent::CompletionStarted,
        ]);

        for line in &rendered {
            assert!(!line.to_string().contains("```"));
        }
    }

    #[test]
    fn asterisk_horizontal_rule_is_preserved() {
        let rendered = lines(vec![
            text("para\n\n***\n\nnext"),
            AgentEvent::CompletionStarted,
        ]);
        let described: Vec<String> = rendered.iter().map(|l| l.to_string()).collect();

        assert!(described.contains(&"***".into()), "{described:?}");
        assert!(!described.iter().any(|l| l == "---"), "{described:?}");
    }

    #[test]
    fn dash_horizontal_rule_stays_a_dash_rule() {
        let rendered = lines(vec![
            text("para\n\n---\n\nnext"),
            AgentEvent::CompletionStarted,
        ]);
        let described: Vec<String> = rendered.iter().map(|l| l.to_string()).collect();

        assert!(described.contains(&"---".into()), "{described:?}");
    }

    #[test]
    fn rule_adjacent_to_paragraph_is_not_merged() {
        let rendered = lines(vec![
            text("para\n***\nnext"),
            AgentEvent::CompletionStarted,
        ]);
        let described: Vec<String> = rendered.iter().map(|l| l.to_string()).collect();

        assert!(described.contains(&"***".into()), "{described:?}");
        assert!(
            !described.iter().any(|l| l.contains("para ***") || l.contains("*** next")),
            "{described:?}"
        );
    }

    #[test]
    fn rule_like_line_inside_code_fence_is_untouched() {
        let rendered = lines(vec![
            text("```bash\n***\n```
"),
            AgentEvent::CompletionStarted,
        ]);
        let described: Vec<String> = rendered.iter().map(|l| l.to_string()).collect();

        assert!(described.contains(&"***".into()), "{described:?}");
    }

    #[test]
    fn underscore_rule_is_preserved() {
        let rendered = lines(vec![
            text("para\n\n___\n\nnext"),
            AgentEvent::CompletionStarted,
        ]);
        let described: Vec<String> = rendered.iter().map(|l| l.to_string()).collect();

        assert!(described.contains(&"___".into()), "{described:?}");
        assert!(!described.iter().any(|l| l == "---"), "{described:?}");
    }

    #[test]
    fn code_block_gets_syntax_highlighting() {
        let rendered = lines(vec![
            text("```python\nx = 1\n```\n"),
            AgentEvent::CompletionStarted,
        ]);

        let has_color = rendered
            .iter()
            .any(|line| line.spans.iter().any(|span| span.style.fg.is_some()));
        assert!(has_color);
    }

    #[test]
    fn vs16_emoji_are_stripped_from_the_scrollback() {
        let rendered = lines(vec![
            text("# \u{23F8}\u{FE0F} Note\n\nbody"),
            AgentEvent::CompletionStarted,
        ]);
        let described: Vec<String> = rendered.iter().map(|l| l.to_string()).collect();

        assert!(
            !described.iter().any(|l| l.contains('\u{FE0F}')),
            "VS16 selector must not reach the scrollback: {described:?}"
        );
        assert!(
            described.iter().any(|l| l.contains("\u{23F8}")),
            "the emoji itself must survive: {described:?}"
        );
    }

    #[test]
    fn vs16_emoji_are_stripped_from_live_answer_lines() {
        let mut renderer = TuiRenderer::new();
        renderer.on_event(text("line with \u{23F8}\u{FE0F} emoji\n"));
        renderer.finish();

        let described: Vec<String> = renderer
            .scrollback()
            .iter()
            .map(|l| l.to_string())
            .collect();
        assert!(
            !described.iter().any(|l| l.contains('\u{FE0F}')),
            "VS16 selector must not reach the scrollback: {described:?}"
        );
    }

    #[test]
    fn replay_context_renders_the_transcript() {
        use openai_oxide::types::chat::{FunctionCall, ToolCall};
        let mut renderer = TuiRenderer::new();
        let mut context = Context::new("sys", 100);
        context.messages.push(Message::User { content: "do the thing".into() });
        context.messages.push(Message::Assistant {
            content: None,
            tool_calls: vec![ToolCall {
                id: "call-1".into(),
                type_: "function".into(),
                function: FunctionCall {
                    name: "bash".into(),
                    arguments: r#"{"command":"ls"}"#.into(),
                },
            }],
        });
        context.messages.push(Message::Tool {
            tool_call_id: "call-1".into(),
            content: "file.txt".into(),
        });
        context.messages.push(Message::Assistant {
            content: Some("done".into()),
            tool_calls: vec![],
        });

        renderer.replay_context(&context);

        let described = describe(&renderer.scrollback());
        assert!(described.iter().any(|(c, _)| c.contains("do the thing")));
        assert!(described.iter().any(|(c, _)| c.contains("bash")));
        assert!(described.iter().any(|(c, _)| c.contains("ls")));
        assert!(described.iter().any(|(c, _)| c.contains("file.txt")));
        assert!(described.iter().any(|(c, _)| c.contains("done")));
    }
}
