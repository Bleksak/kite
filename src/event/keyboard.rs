use bitflags::bitflags;

const ESC_CODEPOINT: char = '\x1b';
const ENTER_CODEPOINT: char = '\r';
const CTRL_A_CODEPOINT: char = '\x01';
const CTRL_Z_CODEPOINT: char = '\x1a';
const LETTER_A: u8 = b'a';

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum KeyCode {
    Char(char),
    Enter,
    Backspace,
    Insert,
    Delete,
    Left,
    Right,
    Up,
    Down,
    Home,
    End,
    PageUp,
    PageDown,
    Tab,
    Esc,
    F(u8),
}

bitflags! {
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub struct KeyModifiers: u8 {
        const NONE = 0;
        const SHIFT = 1;
        const CONTROL = 2;
        const ALT = 4;
        const SUPER = 8;
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct KeyEvent {
    pub code: KeyCode,
    pub modifiers: KeyModifiers,
}

impl KeyEvent {
    pub fn new(code: KeyCode, modifiers: KeyModifiers) -> Self {
        Self { code, modifiers }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum MouseEventKind {
    ScrollUp,
    ScrollDown,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct MouseEvent {
    pub kind: MouseEventKind,
}

#[derive(Clone, PartialEq, Eq)]
pub enum TermEvent {
    Key(KeyEvent),
    Mouse(MouseEvent),
    Paste(String),
}

pub fn map_termwiz_event(event: termwiz::input::InputEvent) -> Option<TermEvent> {
    use termwiz::input::{KeyCode as TermwizKeyCode, Modifiers};
    match event {
        termwiz::input::InputEvent::Key(key_event) => {
            let code = match key_event.key {
                TermwizKeyCode::Char(ESC_CODEPOINT) => KeyCode::Esc,
                TermwizKeyCode::Char(ENTER_CODEPOINT) => KeyCode::Enter,
                TermwizKeyCode::Char(c @ CTRL_A_CODEPOINT..=CTRL_Z_CODEPOINT) => {
                    KeyCode::Char((c as u8 - CTRL_A_CODEPOINT as u8 + LETTER_A) as char)
                }
                TermwizKeyCode::Char(c)
                    if c.is_ascii_uppercase()
                        && key_event.modifiers.contains(Modifiers::CTRL) =>
                {
                    KeyCode::Char(c.to_ascii_lowercase())
                }
                TermwizKeyCode::Char(c) => KeyCode::Char(c),
                TermwizKeyCode::Enter => KeyCode::Enter,
                TermwizKeyCode::Escape => KeyCode::Esc,
                TermwizKeyCode::Backspace => KeyCode::Backspace,
                TermwizKeyCode::Tab => KeyCode::Tab,
                TermwizKeyCode::PageUp => KeyCode::PageUp,
                TermwizKeyCode::PageDown => KeyCode::PageDown,
                TermwizKeyCode::End => KeyCode::End,
                TermwizKeyCode::Home => KeyCode::Home,
                TermwizKeyCode::Insert => KeyCode::Insert,
                TermwizKeyCode::Delete => KeyCode::Delete,
                TermwizKeyCode::LeftArrow => KeyCode::Left,
                TermwizKeyCode::RightArrow => KeyCode::Right,
                TermwizKeyCode::UpArrow => KeyCode::Up,
                TermwizKeyCode::DownArrow => KeyCode::Down,
                TermwizKeyCode::Function(n) => KeyCode::F(n),
                _ => return None,
            };
            let mut modifiers = KeyModifiers::NONE;
            if key_event.modifiers.contains(Modifiers::SHIFT) {
                modifiers |= KeyModifiers::SHIFT;
            }
            if key_event.modifiers.contains(Modifiers::ALT) {
                modifiers |= KeyModifiers::ALT;
            }
            if key_event.modifiers.contains(Modifiers::CTRL) {
                modifiers |= KeyModifiers::CONTROL;
            }
            if key_event.modifiers.contains(Modifiers::SUPER) {
                modifiers |= KeyModifiers::SUPER;
            }
            Some(TermEvent::Key(KeyEvent::new(code, modifiers)))
        }
        termwiz::input::InputEvent::Mouse(mouse) => {
            use termwiz::input::MouseButtons;
            if !mouse.mouse_buttons.contains(MouseButtons::VERT_WHEEL) {
                return None;
            }
            let kind = if mouse.mouse_buttons.contains(MouseButtons::WHEEL_POSITIVE) {
                MouseEventKind::ScrollUp
            } else {
                MouseEventKind::ScrollDown
            };
            Some(TermEvent::Mouse(MouseEvent { kind }))
        }
        termwiz::input::InputEvent::Paste(text) => Some(TermEvent::Paste(text)),
        _ => None,
    }
}
