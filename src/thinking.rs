use std::sync::atomic::{AtomicU8, Ordering};

#[derive(Debug, PartialEq, Eq, Clone, Copy, clap::ValueEnum)]
pub enum ThinkingLevel {
    Off,
    Low,
    Medium,
    XHigh,
}

impl ThinkingLevel {
    pub fn next(self) -> Self {
        match self {
            Self::Off => Self::Low,
            Self::Low => Self::Medium,
            Self::Medium => Self::XHigh,
            Self::XHigh => Self::Off,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::XHigh => "xhigh",
        }
    }

    pub fn body(self, base: Option<bool>) -> Option<serde_json::Value> {
        match self {
            Self::Off => base.is_some().then(|| {
                serde_json::json!({ "chat_template_kwargs": { "enable_thinking": false } })
            }),
            Self::Low => Some(serde_json::json!({
                "chat_template_kwargs": { "enable_thinking": true },
                "reasoning_effort": "low"
            })),
            Self::Medium => Some(serde_json::json!({
                "chat_template_kwargs": { "enable_thinking": true },
                "reasoning_effort": "medium"
            })),
            Self::XHigh => Some(serde_json::json!({
                "chat_template_kwargs": { "enable_thinking": true },
                "reasoning_effort": "xhigh"
            })),
        }
    }
}

pub struct ThinkingLevelCell {
    value: AtomicU8,
}

impl ThinkingLevelCell {
    pub fn new(level: ThinkingLevel) -> ThinkingLevelCell {
        ThinkingLevelCell {
            value: AtomicU8::new(match level {
                ThinkingLevel::Off => 0,
                ThinkingLevel::Low => 1,
                ThinkingLevel::Medium => 2,
                ThinkingLevel::XHigh => 3,
            }),
        }
    }

    pub fn get(&self) -> ThinkingLevel {
        match self.value.load(Ordering::SeqCst) {
            1 => ThinkingLevel::Low,
            2 => ThinkingLevel::Medium,
            3 => ThinkingLevel::XHigh,
            _ => ThinkingLevel::Off,
        }
    }

    pub fn set(&self, level: ThinkingLevel) {
        self.value.store(match level {
            ThinkingLevel::Off => 0,
            ThinkingLevel::Low => 1,
            ThinkingLevel::Medium => 2,
            ThinkingLevel::XHigh => 3,
        }, Ordering::SeqCst);
    }
}

impl Default for ThinkingLevelCell {
    fn default() -> Self {
        Self::new(ThinkingLevel::Off)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn thinking_cycle_wraps_around() {
        let level = ThinkingLevel::Off;
        assert_eq!(level.next(), ThinkingLevel::Low);
        assert_eq!(ThinkingLevel::Low.next(), ThinkingLevel::Medium);
        assert_eq!(ThinkingLevel::Medium.next(), ThinkingLevel::XHigh);
        assert_eq!(ThinkingLevel::XHigh.next(), ThinkingLevel::Off);
    }

    #[test]
    fn off_with_auto_sends_no_override() {
        assert_eq!(ThinkingLevel::Off.body(None), None);
    }

    #[test]
    fn off_with_explicit_base_sends_thinking_disabled() {
        assert_eq!(
            ThinkingLevel::Off.body(Some(false)),
            Some(serde_json::json!({ "chat_template_kwargs": { "enable_thinking": false } }))
        );
    }

    #[test]
    fn each_level_sends_thinking_enabled_with_its_effort() {
        let cases = [
            (ThinkingLevel::Low, "low"),
            (ThinkingLevel::Medium, "medium"),
            (ThinkingLevel::XHigh, "xhigh"),
        ];
        for (level, effort) in cases {
            let body = level.body(None).unwrap();
            assert_eq!(
                body["chat_template_kwargs"]["enable_thinking"], true
            );
            assert_eq!(body["reasoning_effort"], effort);
        }
    }
}
