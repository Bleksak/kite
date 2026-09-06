#[derive(Debug, PartialEq, Eq, Clone, Copy, clap::ValueEnum)]
pub enum ThinkingLevel {
    Off,
    Low,
    Medium,
    High,
    XHigh,
}

impl ThinkingLevel {
    pub fn next(self) -> Self {
        match self {
            Self::Off => Self::Low,
            Self::Low => Self::Medium,
            Self::Medium => Self::High,
            Self::High => Self::XHigh,
            Self::XHigh => Self::Off,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::XHigh => "xhigh",
        }
    }

    pub fn body(self, base: Option<bool>) -> Option<serde_json::Value> {
        match self {
            Self::Off => base.is_some().then(
                || serde_json::json!({ "chat_template_kwargs": { "enable_thinking": false } }),
            ),
            Self::Low => Some(serde_json::json!({
                "chat_template_kwargs": { "enable_thinking": true },
                "reasoning_effort": "low"
            })),
            Self::Medium => Some(serde_json::json!({
                "chat_template_kwargs": { "enable_thinking": true },
                "reasoning_effort": "medium"
            })),
            Self::High => Some(serde_json::json!({
                "chat_template_kwargs": { "enable_thinking": true },
                "reasoning_effort": "high"
            })),
            Self::XHigh => Some(serde_json::json!({
                "chat_template_kwargs": { "enable_thinking": true },
                "reasoning_effort": "xhigh"
            })),
        }
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
        assert_eq!(ThinkingLevel::Medium.next(), ThinkingLevel::High);
        assert_eq!(ThinkingLevel::High.next(), ThinkingLevel::XHigh);
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
            (ThinkingLevel::High, "high"),
            (ThinkingLevel::XHigh, "xhigh"),
        ];
        for (level, effort) in cases {
            let body = level.body(None).unwrap();
            assert_eq!(body["chat_template_kwargs"]["enable_thinking"], true);
            assert_eq!(body["reasoning_effort"], effort);
        }
    }
}
