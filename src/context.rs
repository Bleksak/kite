use openai_oxide::types::chat::ChatCompletionMessageParam;

use crate::message::Message;

pub struct Context {
    system_prompt: String,
    messages: Vec<Message>,
    prompt_tokens: Option<u64>,
    total_prompt_tokens: u64,
    total_completion_tokens: u64,
    max_tokens: u64,
}

impl Context {
    pub fn new(system_prompt: impl Into<String>, max_tokens: u64) -> Context {
        Context {
            system_prompt: system_prompt.into(),
            messages: Vec::new(),
            prompt_tokens: None,
            total_prompt_tokens: 0,
            total_completion_tokens: 0,
            max_tokens,
        }
    }

    pub fn push(&mut self, message: Message) {
        self.messages.push(message);
    }

    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    pub fn build_messages(&self) -> Vec<ChatCompletionMessageParam> {
        std::iter::once(Message::System {
            content: self.system_prompt.clone(),
        })
        .chain(self.messages.iter().cloned())
        .map(|message| message.to_request())
        .collect()
    }

    pub fn record_usage(&mut self, prompt: Option<u64>, completion: Option<u64>) {
        if let Some(prompt) = prompt {
            self.prompt_tokens = Some(prompt);
            self.total_prompt_tokens += prompt;
        }
        if let Some(completion) = completion {
            self.total_completion_tokens += completion;
        }
    }

    pub fn total_prompt_tokens(&self) -> u64 {
        self.total_prompt_tokens
    }

    pub fn total_completion_tokens(&self) -> u64 {
        self.total_completion_tokens
    }

    pub fn needs_compaction(&self) -> bool {
        self.prompt_tokens
            .is_some_and(|tokens| tokens > self.max_tokens)
    }

    pub fn compact(&mut self) {}
}

#[cfg(test)]
mod test {
    use super::*;

    fn context() -> Context {
        Context::new("be concise", 100)
    }

    #[test]
    fn build_messages_puts_system_prompt_first() {
        let mut context = context();
        context.push(Message::User {
            content: "hi".into(),
        });

        let json: Vec<String> = context
            .build_messages()
            .iter()
            .map(|m| serde_json::to_string(m).unwrap())
            .collect();

        assert_eq!(
            json,
            vec![
                r#"{"role":"system","content":"be concise"}"#,
                r#"{"role":"user","content":"hi"}"#
            ]
        );
    }

    #[test]
    fn record_usage_tracks_latest_and_totals() {
        let mut context = context();

        context.record_usage(Some(50), Some(10));
        context.record_usage(None, None);
        context.record_usage(Some(150), Some(20));

        assert!(context.needs_compaction());
        assert_eq!(context.total_prompt_tokens(), 200);
        assert_eq!(context.total_completion_tokens(), 30);
    }

    #[test]
    fn needs_compaction_only_when_over_budget() {
        let mut context = context();

        assert!(!context.needs_compaction());
        context.record_usage(Some(100), None);
        assert!(!context.needs_compaction());
        context.record_usage(Some(101), None);
        assert!(context.needs_compaction());
    }

    #[test]
    fn compact_is_a_no_op_for_now() {
        let mut context = context();
        context.push(Message::User {
            content: "hi".into(),
        });

        context.compact();

        assert_eq!(context.messages().len(), 1);
    }
}
