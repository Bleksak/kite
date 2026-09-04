use openai_oxide::client::OpenAI;
use openai_oxide::error::OpenAIError;
use openai_oxide::types::chat::{ChatCompletionRequest, ChatCompletionResponse};

use crate::message::Message;
use crate::tool::{tool_definitions, Tool};

#[derive(Debug, PartialEq)]
enum Step {
    Continue,
    Done(String),
}

pub struct Agent {
    client: OpenAI,
    model: String,
    system_prompt: String,
    messages: Vec<Message>,
}

impl Agent {
    pub fn new(client: OpenAI, model: impl Into<String>, system_prompt: impl Into<String>) -> Agent {
        Agent {
            client,
            model: model.into(),
            system_prompt: system_prompt.into(),
            messages: Vec::new(),
        }
    }

    pub fn history(&self) -> &[Message] {
        &self.messages
    }

    pub async fn chat(&mut self, user_message: &str) -> Result<String, OpenAIError> {
        self.messages.push(Message::User {
            content: user_message.to_string(),
        });

        loop {
            let mut response = self.complete().await?;
            let message = Message::from_response(response.choices.remove(0).message);
            if let Step::Done(text) = self.handle_response(message).await {
                return Ok(text);
            }
        }
    }

    async fn handle_response(&mut self, message: Message) -> Step {
        let Message::Assistant { content, tool_calls } = message else {
            return Step::Done(String::new());
        };

        if tool_calls.is_empty() {
            self.messages.push(Message::Assistant {
                content: content.clone(),
                tool_calls,
            });
            return Step::Done(content.unwrap_or_default());
        }

        self.messages.push(Message::Assistant {
            content,
            tool_calls: tool_calls.clone(),
        });

        for call in tool_calls {
            let tool_call_id = call.id.clone();
            let content = match Tool::try_from(call) {
                Ok(tool) => match tool.invoke().await {
                    Ok(output) => output,
                    Err(error) => error.to_string(),
                },
                Err(error) => error.to_string(),
            };
            self.messages.push(Message::Tool {
                tool_call_id,
                content,
            });
        }

        Step::Continue
    }

    fn build_request(&self) -> ChatCompletionRequest {
        let mut request = ChatCompletionRequest::new(
            self.model.clone(),
            std::iter::once(Message::System {
                content: self.system_prompt.clone(),
            })
            .chain(self.messages.iter().cloned())
            .map(|message| message.to_request())
            .collect(),
        );
        request.tools = Some(tool_definitions());
        request
    }

    async fn complete(&self) -> Result<ChatCompletionResponse, OpenAIError> {
        self.client
            .chat()
            .completions()
            .create(self.build_request())
            .await
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use openai_oxide::types::chat::{FunctionCall, ToolCall};

    fn tool_call(id: &str, name: &str, arguments: &str) -> ToolCall {
        ToolCall {
            id: id.into(),
            type_: "function".into(),
            function: FunctionCall {
                name: name.into(),
                arguments: arguments.into(),
            },
        }
    }

    fn agent() -> Agent {
        Agent::new(OpenAI::new("test-key"), "test-model", "be concise")
    }

    async fn run_tool_call(agent: &mut Agent, id: &str, name: &str, arguments: &str) -> Step {
        agent.handle_response(Message::Assistant {
            content: None,
            tool_calls: vec![tool_call(id, name, arguments)],
        })
        .await
    }

    async fn finish(agent: &mut Agent, text: &str) -> Step {
        agent.handle_response(Message::Assistant {
            content: Some(text.into()),
            tool_calls: vec![],
        })
        .await
    }

    #[tokio::test]
    async fn simple_answer_returns_text() {
        let mut agent = agent();

        let step = finish(&mut agent, "hello").await;

        assert_eq!(step, Step::Done("hello".into()));
        assert_eq!(
            agent.history()[0],
            Message::Assistant {
                content: Some("hello".into()),
                tool_calls: vec![],
            }
        );
    }

    #[tokio::test]
    async fn tool_call_executes_and_continues() {
        let mut agent = agent();

        let step = run_tool_call(&mut agent, "call_1", "bash", r#"{"command":"echo out"}"#).await;

        assert_eq!(step, Step::Continue);
        assert_eq!(agent.history().len(), 2);
        assert_eq!(
            agent.history()[1],
            Message::Tool {
                tool_call_id: "call_1".into(),
                content: "out\n".into(),
            }
        );
    }

    #[tokio::test]
    async fn tool_error_is_returned_to_model() {
        let mut agent = agent();

        run_tool_call(&mut agent, "call_1", "bash", r#"{"command":"exit 3"}"#).await;

        assert_eq!(
            agent.history()[1],
            Message::Tool {
                tool_call_id: "call_1".into(),
                content: "bash exited with 3\n".into(),
            }
        );
    }

    #[tokio::test]
    async fn invalid_arguments_are_returned_to_model() {
        let mut agent = agent();

        run_tool_call(&mut agent, "call_1", "bash", r#"{"command":"ls""#).await;

        let Message::Tool { content, .. } = &agent.history()[1] else {
            panic!("expected tool message");
        };
        assert!(content.contains("invalid arguments for bash"));
        assert!(content.contains(r#"{"command":"ls""#));
    }

    #[tokio::test]
    async fn unknown_tool_is_returned_to_model() {
        let mut agent = agent();

        run_tool_call(&mut agent, "call_1", "nuke", "{}").await;

        assert_eq!(
            agent.history()[1],
            Message::Tool {
                tool_call_id: "call_1".into(),
                content: "unknown tool nuke".into(),
            }
        );
    }

    #[tokio::test]
    async fn parallel_tool_calls_all_get_results() {
        let mut agent = agent();
        let step = agent
            .handle_response(Message::Assistant {
                content: None,
                tool_calls: vec![
                    tool_call("call_1", "bash", r#"{"command":"echo one"}"#),
                    tool_call("call_2", "bash", r#"{"command":"echo two"}"#),
                ],
            })
            .await;
        assert_eq!(step, Step::Continue);

        assert_eq!(
            agent.history()[1],
            Message::Tool {
                tool_call_id: "call_1".into(),
                content: "one\n".into(),
            }
        );
        assert_eq!(
            agent.history()[2],
            Message::Tool {
                tool_call_id: "call_2".into(),
                content: "two\n".into(),
            }
        );
    }

    #[test]
    fn request_contains_system_prompt_and_tools() {
        let mut agent = agent();
        agent.messages.push(Message::User {
            content: "hello".into(),
        });

        let request = agent.build_request();

        assert_eq!(request.model, "test-model");
        assert_eq!(request.messages.len(), 2);

        let openai_oxide::types::chat::ChatCompletionMessageParam::System { content, .. } =
            &request.messages[0]
        else {
            panic!("expected system message first");
        };
        assert_eq!(content, "be concise");

        let openai_oxide::types::chat::ChatCompletionMessageParam::User { content, .. } =
            &request.messages[1]
        else {
            panic!("expected user message second");
        };
        let openai_oxide::types::chat::UserContent::Text(text) = content else {
            panic!("expected text content");
        };
        assert_eq!(text, "hello");

        let tools = request.tools.as_ref().unwrap();
        assert_eq!(tools.len(), 4);
        assert_eq!(
            tools.iter().map(|t| t.function.name.as_str()).collect::<Vec<_>>(),
            vec!["bash", "read_file", "write_file", "edit_file"]
        );
    }
}
