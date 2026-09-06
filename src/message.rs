use openai_oxide::types::chat::{ChatCompletionMessageParam, ToolCall, UserContent};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Message {
    System {
        content: String,
    },
    User {
        content: String,
    },
    Assistant {
        content: Option<String>,
        tool_calls: Vec<ToolCall>,
    },
    Tool {
        tool_call_id: String,
        content: String,
    },
}

impl Message {
    pub fn to_request(&self) -> ChatCompletionMessageParam {
        match self {
            Message::System { content } => ChatCompletionMessageParam::System {
                content: content.clone(),
                name: None,
            },
            Message::User { content } => ChatCompletionMessageParam::User {
                content: UserContent::Text(content.clone()),
                name: None,
            },
            Message::Assistant {
                content,
                tool_calls,
            } => {
                let tool_calls = if tool_calls.is_empty() {
                    None
                } else {
                    Some(tool_calls.clone())
                };
                ChatCompletionMessageParam::Assistant {
                    content: content.clone(),
                    name: None,
                    tool_calls,
                    refusal: None,
                }
            }
            Message::Tool {
                tool_call_id,
                content,
            } => ChatCompletionMessageParam::Tool {
                content: content.clone(),
                tool_call_id: tool_call_id.clone(),
            },
        }
    }
}

impl PartialEq for Message {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Message::System { content: a }, Message::System { content: b }) => a == b,
            (Message::User { content: a }, Message::User { content: b }) => a == b,
            (
                Message::Assistant {
                    content: a,
                    tool_calls: ta,
                },
                Message::Assistant {
                    content: b,
                    tool_calls: tb,
                },
            ) => {
                a == b
                    && ta.len() == tb.len()
                    && ta.iter().zip(tb).all(|(x, y)| {
                        x.id == y.id
                            && x.type_ == y.type_
                            && x.function.name == y.function.name
                            && x.function.arguments == y.function.arguments
                    })
            }
            (
                Message::Tool {
                    tool_call_id: a,
                    content: ca,
                },
                Message::Tool {
                    tool_call_id: b,
                    content: cb,
                },
            ) => a == b && ca == cb,
            _ => false,
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use openai_oxide::types::chat::FunctionCall;

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

    #[test]
    fn system_and_user_serialize() {
        let system = Message::System {
            content: "be nice".into(),
        }
        .to_request();
        assert_eq!(
            serde_json::to_string(&system).unwrap(),
            r#"{"role":"system","content":"be nice"}"#
        );

        let user = Message::User {
            content: "hello".into(),
        }
        .to_request();
        assert_eq!(
            serde_json::to_string(&user).unwrap(),
            r#"{"role":"user","content":"hello"}"#
        );
    }

    #[test]
    fn assistant_tool_calls_wire_shape() {
        let message = Message::Assistant {
            content: None,
            tool_calls: vec![tool_call("call_123", "bash", r#"{"command":"ls"}"#)],
        };

        let json = serde_json::to_string(&message.to_request()).unwrap();

        assert_eq!(
            json,
            r#"{"role":"assistant","tool_calls":[{"id":"call_123","type":"function","function":{"name":"bash","arguments":"{\"command\":\"ls\"}"}}]}"#
        );
    }

    #[test]
    fn assistant_text_only_has_no_tool_calls() {
        let message = Message::Assistant {
            content: Some("hi".into()),
            tool_calls: vec![],
        };

        let json = serde_json::to_string(&message.to_request()).unwrap();

        assert_eq!(json, r#"{"role":"assistant","content":"hi"}"#);
    }

    #[test]
    fn tool_message_carries_tool_call_id() {
        let message = Message::Tool {
            tool_call_id: "call_123".into(),
            content: "ok".into(),
        };

        let json = serde_json::to_string(&message.to_request()).unwrap();

        assert_eq!(
            json,
            r#"{"role":"tool","content":"ok","tool_call_id":"call_123"}"#
        );
    }
}
