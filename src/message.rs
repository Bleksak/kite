use openai::chat::{
    ChatCompletionMessage, ChatCompletionMessageRole, ToolCall as OpenAIToolCall, ToolCallFunction,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
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
    pub fn to_request(&self) -> ChatCompletionMessage {
        match self {
            Message::System { content } => {
                wire(ChatCompletionMessageRole::System, Some(content.as_str()))
            }
            Message::User { content } => {
                wire(ChatCompletionMessageRole::User, Some(content.as_str()))
            }
            Message::Assistant {
                content,
                tool_calls,
            } => {
                let mut message = wire(ChatCompletionMessageRole::Assistant, content.as_deref());
                if !tool_calls.is_empty() {
                    message.tool_calls =
                        Some(tool_calls.iter().map(OpenAIToolCall::from).collect());
                }
                message
            }
            Message::Tool {
                tool_call_id,
                content,
            } => {
                let mut message = wire(ChatCompletionMessageRole::Tool, Some(content.as_str()));
                message.tool_call_id = Some(tool_call_id.clone());
                message
            }
        }
    }

    pub fn from_response(message: ChatCompletionMessage) -> Message {
        match message.role {
            ChatCompletionMessageRole::System | ChatCompletionMessageRole::Developer => {
                Message::System {
                    content: message.content.unwrap_or_default(),
                }
            }
            ChatCompletionMessageRole::User => Message::User {
                content: message.content.unwrap_or_default(),
            },
            ChatCompletionMessageRole::Assistant => Message::Assistant {
                content: message.content,
                tool_calls: message
                    .tool_calls
                    .unwrap_or_default()
                    .into_iter()
                    .map(ToolCall::from)
                    .collect(),
            },
            ChatCompletionMessageRole::Tool => Message::Tool {
                tool_call_id: message
                    .tool_call_id
                    .expect("tool message without tool_call_id"),
                content: message.content.unwrap_or_default(),
            },
            ChatCompletionMessageRole::Function => panic!("legacy function role is not supported"),
        }
    }
}

fn wire(role: ChatCompletionMessageRole, content: Option<&str>) -> ChatCompletionMessage {
    ChatCompletionMessage {
        role,
        content: content.map(str::to_owned),
        name: None,
        function_call: None,
        tool_call_id: None,
        tool_calls: None,
    }
}

impl From<&ToolCall> for OpenAIToolCall {
    fn from(call: &ToolCall) -> OpenAIToolCall {
        OpenAIToolCall {
            id: call.id.clone(),
            r#type: "function".into(),
            function: ToolCallFunction {
                name: call.name.clone(),
                arguments: call.arguments.clone(),
            },
        }
    }
}

impl From<OpenAIToolCall> for ToolCall {
    fn from(call: OpenAIToolCall) -> ToolCall {
        ToolCall {
            id: call.id,
            name: call.function.name,
            arguments: call.function.arguments,
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use serde_json::json;

    #[test]
    fn system_and_user_serialize() {
        let system = Message::System {
            content: "be nice".into(),
        }
        .to_request();
        assert_eq!(system.role, ChatCompletionMessageRole::System);
        assert_eq!(system.content.as_deref(), Some("be nice"));

        let user = Message::User {
            content: "hello".into(),
        }
        .to_request();
        assert_eq!(user.role, ChatCompletionMessageRole::User);
        assert_eq!(user.content.as_deref(), Some("hello"));
    }

    #[test]
    fn assistant_tool_calls_wire_shape() {
        let message = Message::Assistant {
            content: None,
            tool_calls: vec![ToolCall {
                id: "call_123".into(),
                name: "bash".into(),
                arguments: r#"{"command":"ls"}"#.into(),
            }],
        };

        let json = serde_json::to_string(&message.to_request()).unwrap();

        assert_eq!(
            json,
            r#"{"role":"assistant","content":null,"tool_calls":[{"id":"call_123","type":"function","function":{"name":"bash","arguments":"{\"command\":\"ls\"}"}}]}"#
        );
    }

    #[test]
    fn assistant_text_only_has_no_tool_calls() {
        let message = Message::Assistant {
            content: Some("hi".into()),
            tool_calls: vec![],
        };
        let wire = message.to_request();

        assert_eq!(wire.role, ChatCompletionMessageRole::Assistant);
        assert_eq!(wire.content.as_deref(), Some("hi"));
        assert!(wire.tool_calls.is_none());
    }

    #[test]
    fn tool_message_carries_tool_call_id() {
        let message = Message::Tool {
            tool_call_id: "call_123".into(),
            content: "ok".into(),
        };
        let wire = message.to_request();

        assert_eq!(wire.role, ChatCompletionMessageRole::Tool);
        assert_eq!(wire.tool_call_id.as_deref(), Some("call_123"));
        assert_eq!(wire.content.as_deref(), Some("ok"));
    }

    #[test]
    fn from_response_parses_assistant_tool_calls() {
        let value = json!({
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "id": "call_123",
                "type": "function",
                "function": { "name": "read_file", "arguments": "{\"path\":\"a.txt\"}" }
            }]
        });
        let wire: ChatCompletionMessage = serde_json::from_value(value).unwrap();

        let message = Message::from_response(wire);

        assert_eq!(
            message,
            Message::Assistant {
                content: None,
                tool_calls: vec![ToolCall {
                    id: "call_123".into(),
                    name: "read_file".into(),
                    arguments: "{\"path\":\"a.txt\"}".into(),
                }],
            }
        );
    }

    #[test]
    fn from_response_parses_tool_message() {
        let value = json!({
            "role": "tool",
            "tool_call_id": "call_123",
            "content": "file content"
        });
        let wire: ChatCompletionMessage = serde_json::from_value(value).unwrap();

        let message = Message::from_response(wire);

        assert_eq!(
            message,
            Message::Tool {
                tool_call_id: "call_123".into(),
                content: "file content".into(),
            }
        );
    }

    #[test]
    fn from_response_missing_content_is_empty() {
        let value = json!({ "role": "user" });
        let wire: ChatCompletionMessage = serde_json::from_value(value).unwrap();

        assert_eq!(
            Message::from_response(wire),
            Message::User {
                content: String::new()
            }
        );
    }

    #[test]
    fn roundtrip_preserves_every_message_kind() {
        let messages = vec![
            Message::System {
                content: "sys".into(),
            },
            Message::User {
                content: "hi".into(),
            },
            Message::Assistant {
                content: Some("hello".into()),
                tool_calls: vec![],
            },
            Message::Assistant {
                content: None,
                tool_calls: vec![
                    ToolCall {
                        id: "call_1".into(),
                        name: "bash".into(),
                        arguments: "{}".into(),
                    },
                    ToolCall {
                        id: "call_2".into(),
                        name: "write_file".into(),
                        arguments: "{\"path\":\"a.txt\",\"content\":\"x\"}".into(),
                    },
                ],
            },
            Message::Tool {
                tool_call_id: "call_1".into(),
                content: "result".into(),
            },
        ];

        for message in messages {
            let wire: ChatCompletionMessage =
                serde_json::from_str(&serde_json::to_string(&message.to_request()).unwrap())
                    .unwrap();
            assert_eq!(Message::from_response(wire), message);
        }
    }
}
