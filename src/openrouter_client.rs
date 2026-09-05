//! A [`ModelClient`] backed by the real `OpenRouter` API via `openrouter-rs`.
//! The Bash tool's JSON schema is hardcoded here, since it never varies.

use eyre::{Result, WrapErr};
use openrouter_rs::{
    OpenRouterClient as RealClient,
    api::chat::{ChatCompletionRequest, Message},
    types::{Role, Tool, ToolCall},
};
use serde_json::{Value, json};

use crate::model_client::{
    Context, ContextMessage, ModelClient, ModelResponse, ToolCallRequest, Usage,
};

const BASH_TOOL_NAME: &str = "bash";

fn bash_tool() -> Tool {
    Tool::new(
        BASH_TOOL_NAME,
        "Runs a shell command via `bash -c` and returns its stdout, stderr, and exit code.",
        json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to run."
                }
            },
            "required": ["command"]
        }),
    )
}

fn to_openrouter_tool_call(tool_call: &ToolCallRequest) -> ToolCall {
    ToolCall::new(
        tool_call.id.clone(),
        BASH_TOOL_NAME,
        json!({ "command": tool_call.command }).to_string(),
    )
}

fn to_openrouter_message(message: &ContextMessage) -> Message {
    match message {
        ContextMessage::System(text) => Message::new(Role::System, text.clone()),
        ContextMessage::User(text) => Message::new(Role::User, text.clone()),
        ContextMessage::Assistant { text, tool_calls } if tool_calls.is_empty() => {
            Message::new(Role::Assistant, text.clone().unwrap_or_default())
        }
        ContextMessage::Assistant { text, tool_calls } => Message::assistant_with_tool_calls(
            text.clone().unwrap_or_default(),
            tool_calls.iter().map(to_openrouter_tool_call).collect(),
        ),
        ContextMessage::ToolResult {
            tool_call_id,
            content,
        } => Message::tool_response(tool_call_id, content.clone()),
    }
}

fn from_openrouter_tool_call(tool_call: &ToolCall) -> Result<ToolCallRequest> {
    let arguments: Value = serde_json::from_str(&tool_call.function.arguments)
        .wrap_err("model requested the bash tool with arguments that are not valid JSON")?;
    let command = arguments
        .get("command")
        .and_then(Value::as_str)
        .ok_or_else(|| eyre::eyre!("model's bash tool call is missing a `command` string"))?;

    Ok(ToolCallRequest {
        id: tool_call.id.clone(),
        command: command.to_string(),
    })
}

/// A [`ModelClient`] that talks to the real `OpenRouter` API for a fixed
/// model slug.
pub struct OpenRouterClient {
    client: RealClient,
    model: String,
}

impl OpenRouterClient {
    /// Builds a client for `model`, reading `OPENROUTER_API_KEY` from the
    /// environment. Per the project spec, the API key is never read from
    /// config or any other source.
    ///
    /// # Errors
    ///
    /// Returns an error if `OPENROUTER_API_KEY` is unset or the underlying
    /// HTTP client fails to build.
    pub fn new(model: impl Into<String>) -> Result<Self> {
        let api_key = std::env::var("OPENROUTER_API_KEY")
            .wrap_err("OPENROUTER_API_KEY environment variable is not set")?;
        let client = RealClient::builder()
            .api_key(api_key)
            .http_referer("https://github.com/thibthibaut/tib")
            .x_title("tib")
            .build()
            .wrap_err("failed to build OpenRouter client")?;

        Ok(Self {
            client,
            model: model.into(),
        })
    }
}

impl ModelClient for OpenRouterClient {
    async fn complete(&self, context: &Context) -> Result<ModelResponse> {
        let messages: Vec<Message> = context.messages.iter().map(to_openrouter_message).collect();

        let request = ChatCompletionRequest::builder()
            .model(self.model.clone())
            .messages(messages)
            .tools(vec![bash_tool()])
            .tool_choice_auto()
            .build()
            .wrap_err("failed to build chat completion request")?;

        let response = self
            .client
            .chat()
            .create(&request)
            .await
            .wrap_err("OpenRouter chat completion request failed")?;

        let choice = response
            .choices
            .first()
            .ok_or_else(|| eyre::eyre!("model response contained no choices"))?;

        if let Some(error) = choice.error() {
            return Err(eyre::eyre!(
                "model response contained a provider error: {}",
                error.message
            ));
        }

        let text = choice.content().map(str::to_string);
        let tool_calls = choice
            .tool_calls()
            .unwrap_or_default()
            .iter()
            .map(from_openrouter_tool_call)
            .collect::<Result<Vec<_>>>()?;

        let usage = response.usage.map_or(
            Usage {
                total_tokens: 0,
                cost: 0.0,
            },
            |usage| Usage {
                total_tokens: usage.total_tokens,
                cost: usage.cost.unwrap_or(0.0),
            },
        );

        Ok(ModelResponse {
            text,
            tool_calls,
            usage,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use openrouter_rs::api::chat::Content;

    fn text_of(message: &Message) -> &str {
        match &message.content {
            Content::Text(text) => text,
            _ => panic!("expected text content"),
        }
    }

    #[test]
    fn system_and_user_messages_convert_to_plain_text_messages() {
        let system = to_openrouter_message(&ContextMessage::System("be terse".to_string()));
        assert_eq!(system.role, Role::System);
        assert_eq!(text_of(&system), "be terse");

        let user = to_openrouter_message(&ContextMessage::User("hi".to_string()));
        assert_eq!(user.role, Role::User);
        assert_eq!(text_of(&user), "hi");
    }

    #[test]
    fn text_only_assistant_message_has_no_tool_calls() {
        let assistant = to_openrouter_message(&ContextMessage::Assistant {
            text: Some("hello".to_string()),
            tool_calls: Vec::new(),
        });

        assert_eq!(assistant.role, Role::Assistant);
        assert_eq!(text_of(&assistant), "hello");
        assert!(assistant.tool_calls.is_none());
    }

    #[test]
    fn assistant_message_with_tool_calls_carries_them_as_bash_function_calls() {
        let assistant = to_openrouter_message(&ContextMessage::Assistant {
            text: None,
            tool_calls: vec![ToolCallRequest {
                id: "call_1".to_string(),
                command: "ls -la".to_string(),
            }],
        });

        let tool_calls = assistant.tool_calls.unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id, "call_1");
        assert_eq!(tool_calls[0].function.name, BASH_TOOL_NAME);
        assert_eq!(
            tool_calls[0].function.arguments,
            json!({ "command": "ls -la" }).to_string()
        );
    }

    #[test]
    fn tool_result_message_carries_its_tool_call_id() {
        let tool_result = to_openrouter_message(&ContextMessage::ToolResult {
            tool_call_id: "call_1".to_string(),
            content: "exit_code: 0".to_string(),
        });

        assert_eq!(tool_result.role, Role::Tool);
        assert_eq!(tool_result.tool_call_id.as_deref(), Some("call_1"));
        assert_eq!(text_of(&tool_result), "exit_code: 0");
    }

    #[test]
    fn from_openrouter_tool_call_extracts_the_command() {
        let tool_call = ToolCall::new("call_1", BASH_TOOL_NAME, r#"{"command":"echo hi"}"#);

        let request = from_openrouter_tool_call(&tool_call).unwrap();

        assert_eq!(request.id, "call_1");
        assert_eq!(request.command, "echo hi");
    }

    #[test]
    fn from_openrouter_tool_call_rejects_arguments_missing_a_command_field() {
        let tool_call = ToolCall::new("call_1", BASH_TOOL_NAME, r#"{"not_command":"echo hi"}"#);

        let error = from_openrouter_tool_call(&tool_call).unwrap_err();

        assert!(format!("{error}").contains("command"));
    }

    #[test]
    fn from_openrouter_tool_call_rejects_malformed_json_arguments() {
        let tool_call = ToolCall::new("call_1", BASH_TOOL_NAME, "not json");

        let error = from_openrouter_tool_call(&tool_call).unwrap_err();

        assert!(format!("{error}").contains("JSON"));
    }
}
