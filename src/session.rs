//! The `Session`/`SessionState` state machine that drives a conversation.
//!
//! One step at a time, executing any requested Bash tool calls for real,
//! until the model stops asking for them or the step budget runs out.

use std::time::Duration;

use eyre::Result;

use crate::bash_tool::{BashOutput, run_bash_command};
use crate::config::Config;
use crate::model_client::{Context, ContextMessage, ModelClient, ToolCallRequest, Usage};

/// The captured result of running one requested Bash tool call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolResult {
    pub tool_call_id: String,
    pub content: String,
}

/// The phase currently driving [`Session`] forward. See `CONTEXT.md`'s Loop
/// phases (Awaiting Input, Calling Model, Executing Tools).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionState {
    AwaitingUserInput,
    CallingModel {
        step: u32,
    },
    ExecutingTools {
        step: u32,
        pending: Vec<ToolCallRequest>,
        results: Vec<ToolResult>,
    },
}

fn format_tool_result(output: &BashOutput) -> String {
    let exit_code = output
        .exit_code
        .map_or_else(|| "none".to_string(), |code| code.to_string());
    format!(
        "exit_code: {exit_code}\ntimed_out: {}\nstdout:\n{}\nstderr:\n{}",
        output.timed_out, output.stdout, output.stderr
    )
}

/// The single, ephemeral conversation between the user and the model.
#[derive(Debug, Clone)]
pub struct Session {
    pub context: Context,
    pub state: SessionState,
    pub step_count: u32,
    pub tool_call_count: u32,
    /// [`Usage`] from the most recently completed [`ModelClient::complete`]
    /// call. The Context Visualization's token total lags by one step
    /// because it's only known once a response has completed (ADR-0004).
    pub last_usage: Option<Usage>,
}

impl Session {
    #[must_use]
    pub fn new(system_prompt: impl Into<String>) -> Self {
        Self {
            context: Context {
                messages: vec![ContextMessage::System(system_prompt.into())],
            },
            state: SessionState::AwaitingUserInput,
            step_count: 0,
            tool_call_count: 0,
            last_usage: None,
        }
    }

    /// Pushes `user_message` onto the context, then drives the loop forward:
    /// calling `client` and executing any requested Bash tool calls for
    /// real, until a step's response contains no tool calls or
    /// `config.max_steps` — this turn's budget, reset at the start of every
    /// call — is reached. Returns to [`SessionState::AwaitingUserInput`]
    /// either way, even on error.
    ///
    /// `on_update` is called with each step's assistant text accumulated so
    /// far as it streams in (see [`ModelClient::complete_streaming`]).
    ///
    /// # Errors
    ///
    /// Returns an error if `client` fails. A failing Bash tool call does not
    /// error the turn: its failure becomes that tool call's result content
    /// instead, so every `tool_call` still gets a matching result.
    pub async fn send_user_message<C: ModelClient + Sync>(
        &mut self,
        user_message: String,
        client: &C,
        config: &Config,
        on_update: impl FnMut(&str) + Send,
    ) -> Result<()> {
        self.context.push(ContextMessage::User(user_message));

        let result = self.run_steps(client, config, on_update).await;
        self.state = SessionState::AwaitingUserInput;
        result
    }

    async fn run_steps<C: ModelClient + Sync>(
        &mut self,
        client: &C,
        config: &Config,
        mut on_update: impl FnMut(&str) + Send,
    ) -> Result<()> {
        self.step_count = 0;

        while self.step_count < config.max_steps {
            self.step_count = self.step_count.saturating_add(1);
            self.state = SessionState::CallingModel {
                step: self.step_count,
            };

            let response = client
                .complete_streaming(&self.context, &mut on_update)
                .await?;
            self.last_usage = Some(response.usage);
            self.context.push(ContextMessage::Assistant {
                text: response.text,
                tool_calls: response.tool_calls.clone(),
            });

            if response.tool_calls.is_empty() {
                break;
            }

            self.state = SessionState::ExecutingTools {
                step: self.step_count,
                pending: response.tool_calls,
                results: Vec::new(),
            };

            loop {
                let tool_call = match &mut self.state {
                    SessionState::ExecutingTools { pending, .. } if !pending.is_empty() => {
                        pending.remove(0)
                    }
                    _ => break,
                };

                let command = tool_call.command.clone();
                let timeout = Duration::from_secs(config.tool_timeout_seconds);
                let truncate_limit = config.tool_output_truncate_chars;
                // Every tool_call in the assistant message already pushed to
                // context (above) needs a matching ToolResult, or the next
                // request sends OpenRouter an unanswered tool call and gets
                // rejected. So a failure here becomes the result's content
                // instead of aborting the batch via `?`.
                let content = match tokio::task::spawn_blocking(move || {
                    run_bash_command(&command, timeout, truncate_limit)
                })
                .await
                {
                    Ok(Ok(output)) => format_tool_result(&output),
                    Ok(Err(error)) => format!("error running command: {error}"),
                    Err(join_error) => format!("bash tool task panicked: {join_error}"),
                };

                let result = ToolResult {
                    tool_call_id: tool_call.id.clone(),
                    content,
                };
                self.context.push(ContextMessage::ToolResult {
                    tool_call_id: result.tool_call_id.clone(),
                    content: result.content.clone(),
                });
                self.tool_call_count = self.tool_call_count.saturating_add(1);

                if let SessionState::ExecutingTools { results, .. } = &mut self.state {
                    results.push(result);
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_client::{ModelResponse, Usage};
    use std::sync::Mutex;

    fn test_config(max_steps: u32) -> Config {
        Config {
            model: "test/model".to_string(),
            max_steps,
            system_prompt: "You are a test assistant.".to_string(),
            tool_timeout_seconds: 5,
            tool_output_truncate_chars: 1000,
        }
    }

    fn text_response(text: &str) -> ModelResponse {
        ModelResponse {
            text: Some(text.to_string()),
            tool_calls: Vec::new(),
            usage: Usage {
                total_tokens: 10,
                cost: 0.0,
            },
        }
    }

    fn tool_call_response(calls: &[(&str, &str)]) -> ModelResponse {
        ModelResponse {
            text: None,
            tool_calls: calls
                .iter()
                .map(|(id, command)| ToolCallRequest {
                    id: (*id).to_string(),
                    command: (*command).to_string(),
                })
                .collect(),
            usage: Usage {
                total_tokens: 10,
                cost: 0.0,
            },
        }
    }

    struct AlwaysErrorsModelClient;

    impl ModelClient for AlwaysErrorsModelClient {
        async fn complete(&self, _context: &Context) -> Result<ModelResponse> {
            Err(eyre::eyre!("boom"))
        }
    }

    /// Returns a pre-scripted sequence of [`ModelResponse`]s, one per call,
    /// with no network involved.
    struct FakeModelClient {
        responses: Mutex<Vec<ModelResponse>>,
    }

    impl FakeModelClient {
        fn new(responses: Vec<ModelResponse>) -> Self {
            Self {
                responses: Mutex::new(responses.into_iter().rev().collect()),
            }
        }
    }

    impl ModelClient for FakeModelClient {
        async fn complete(&self, _context: &Context) -> Result<ModelResponse> {
            self.responses
                .lock()
                .unwrap()
                .pop()
                .ok_or_else(|| eyre::eyre!("FakeModelClient ran out of scripted responses"))
        }
    }

    #[tokio::test]
    async fn text_only_response_returns_to_awaiting_user_input() {
        let client = FakeModelClient::new(vec![text_response("hi there")]);
        let config = test_config(10);
        let mut session = Session::new("system prompt");

        session
            .send_user_message("hello".to_string(), &client, &config, |_delta: &str| {})
            .await
            .unwrap();

        assert_eq!(session.state, SessionState::AwaitingUserInput);
        assert_eq!(session.step_count, 1);
        assert_eq!(session.tool_call_count, 0);
    }

    #[tokio::test]
    async fn tool_calls_run_for_real_and_the_loop_continues_until_text_only() {
        let client = FakeModelClient::new(vec![
            tool_call_response(&[("call_1", "echo one")]),
            text_response("done"),
        ]);
        let config = test_config(10);
        let mut session = Session::new("system prompt");

        session
            .send_user_message(
                "run something".to_string(),
                &client,
                &config,
                |_delta: &str| {},
            )
            .await
            .unwrap();

        assert_eq!(session.state, SessionState::AwaitingUserInput);
        assert_eq!(session.step_count, 2);
        assert_eq!(session.tool_call_count, 1);

        let tool_result = session
            .context
            .messages
            .iter()
            .find_map(|message| match message {
                ContextMessage::ToolResult { content, .. } => Some(content),
                _ => None,
            })
            .unwrap();
        assert!(tool_result.contains("one"));
    }

    #[tokio::test]
    async fn multiple_tool_calls_in_one_step_all_run_and_each_increments_the_counter() {
        let client = FakeModelClient::new(vec![
            tool_call_response(&[("call_1", "echo one"), ("call_2", "echo two")]),
            text_response("done"),
        ]);
        let config = test_config(10);
        let mut session = Session::new("system prompt");

        session
            .send_user_message(
                "run two things".to_string(),
                &client,
                &config,
                |_delta: &str| {},
            )
            .await
            .unwrap();

        assert_eq!(session.step_count, 2);
        assert_eq!(session.tool_call_count, 2);
    }

    #[tokio::test]
    async fn loop_stops_at_exactly_the_configured_max_steps() {
        let client = FakeModelClient::new(vec![
            tool_call_response(&[("call_1", "echo one")]),
            tool_call_response(&[("call_2", "echo two")]),
            tool_call_response(&[("call_3", "echo three")]),
        ]);
        let config = test_config(2);
        let mut session = Session::new("system prompt");

        session
            .send_user_message(
                "keep going".to_string(),
                &client,
                &config,
                |_delta: &str| {},
            )
            .await
            .unwrap();

        assert_eq!(session.state, SessionState::AwaitingUserInput);
        assert_eq!(session.step_count, 2);
        assert_eq!(session.tool_call_count, 2);
    }

    #[tokio::test]
    async fn state_resets_to_awaiting_user_input_even_when_the_client_errors() {
        let client = AlwaysErrorsModelClient;
        let config = test_config(10);
        let mut session = Session::new("system prompt");

        let result = session
            .send_user_message("hello".to_string(), &client, &config, |_delta: &str| {})
            .await;

        assert!(result.is_err());
        assert_eq!(session.state, SessionState::AwaitingUserInput);
    }

    #[tokio::test]
    async fn max_steps_is_a_fresh_budget_for_every_turn_not_a_session_lifetime_total() {
        let client = FakeModelClient::new(vec![
            text_response("first done"),
            text_response("second done"),
        ]);
        let config = test_config(1);
        let mut session = Session::new("system prompt");

        session
            .send_user_message("first".to_string(), &client, &config, |_delta: &str| {})
            .await
            .unwrap();
        assert_eq!(session.step_count, 1);

        session
            .send_user_message("second".to_string(), &client, &config, |_delta: &str| {})
            .await
            .unwrap();

        assert_eq!(session.state, SessionState::AwaitingUserInput);
        assert_eq!(session.step_count, 1);
    }

    #[tokio::test]
    async fn on_update_is_called_once_per_step_with_the_step_s_full_text() {
        // FakeModelClient never overrides `complete_streaming`, so this
        // exercises the trait's default implementation end to end.
        let client = FakeModelClient::new(vec![
            tool_call_response(&[("call_1", "echo one")]),
            text_response("done"),
        ]);
        let config = test_config(10);
        let mut session = Session::new("system prompt");
        let deltas = Mutex::new(Vec::new());

        session
            .send_user_message("run something".to_string(), &client, &config, |delta| {
                deltas.lock().unwrap().push(delta.to_string());
            })
            .await
            .unwrap();

        // Step 1's response is tool-calls-only (no text), so it contributes
        // no delta; step 2's "done" contributes exactly one.
        assert_eq!(deltas.into_inner().unwrap(), vec!["done".to_string()]);
    }
}
