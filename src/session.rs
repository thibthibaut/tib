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

/// One update from a Turn in progress.
///
/// Reported via [`Session::send_user_message`]'s `on_event` callback for
/// every change a caller would otherwise only see once the whole Turn
/// finishes and it refreshes from a `Session` snapshot: streamed text,
/// [`SessionState`] transitions, each message as it's appended (an
/// Assistant message's tool calls and each one's result included), and the
/// step/tool-call counters.
#[derive(Debug, Clone, PartialEq)]
pub enum TurnEvent {
    /// The current step's assistant text, accumulated so far (see
    /// [`ModelClient::complete_streaming`]).
    Text(String),
    /// [`Session::state`] just changed to this value.
    StateChanged(SessionState),
    /// A message was just appended to [`Session::context`] — an Assistant
    /// message (with any tool calls it requested) or the result of running
    /// one.
    MessageAppended(ContextMessage),
    /// [`Session::step_count`] and/or [`Session::tool_call_count`] and/or
    /// [`Session::last_usage`] just changed to these values (sent whenever
    /// any of them does, so all three are always current together).
    CountersChanged {
        step_count: u32,
        tool_call_count: u32,
        last_usage: Option<Usage>,
    },
}

/// Formats one Bash tool call's result. `stdout`/`stderr` sections are
/// omitted entirely when empty, rather than shown as empty-bodied headers,
/// so a command that only writes to one stream doesn't pad the transcript
/// (and the model's context) with a section that says nothing.
fn format_tool_result(output: &BashOutput) -> String {
    let exit_code = output
        .exit_code
        .map_or_else(|| "none".to_string(), |code| code.to_string());
    let mut sections = vec![
        format!("exit_code: {exit_code}"),
        format!("timed_out: {}", output.timed_out),
    ];
    if !output.stdout.is_empty() {
        sections.push(format!("stdout:\n{}", output.stdout));
    }
    if !output.stderr.is_empty() {
        sections.push(format!("stderr:\n{}", output.stderr));
    }
    sections.join("\n")
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
    /// `on_event` reports the Turn's live progress (see [`TurnEvent`]) —
    /// streamed text, state transitions, each appended message, and the
    /// counters — so a caller can reflect all of it as it happens, without
    /// waiting for the whole Turn to finish.
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
        on_event: impl FnMut(TurnEvent) + Send,
    ) -> Result<()> {
        self.push_user_message(user_message);
        self.run_turn(client, config, on_event).await
    }

    /// Pushes `user_message` onto the context, starting a new Turn. Split
    /// out from `send_user_message` so a caller (the Controller) can show
    /// the message immediately, before the model call that follows it —
    /// which may take a while — has even started; pair it with
    /// [`Session::run_turn`] to actually drive the loop.
    pub fn push_user_message(&mut self, user_message: String) {
        self.context.push(ContextMessage::User(user_message));
    }

    /// Drives the current Turn forward — calling `client` and executing any
    /// requested Bash tool calls for real, the same as `send_user_message` —
    /// without pushing a user message first, for a caller that already
    /// pushed one separately via [`Session::push_user_message`].
    ///
    /// # Errors
    ///
    /// Returns an error under the same conditions as `send_user_message`.
    pub async fn run_turn<C: ModelClient + Sync>(
        &mut self,
        client: &C,
        config: &Config,
        on_event: impl FnMut(TurnEvent) + Send,
    ) -> Result<()> {
        let result = self.run_steps(client, config, on_event).await;
        self.state = SessionState::AwaitingUserInput;
        result
    }

    /// A [`TurnEvent::CountersChanged`] snapshotting the current counters.
    const fn counters_event(&self) -> TurnEvent {
        TurnEvent::CountersChanged {
            step_count: self.step_count,
            tool_call_count: self.tool_call_count,
            last_usage: self.last_usage,
        }
    }

    async fn run_steps<C: ModelClient + Sync>(
        &mut self,
        client: &C,
        config: &Config,
        mut on_event: impl FnMut(TurnEvent) + Send,
    ) -> Result<()> {
        self.step_count = 0;

        while self.step_count < config.max_steps {
            self.step_count = self.step_count.saturating_add(1);
            on_event(self.counters_event());
            self.state = SessionState::CallingModel {
                step: self.step_count,
            };
            on_event(TurnEvent::StateChanged(self.state.clone()));

            let response = client
                .complete_streaming(&self.context, |text: &str| {
                    on_event(TurnEvent::Text(text.to_string()));
                })
                .await?;
            self.last_usage = Some(response.usage);
            on_event(self.counters_event());
            let assistant_message = ContextMessage::Assistant {
                text: response.text,
                tool_calls: response.tool_calls.clone(),
            };
            self.context.push(assistant_message.clone());
            on_event(TurnEvent::MessageAppended(assistant_message));

            if response.tool_calls.is_empty() {
                break;
            }

            self.state = SessionState::ExecutingTools {
                step: self.step_count,
                pending: response.tool_calls,
                results: Vec::new(),
            };
            on_event(TurnEvent::StateChanged(self.state.clone()));

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
                let tool_result_message = ContextMessage::ToolResult {
                    tool_call_id: result.tool_call_id.clone(),
                    content: result.content.clone(),
                };
                self.context.push(tool_result_message.clone());
                on_event(TurnEvent::MessageAppended(tool_result_message));
                self.tool_call_count = self.tool_call_count.saturating_add(1);
                on_event(self.counters_event());

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

    #[tokio::test]
    async fn run_turn_drives_an_already_pushed_message_without_pushing_another() {
        let client = FakeModelClient::new(vec![text_response("hi there")]);
        let config = test_config(10);
        let mut session = Session::new("system prompt");
        session.push_user_message("hello".to_string());

        session
            .run_turn(&client, &config, |_event: TurnEvent| {})
            .await
            .unwrap();

        assert_eq!(session.context.turn_count(), 1);
        assert_eq!(session.state, SessionState::AwaitingUserInput);
        assert_eq!(session.step_count, 1);
    }

    #[test]
    fn push_user_message_appears_in_context_immediately_without_a_model_call() {
        let mut session = Session::new("system prompt");

        session.push_user_message("hello".to_string());

        assert_eq!(session.context.turn_count(), 1);
        assert_eq!(
            session.context.messages.last(),
            Some(&ContextMessage::User("hello".to_string()))
        );
    }

    fn bash_output(stdout: &str, stderr: &str) -> BashOutput {
        BashOutput {
            stdout: stdout.to_string(),
            stderr: stderr.to_string(),
            exit_code: Some(0),
            timed_out: false,
        }
    }

    #[test]
    fn format_tool_result_omits_stdout_and_stderr_when_both_are_empty() {
        let formatted = format_tool_result(&bash_output("", ""));

        assert_eq!(formatted, "exit_code: 0\ntimed_out: false");
    }

    #[test]
    fn format_tool_result_includes_stdout_only_when_stderr_is_empty() {
        let formatted = format_tool_result(&bash_output("hello\n", ""));

        assert_eq!(
            formatted,
            "exit_code: 0\ntimed_out: false\nstdout:\nhello\n"
        );
        assert!(!formatted.contains("stderr"));
    }

    #[test]
    fn format_tool_result_includes_stderr_only_when_stdout_is_empty() {
        let formatted = format_tool_result(&bash_output("", "oops\n"));

        assert_eq!(formatted, "exit_code: 0\ntimed_out: false\nstderr:\noops\n");
        assert!(!formatted.contains("stdout"));
    }

    #[test]
    fn format_tool_result_includes_both_when_both_are_present() {
        let formatted = format_tool_result(&bash_output("hello\n", "oops\n"));

        assert_eq!(
            formatted,
            "exit_code: 0\ntimed_out: false\nstdout:\nhello\n\nstderr:\noops\n"
        );
    }

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
            .send_user_message(
                "hello".to_string(),
                &client,
                &config,
                |_event: TurnEvent| {},
            )
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
                |_event: TurnEvent| {},
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
                |_event: TurnEvent| {},
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
                |_event: TurnEvent| {},
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
            .send_user_message(
                "hello".to_string(),
                &client,
                &config,
                |_event: TurnEvent| {},
            )
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
            .send_user_message(
                "first".to_string(),
                &client,
                &config,
                |_event: TurnEvent| {},
            )
            .await
            .unwrap();
        assert_eq!(session.step_count, 1);

        session
            .send_user_message(
                "second".to_string(),
                &client,
                &config,
                |_event: TurnEvent| {},
            )
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
            .send_user_message("run something".to_string(), &client, &config, |event| {
                if let TurnEvent::Text(text) = event {
                    deltas.lock().unwrap().push(text);
                }
            })
            .await
            .unwrap();

        // Step 1's response is tool-calls-only (no text), so it contributes
        // no delta; step 2's "done" contributes exactly one.
        assert_eq!(deltas.into_inner().unwrap(), vec!["done".to_string()]);
    }

    #[tokio::test]
    async fn on_event_reports_every_state_change_as_the_turn_progresses() {
        let client = FakeModelClient::new(vec![
            tool_call_response(&[("call_1", "echo one")]),
            text_response("done"),
        ]);
        let config = test_config(10);
        let mut session = Session::new("system prompt");
        let states = Mutex::new(Vec::new());

        session
            .send_user_message("run something".to_string(), &client, &config, |event| {
                if let TurnEvent::StateChanged(state) = event {
                    states.lock().unwrap().push(state);
                }
            })
            .await
            .unwrap();

        assert_eq!(
            states.into_inner().unwrap(),
            vec![
                SessionState::CallingModel { step: 1 },
                SessionState::ExecutingTools {
                    step: 1,
                    pending: vec![ToolCallRequest {
                        id: "call_1".to_string(),
                        command: "echo one".to_string(),
                    }],
                    results: Vec::new(),
                },
                SessionState::CallingModel { step: 2 },
            ]
        );
    }

    #[tokio::test]
    async fn on_event_reports_every_message_as_it_s_appended() {
        let client = FakeModelClient::new(vec![
            tool_call_response(&[("call_1", "echo one")]),
            text_response("done"),
        ]);
        let config = test_config(10);
        let mut session = Session::new("system prompt");
        let messages = Mutex::new(Vec::new());

        session
            .send_user_message("run something".to_string(), &client, &config, |event| {
                if let TurnEvent::MessageAppended(message) = event {
                    messages.lock().unwrap().push(message);
                }
            })
            .await
            .unwrap();

        // Matches exactly the non-User messages that ended up in context,
        // in the order they were appended — a live caller sees each one as
        // it happens, not only once the whole Turn finishes.
        let expected: Vec<_> = session
            .context
            .messages
            .iter()
            .filter(|message| {
                !matches!(message, ContextMessage::System(_) | ContextMessage::User(_))
            })
            .cloned()
            .collect();
        assert_eq!(messages.into_inner().unwrap(), expected);
        assert_eq!(expected.len(), 3); // tool-call assistant msg, tool result, final assistant msg
    }

    #[tokio::test]
    async fn on_event_reports_counters_live_as_they_change() {
        let client = FakeModelClient::new(vec![
            tool_call_response(&[("call_1", "echo one")]),
            text_response("done"),
        ]);
        let config = test_config(10);
        let mut session = Session::new("system prompt");
        let snapshots = Mutex::new(Vec::new());

        session
            .send_user_message("run something".to_string(), &client, &config, |event| {
                if let TurnEvent::CountersChanged {
                    step_count,
                    tool_call_count,
                    ..
                } = event
                {
                    snapshots
                        .lock()
                        .unwrap()
                        .push((step_count, tool_call_count));
                }
            })
            .await
            .unwrap();

        assert_eq!(
            snapshots.into_inner().unwrap(),
            vec![
                (1, 0), // step 1 starts
                (1, 0), // step 1's response arrived (usage updated)
                (1, 1), // the tool call it requested just ran
                (2, 1), // step 2 starts
                (2, 1), // step 2's response arrived
            ]
        );
    }
}
