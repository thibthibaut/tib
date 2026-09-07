//! The `Session`/`SessionState` state machine that drives a conversation.
//!
//! One step at a time, executing any requested Bash tool calls for real,
//! until the model stops asking for them or the step budget runs out.

use std::time::Duration;

use eyre::Result;
use tokio::sync::oneshot;

use crate::bash_tool::{BashOutput, run_bash_command};
use crate::config::Config;
use crate::local_model::{DangerVerdict, LocalModel};
use crate::model_client::{Context, ContextMessage, ModelClient, ToolCallRequest, Usage};

/// The reason shown to the user when Danger Classification itself couldn't
/// produce a verdict — fail-closed (see `CONTEXT.md`'s Approval): treated
/// exactly like an explicit `dangerous` verdict.
const CLASSIFICATION_FAILED_REASON: &str =
    "the local model couldn't classify this command, so it's treated as dangerous";

/// The reason shown when there is no Local Model to classify with at all
/// (Degraded Mode — see `CONTEXT.md`).
const DEGRADED_MODE_REASON: &str =
    "the local model is unavailable this session, so every command requires approval";

/// The captured result of running one requested Bash tool call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolResult {
    pub tool_call_id: String,
    pub content: String,
    /// See [`ContextMessage::ToolResult`]'s `raw` field doc.
    pub raw: Option<String>,
}

/// The phase currently driving [`Session`] forward. See `CONTEXT.md`'s Loop
/// phases (Awaiting Input, Calling Model, Executing Tools, Awaiting
/// Approval).
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
    /// Paused on one Tool Call that Danger Classification marked
    /// `dangerous`, waiting on the user's Approval (see `CONTEXT.md`).
    AwaitingApproval {
        step: u32,
        tool_call: ToolCallRequest,
        reason: String,
    },
    /// Running Tool Output Compression on one just-executed Tool Call's
    /// output, before its result is appended to the Context.
    CompressingOutput {
        step: u32,
        tool_call_id: String,
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
    /// `tool_call` was classified `dangerous` (or Danger Classification
    /// itself failed, or there's no Local Model at all — see
    /// `CONTEXT.md`'s Approval) and needs the user's y/n Approval before it
    /// runs. Send exactly one `bool` into `respond` — `true` to run it,
    /// `false` to deny it — to resume the Turn.
    ApprovalNeeded {
        tool_call: ToolCallRequest,
        reason: String,
        respond: oneshot::Sender<bool>,
    },
}

/// Formats just the metadata of one Bash tool call's result — never
/// touched by Tool Output Compression, unlike the captured stdout/stderr
/// text (see `format_captured_text`).
fn format_metadata(output: &BashOutput) -> String {
    let exit_code = output
        .exit_code
        .map_or_else(|| "none".to_string(), |code| code.to_string());
    format!("exit_code: {exit_code}\ntimed_out: {}", output.timed_out)
}

/// Formats `stdout`/`stderr` for display or compression input. Sections are
/// omitted entirely when empty, rather than shown as empty-bodied headers,
/// so a command that only writes to one stream doesn't pad the transcript
/// (and the model's context) with a section that says nothing.
fn format_captured_text(stdout: &str, stderr: &str) -> String {
    let mut sections = Vec::new();
    if !stdout.is_empty() {
        sections.push(format!("stdout:\n{stdout}"));
    }
    if !stderr.is_empty() {
        sections.push(format!("stderr:\n{stderr}"));
    }
    sections.join("\n")
}

/// Joins `metadata` and `text` the way a Tool Result's `content` is shown:
/// `text` is omitted entirely when empty, rather than left as a trailing
/// blank section.
fn join_metadata_and_text(metadata: &str, text: &str) -> String {
    if text.is_empty() {
        metadata.to_string()
    } else {
        format!("{metadata}\n{text}")
    }
}

/// Decides what a Tool Result's final text and (if distinct) display-only
/// raw counterpart should be, from `raw_text` (must be non-empty — callers
/// skip compression entirely for empty output) and Tool Output
/// Compression's outcome. Falls back to `raw_text` unchanged (`None` raw,
/// since there's nothing distinct to show) whenever `compressed`: errored,
/// came back empty (treated the same as a failure — an empty result would
/// otherwise silently drop the command's entire captured output from what
/// the model sees), or is identical to the input.
fn resolve_compressed_text(
    raw_text: String,
    compressed: eyre::Result<String>,
) -> (String, Option<String>) {
    match compressed {
        Ok(compressed) if !compressed.is_empty() && compressed != raw_text => {
            (compressed, Some(raw_text))
        }
        Ok(_) | Err(_) => (raw_text, None),
    }
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
        local_model: Option<&LocalModel>,
        on_event: impl FnMut(TurnEvent) + Send,
    ) -> Result<()> {
        self.push_user_message(user_message);
        self.run_turn(client, config, local_model, on_event).await
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
        local_model: Option<&LocalModel>,
        on_event: impl FnMut(TurnEvent) + Send,
    ) -> Result<()> {
        let result = self.run_steps(client, config, local_model, on_event).await;
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
        local_model: Option<&LocalModel>,
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

            let step = self.step_count;
            let mut pending = response.tool_calls;
            let mut results: Vec<ToolResult> = Vec::new();

            while !pending.is_empty() {
                let executing_tools = SessionState::ExecutingTools {
                    step,
                    pending: pending.clone(),
                    results: results.clone(),
                };
                self.state = executing_tools.clone();
                on_event(TurnEvent::StateChanged(executing_tools));

                let tool_call = pending.remove(0);

                let approved = resolve_approval(local_model, &tool_call, step, &mut on_event).await;

                let result = if approved {
                    run_and_compress_tool_call(&tool_call, config, local_model, step, &mut on_event)
                        .await
                } else {
                    ToolResult {
                        tool_call_id: tool_call.id.clone(),
                        content: "denied by user: command not run".to_string(),
                        raw: None,
                    }
                };

                let tool_result_message = ContextMessage::ToolResult {
                    tool_call_id: result.tool_call_id.clone(),
                    content: result.content.clone(),
                    raw: result.raw.clone(),
                };
                self.context.push(tool_result_message.clone());
                on_event(TurnEvent::MessageAppended(tool_result_message));
                self.tool_call_count = self.tool_call_count.saturating_add(1);
                on_event(self.counters_event());
                results.push(result);
            }
        }

        Ok(())
    }
}

/// Resolves whether `tool_call` may run: `true` to run it, `false` if the
/// user denied it. Classifies via `local_model` — or fails closed with
/// [`DEGRADED_MODE_REASON`] if there is none (Degraded Mode, see
/// `CONTEXT.md`) — and, for anything not classified `safe`, transitions to
/// [`SessionState::AwaitingApproval`], then emits
/// [`TurnEvent::ApprovalNeeded`] and awaits the user's y/n response. A
/// dropped response channel (e.g. the app quit while paused) resolves to
/// denied rather than panicking.
async fn resolve_approval(
    local_model: Option<&LocalModel>,
    tool_call: &ToolCallRequest,
    step: u32,
    on_event: &mut (impl FnMut(TurnEvent) + Send),
) -> bool {
    let reason = match local_model {
        Some(model) => match model.classify_danger(&tool_call.command).await {
            Ok(DangerVerdict::Safe) => return true,
            Ok(DangerVerdict::Dangerous { reason }) => reason,
            Err(_) => CLASSIFICATION_FAILED_REASON.to_string(),
        },
        None => DEGRADED_MODE_REASON.to_string(),
    };

    on_event(TurnEvent::StateChanged(SessionState::AwaitingApproval {
        step,
        tool_call: tool_call.clone(),
        reason: reason.clone(),
    }));
    let (respond, receiver) = oneshot::channel();
    on_event(TurnEvent::ApprovalNeeded {
        tool_call: tool_call.clone(),
        reason,
        respond,
    });
    receiver.await.unwrap_or(false)
}

/// Runs `tool_call` for real, then compresses its captured output (see
/// `CONTEXT.md`'s Tool Output Compression) via `local_model` if there's
/// output to compress and a Local Model to compress it with — falling back
/// to the raw (capped) output on any compression failure, and skipping
/// compression entirely for empty output or in Degraded Mode.
///
/// Every `tool_call` from the assistant message already pushed to the
/// Context needs a matching [`ToolResult`], or the next request sends
/// `OpenRouter` an unanswered tool call and gets rejected — so a failure here
/// becomes the result's content instead of propagating an error.
async fn run_and_compress_tool_call(
    tool_call: &ToolCallRequest,
    config: &Config,
    local_model: Option<&LocalModel>,
    step: u32,
    on_event: &mut (impl FnMut(TurnEvent) + Send),
) -> ToolResult {
    let command = tool_call.command.clone();
    let timeout = Duration::from_secs(config.tool_timeout_seconds);
    let truncate_limit = config.tool_output_truncate_chars;
    let output = match tokio::task::spawn_blocking(move || {
        run_bash_command(&command, timeout, truncate_limit)
    })
    .await
    {
        Ok(Ok(output)) => output,
        Ok(Err(error)) => {
            return ToolResult {
                tool_call_id: tool_call.id.clone(),
                content: format!("error running command: {error}"),
                raw: None,
            };
        }
        Err(join_error) => {
            return ToolResult {
                tool_call_id: tool_call.id.clone(),
                content: format!("bash tool task panicked: {join_error}"),
                raw: None,
            };
        }
    };

    let metadata = format_metadata(&output);
    let raw_text = format_captured_text(&output.stdout, &output.stderr);

    let (final_text, raw_for_display) = if raw_text.is_empty() {
        (raw_text, None)
    } else if let Some(model) = local_model {
        on_event(TurnEvent::StateChanged(SessionState::CompressingOutput {
            step,
            tool_call_id: tool_call.id.clone(),
        }));
        let compressed = model.compress_output(&raw_text).await;
        resolve_compressed_text(raw_text, compressed)
    } else {
        (raw_text, None)
    };

    ToolResult {
        tool_call_id: tool_call.id.clone(),
        content: join_metadata_and_text(&metadata, &final_text),
        raw: raw_for_display,
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
            .run_turn(&client, &config, None, |_event: TurnEvent| {})
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
    fn format_metadata_reports_exit_code_and_timed_out() {
        assert_eq!(
            format_metadata(&bash_output("", "")),
            "exit_code: 0\ntimed_out: false"
        );
    }

    #[test]
    fn format_captured_text_omits_stdout_and_stderr_when_both_are_empty() {
        assert_eq!(format_captured_text("", ""), "");
    }

    #[test]
    fn format_captured_text_includes_stdout_only_when_stderr_is_empty() {
        let formatted = format_captured_text("hello\n", "");

        assert_eq!(formatted, "stdout:\nhello\n");
        assert!(!formatted.contains("stderr"));
    }

    #[test]
    fn format_captured_text_includes_stderr_only_when_stdout_is_empty() {
        let formatted = format_captured_text("", "oops\n");

        assert_eq!(formatted, "stderr:\noops\n");
        assert!(!formatted.contains("stdout"));
    }

    #[test]
    fn format_captured_text_includes_both_when_both_are_present() {
        let formatted = format_captured_text("hello\n", "oops\n");

        assert_eq!(formatted, "stdout:\nhello\n\nstderr:\noops\n");
    }

    #[test]
    fn join_metadata_and_text_omits_a_trailing_blank_section_for_empty_text() {
        assert_eq!(join_metadata_and_text("exit_code: 0", ""), "exit_code: 0");
    }

    #[test]
    fn join_metadata_and_text_appends_non_empty_text_on_its_own_line() {
        assert_eq!(
            join_metadata_and_text("exit_code: 0", "stdout:\nhi\n"),
            "exit_code: 0\nstdout:\nhi\n"
        );
    }

    #[test]
    fn resolve_compressed_text_uses_the_compressed_result_when_it_differs() {
        let (text, raw) =
            resolve_compressed_text("a lot of raw text".to_string(), Ok("short".to_string()));

        assert_eq!(text, "short");
        assert_eq!(raw, Some("a lot of raw text".to_string()));
    }

    #[test]
    fn resolve_compressed_text_falls_back_to_raw_on_error() {
        let (text, raw) = resolve_compressed_text("raw text".to_string(), Err(eyre::eyre!("boom")));

        assert_eq!(text, "raw text");
        assert_eq!(raw, None);
    }

    #[test]
    fn resolve_compressed_text_falls_back_to_raw_when_compression_is_identical() {
        let (text, raw) =
            resolve_compressed_text("same text".to_string(), Ok("same text".to_string()));

        assert_eq!(text, "same text");
        assert_eq!(raw, None);
    }

    #[test]
    fn resolve_compressed_text_falls_back_to_raw_when_compression_is_empty() {
        // A non-empty `raw_text` guarantees there is something to say; an
        // empty compressed result must not silently discard it.
        let (text, raw) = resolve_compressed_text("raw text".to_string(), Ok(String::new()));

        assert_eq!(text, "raw text");
        assert_eq!(raw, None);
    }

    fn test_config(max_steps: u32) -> Config {
        Config {
            model: "test/model".to_string(),
            max_steps,
            system_prompt: "You are a test assistant.".to_string(),
            tool_timeout_seconds: 5,
            tool_output_truncate_chars: 1000,
            local_model_timeout_seconds: 5,
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

    /// Wraps `on_event` so any [`TurnEvent::ApprovalNeeded`] is immediately
    /// approved and not forwarded on — for tests exercising loop mechanics
    /// (step/tool-call counting, message ordering) that aren't about the
    /// Approval gate itself, so they don't need a real classification call
    /// to avoid hanging on an unanswered prompt.
    fn auto_approve_tool_calls(
        mut on_event: impl FnMut(TurnEvent) + Send,
    ) -> impl FnMut(TurnEvent) + Send {
        move |event: TurnEvent| match event {
            TurnEvent::ApprovalNeeded { respond, .. } => {
                let _unused = respond.send(true);
            }
            other => on_event(other),
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
                None,
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
                None,
                auto_approve_tool_calls(|_event: TurnEvent| {}),
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
                None,
                auto_approve_tool_calls(|_event: TurnEvent| {}),
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
                None,
                auto_approve_tool_calls(|_event: TurnEvent| {}),
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
                None,
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
                None,
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
                None,
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
            .send_user_message(
                "run something".to_string(),
                &client,
                &config,
                None,
                auto_approve_tool_calls(|event| {
                    if let TurnEvent::Text(text) = event {
                        deltas.lock().unwrap().push(text);
                    }
                }),
            )
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
            .send_user_message(
                "run something".to_string(),
                &client,
                &config,
                None,
                auto_approve_tool_calls(|event| {
                    if let TurnEvent::StateChanged(state) = event {
                        states.lock().unwrap().push(state);
                    }
                }),
            )
            .await
            .unwrap();

        let call_1 = ToolCallRequest {
            id: "call_1".to_string(),
            command: "echo one".to_string(),
        };
        // No Local Model was given (Degraded Mode), so Danger Classification
        // fails closed and every tool call pauses at Awaiting Approval, even
        // though it's auto-approved here — see `auto_approve_tool_calls`.
        assert_eq!(
            states.into_inner().unwrap(),
            vec![
                SessionState::CallingModel { step: 1 },
                SessionState::ExecutingTools {
                    step: 1,
                    pending: vec![call_1.clone()],
                    results: Vec::new(),
                },
                SessionState::AwaitingApproval {
                    step: 1,
                    tool_call: call_1,
                    reason: DEGRADED_MODE_REASON.to_string(),
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
            .send_user_message(
                "run something".to_string(),
                &client,
                &config,
                None,
                auto_approve_tool_calls(|event| {
                    if let TurnEvent::MessageAppended(message) = event {
                        messages.lock().unwrap().push(message);
                    }
                }),
            )
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
            .send_user_message(
                "run something".to_string(),
                &client,
                &config,
                None,
                auto_approve_tool_calls(|event| {
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
                }),
            )
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

    #[tokio::test]
    async fn degraded_mode_fails_closed_with_a_reason_even_without_a_local_model() {
        let client = FakeModelClient::new(vec![
            tool_call_response(&[("call_1", "echo one")]),
            text_response("done"),
        ]);
        let config = test_config(10);
        let mut session = Session::new("system prompt");
        let reasons = Mutex::new(Vec::new());

        session
            .send_user_message(
                "run something".to_string(),
                &client,
                &config,
                None,
                |event| {
                    if let TurnEvent::ApprovalNeeded {
                        reason, respond, ..
                    } = event
                    {
                        reasons.lock().unwrap().push(reason);
                        let _unused = respond.send(true);
                    }
                },
            )
            .await
            .unwrap();

        assert_eq!(
            reasons.into_inner().unwrap(),
            vec![DEGRADED_MODE_REASON.to_string()]
        );
    }

    #[tokio::test]
    async fn a_dangerous_command_pauses_for_approval_and_runs_once_approved() {
        let temp_dir = std::env::temp_dir().join(format!(
            "tib-session-test-approve-{}-{}",
            std::process::id(),
            line!()
        ));
        std::fs::create_dir_all(&temp_dir).unwrap();
        let command = format!("rm -rf {}", temp_dir.display());
        let client = FakeModelClient::new(vec![
            tool_call_response(&[("call_1", &command)]),
            text_response("done"),
        ]);
        let config = test_config(10);
        let mut session = Session::new("system prompt");
        let model = crate::local_model::tests::test_model().await;
        let approval_seen = Mutex::new(false);

        session
            .send_user_message(
                "clean up".to_string(),
                &client,
                &config,
                Some(model),
                |event| {
                    if let TurnEvent::ApprovalNeeded { respond, .. } = event {
                        *approval_seen.lock().unwrap() = true;
                        let _unused = respond.send(true);
                    }
                },
            )
            .await
            .unwrap();

        assert!(*approval_seen.lock().unwrap());
        assert!(!temp_dir.exists());
    }

    #[tokio::test]
    async fn a_denied_command_never_runs_and_records_a_synthetic_result() {
        let temp_dir = std::env::temp_dir().join(format!(
            "tib-session-test-deny-{}-{}",
            std::process::id(),
            line!()
        ));
        std::fs::create_dir_all(&temp_dir).unwrap();
        let command = format!("rm -rf {}", temp_dir.display());
        let client = FakeModelClient::new(vec![
            tool_call_response(&[("call_1", &command)]),
            text_response("done"),
        ]);
        let config = test_config(10);
        let mut session = Session::new("system prompt");
        let model = crate::local_model::tests::test_model().await;

        session
            .send_user_message(
                "clean up".to_string(),
                &client,
                &config,
                Some(model),
                |event| {
                    if let TurnEvent::ApprovalNeeded { respond, .. } = event {
                        let _unused = respond.send(false);
                    }
                },
            )
            .await
            .unwrap();

        assert!(temp_dir.exists());
        std::fs::remove_dir_all(&temp_dir).unwrap();
        let tool_result = session
            .context
            .messages
            .iter()
            .find_map(|message| match message {
                ContextMessage::ToolResult { content, .. } => Some(content.clone()),
                _ => None,
            })
            .unwrap();
        assert!(tool_result.contains("denied by user"));
    }

    #[tokio::test]
    async fn a_long_output_gets_compressed() {
        // A 1.7B classifier's verdict on any specific command isn't
        // reliably predictable (surprisingly, even `yes | head` has been
        // observed classified dangerous, on an invented rationale about
        // "downloading and extracting data"), and neither is exactly what
        // it decides to do with a given output when compressing (a
        // uniform, already-repetitive input may come back byte-identical,
        // correctly leaving `raw` as `None` per Q18's "only show a
        // duplicate section when it differs"). So rather than pin down
        // either, this test only checks what's deterministically
        // guaranteed by the code path itself: with a Local Model present,
        // an approved Tool Call still runs for real and produces a well-formed
        // result. Compression quality itself (e.g. preserving an error
        // line) is covered separately by
        // `local_model::tests::compress_output_preserves_an_error_line_from_a_noisy_log`,
        // which calls `LocalModel::compress_output` directly.
        let command = "yes hello | head -n 500".to_string();
        let client = FakeModelClient::new(vec![
            tool_call_response(&[("call_1", &command)]),
            text_response("done"),
        ]);
        let config = test_config(10);
        let mut session = Session::new("system prompt");
        let model = crate::local_model::tests::test_model().await;

        session
            .send_user_message(
                "build it".to_string(),
                &client,
                &config,
                Some(model),
                |event| {
                    if let TurnEvent::ApprovalNeeded { respond, .. } = event {
                        let _unused = respond.send(true);
                    }
                },
            )
            .await
            .unwrap();

        assert_eq!(session.tool_call_count, 1);
        let content = session
            .context
            .messages
            .iter()
            .find_map(|message| match message {
                ContextMessage::ToolResult { content, .. } => Some(content.clone()),
                _ => None,
            })
            .unwrap();
        assert!(content.contains("exit_code: 0"));
    }
}
