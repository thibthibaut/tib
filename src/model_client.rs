//! The `ModelClient` trait and the domain types that flow through it: the
//! single test seam between the loop (`Session`) and the network.

use eyre::Result;

/// A single execution of the Bash tool the model requested within a step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCallRequest {
    pub id: String,
    pub command: String,
}

/// The role a [`ContextMessage`] is tagged with, per `CONTEXT.md`'s glossary.
/// Shared by the transcript pane and the Context Visualization so both use
/// the same role→color mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// One message in a [`Context`], tagged by role per `CONTEXT.md`'s glossary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContextMessage {
    System(String),
    User(String),
    Assistant {
        text: Option<String>,
        tool_calls: Vec<ToolCallRequest>,
    },
    ToolResult {
        tool_call_id: String,
        /// What the model actually sees: metadata (`exit_code`/`timed_out`)
        /// plus either Tool Output Compression's result or, when
        /// compression didn't run or fell back, the raw captured text.
        content: String,
        /// The raw captured text, kept only when it differs from what
        /// `content` ended up holding (i.e. compression actually changed
        /// it) — display-only, per `CONTEXT.md`'s Tool Output Compression;
        /// never sent to the model. `None` when compression was skipped,
        /// failed, or the output was empty, so the transcript doesn't
        /// render a pointless duplicate of `content`.
        raw: Option<String>,
    },
}

impl ContextMessage {
    #[must_use]
    pub const fn role(&self) -> Role {
        match self {
            Self::System(_) => Role::System,
            Self::User(_) => Role::User,
            Self::Assistant { .. } => Role::Assistant,
            Self::ToolResult { .. } => Role::Tool,
        }
    }

    /// The text used to display this message in the transcript pane, and to
    /// estimate its token count (see `token_heuristic`).
    #[must_use]
    pub fn display_text(&self) -> String {
        match self {
            Self::System(text) | Self::User(text) => text.clone(),
            Self::Assistant { text, tool_calls } => {
                let mut lines = Vec::new();
                if let Some(text) = text {
                    lines.push(text.clone());
                }
                for tool_call in tool_calls {
                    lines.push(format!("→ bash: {}", tool_call.command));
                }
                lines.join("\n")
            }
            Self::ToolResult { content, .. } => content.clone(),
        }
    }

    /// The raw pre-compression text of a Tool Result, when it differs from
    /// `content` (see the field doc on [`ContextMessage::ToolResult::raw`]).
    /// `None` for every other message role, and for a Tool Result with
    /// nothing distinct to show.
    #[must_use]
    pub fn raw_tool_output(&self) -> Option<&str> {
        match self {
            Self::ToolResult { raw, .. } => raw.as_deref(),
            Self::System(_) | Self::User(_) | Self::Assistant { .. } => None,
        }
    }
}

/// The full ordered set of messages that will be sent to the model on the
/// next step.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Context {
    pub messages: Vec<ContextMessage>,
}

impl Context {
    pub fn push(&mut self, message: ContextMessage) {
        self.messages.push(message);
    }

    /// The number of Turns so far, per `CONTEXT.md`'s glossary: one per
    /// `User` message, since `Session::send_user_message` pushes exactly one
    /// onto the context at the start of every Turn.
    #[must_use]
    pub fn turn_count(&self) -> usize {
        self.messages
            .iter()
            .filter(|message| matches!(message, ContextMessage::User(_)))
            .count()
    }
}

/// Token and cost accounting for one [`ModelClient::complete`] call.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Usage {
    pub total_tokens: u32,
    pub cost: f64,
}

/// One step's full model response.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelResponse {
    pub text: Option<String>,
    pub tool_calls: Vec<ToolCallRequest>,
    pub usage: Usage,
}

/// The one seam `Session` calls through.
///
/// Implemented by `OpenRouterClient` for real use, and by a
/// `FakeModelClient` in tests. Dispatch is generic (no `dyn`, no
/// `async-trait`), so swapping fake vs. real is just what gets passed at the
/// call site.
pub trait ModelClient {
    /// Sends `context` to the model and returns its response.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying request fails or the response
    /// can't be interpreted as a [`ModelResponse`].
    async fn complete(&self, context: &Context) -> Result<ModelResponse>;

    /// Like [`complete`](Self::complete), but calls `on_update` with the
    /// assistant text accumulated so far for the current step, each time
    /// more of it arrives — a caller renders the reply incrementally by
    /// replacing its display buffer with each call's argument, not
    /// appending to it. Tool calls are never streamed piecemeal: they only
    /// appear, fully formed, in the returned [`ModelResponse`].
    ///
    /// The default implementation calls `complete` and reports its whole
    /// text in a single call, so implementers that don't support streaming
    /// (and tests using a scripted client) get correct, if non-incremental,
    /// behavior for free.
    ///
    /// # Errors
    ///
    /// Returns an error under the same conditions as `complete`.
    async fn complete_streaming(
        &self,
        context: &Context,
        mut on_update: impl FnMut(&str) + Send,
    ) -> Result<ModelResponse> {
        let response = self.complete(context).await?;
        if let Some(text) = &response.text {
            on_update(text);
        }
        Ok(response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn turn_count_is_zero_for_an_empty_context() {
        assert_eq!(Context::default().turn_count(), 0);
    }

    #[test]
    fn turn_count_counts_only_user_messages() {
        let context = Context {
            messages: vec![
                ContextMessage::System("system prompt".to_string()),
                ContextMessage::User("first".to_string()),
                ContextMessage::Assistant {
                    text: Some("reply".to_string()),
                    tool_calls: Vec::new(),
                },
                ContextMessage::ToolResult {
                    tool_call_id: "call_1".to_string(),
                    content: "output".to_string(),
                    raw: None,
                },
                ContextMessage::User("second".to_string()),
            ],
        };

        assert_eq!(context.turn_count(), 2);
    }
}
