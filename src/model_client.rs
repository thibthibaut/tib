//! The `ModelClient` trait and the domain types that flow through it: the
//! single test seam between the loop (`Session`) and the network.

use eyre::Result;

/// A single execution of the Bash tool the model requested within a step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCallRequest {
    pub id: String,
    pub command: String,
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
        content: String,
    },
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
}
