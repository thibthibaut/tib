# Tib

A minimal, single-session terminal chat agent that loops between calling a model over OpenRouter and running a single Bash tool, built to make the mechanics of an agent loop directly observable.

## Language

**Session**:
The single, ephemeral conversation between the user and the model for the lifetime of one run of the app. Never saved, never resumed, and only one exists at a time.
_Avoid_: conversation, chat, thread

**Context**:
The full ordered set of messages (system, user, assistant, tool) that will be sent to the model on the next step.
_Avoid_: history, prompt

**Step**:
One full round-trip to the model: send the current context, receive one response. `max_steps` caps how many steps run in response to a single user message; the count resets to zero every time the user sends a new one — it is not a whole-session total (see ADR-0006).
_Avoid_: round-trip, iteration

**Turn**:
Everything that happens in response to one user message: pushing it onto the Context and driving the loop forward, Step by Step, until it returns to Awaiting Input. A Turn spans exactly one Step when the model's first response has no tool calls, and more than one when it does.
_Avoid_: (none — this is Turn's canonical name; don't use "turn" loosely as a synonym for Step, which is the narrower, single-round-trip concept)

**Tool Call**:
A single execution of the Bash tool that the model requested within a step. A step may trigger zero or more tool calls; the running total of tool calls is tracked separately from the step count and is never capped. Unlike the step count, this total accumulates for the whole session and is never reset between user messages.
_Avoid_: action, invocation, tool execution

**Context Visualization**:
The info pane's live picture of the current context's composition: one dot per ~1000 tokens, colored by the role of the message each dot's tokens came from. Unused capacity up to the model's context window is shown as empty dots (`·`), fetched once from OpenRouter's per-model metadata at startup; if that lookup fails or the model reports no context length, only the used dots are shown.
_Avoid_: token graph, usage bar, dot grid

### Loop phases

**Awaiting Input**:
The phase where the session is idle, waiting for the user to type a message.
_Avoid_: idle, ready

**Calling Model**:
The phase where a step's request is in flight to OpenRouter. Response text may arrive incrementally as it streams in; the phase doesn't end until the step's full response — including any tool calls — is complete.
_Avoid_: waiting, thinking

**Executing Tools**:
The phase where the current step's model response contained one or more tool calls, and Tib is running them before the step is complete and the loop can proceed.
_Avoid_: running, tool phase
