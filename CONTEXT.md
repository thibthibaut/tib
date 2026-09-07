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
A single execution of the Bash tool that the model requested within a step. A step may trigger zero or more tool calls; the running total of tool calls is tracked separately from the step count and is never capped. Unlike the step count, this total accumulates for the whole session and is never reset between user messages. A Tool Call denied at Approval still counts here: it produced a matching Tool Result, just not a command execution.
_Avoid_: action, invocation, tool execution

**Local Model**:
The Qwen3-1.7B GGUF model embedded in the Tib binary and run in-process via mistral.rs, entirely separate from the (remote, OpenRouter-hosted) model that drives the main loop. Powers Danger Classification and Tool Output Compression only; never sees the Context and never drives a Step.
_Avoid_: "the model" (unqualified — always means the OpenRouter model per Step's definition), classifier model

**Danger Classification**:
The Local Model's binary verdict (`safe` or `dangerous`) on a pending Tool Call, returned as schema-constrained JSON with a short `reason`, before the command runs. The Local Model is the sole judge — no hardcoded pattern rules override or bypass it.
_Avoid_: dangerosity, risk score (implies a spectrum; this is binary)

**Approval**:
The user's y/n response to a Tool Call that Danger Classification marked `dangerous`, shown together with the classifier's `reason`. Approving runs the command normally; denying skips it and records a synthetic Tool Result instead ("denied by user: command not run"), and the Turn continues. Granted or denied per Tool Call, never batched across a step's several pending calls.
_Avoid_: confirmation, permission

**Tool Output Compression**:
The Local Model's rewrite of a Tool Result's stdout/stderr into a shorter form before it enters the Context, preserving `exit_code`/`timed_out` untouched (compression never touches metadata, only the captured text) and always run, regardless of output size. Replaces truncation as what shapes what the model sees; a separate, much larger memory-safety cap on raw capture still exists underneath it. On failure, falls back to the raw (capped) output rather than losing the result.
_Avoid_: summarization (implies discarding structure; this explicitly preserves errors/warnings), truncation

**Degraded Mode**:
The session-wide fallback entered if the Local Model's startup self-test inference fails. Every Tool Call is then treated as unclassified (Danger Classification always fails closed to Approval) and Tool Output Compression always falls back to raw output, for the rest of the session.
_Avoid_: safe mode, fallback mode

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
The phase where the current step's model response contained one or more tool calls, and Tib is running them before the step is complete and the loop can proceed. A Tool Call classified `dangerous` pauses this phase at Awaiting Approval before it runs.
_Avoid_: running, tool phase

**Awaiting Approval**:
The phase where Executing Tools has paused on one Tool Call that Danger Classification marked `dangerous`, waiting on the user's Approval before running it (or recording a denial) and resuming the rest of that step's pending Tool Calls. Not a Step of its own — it never calls the OpenRouter model.
_Avoid_: confirming, waiting for approval

**Compressing Output**:
The phase where Executing Tools has paused right after running one Tool Call, while Tool Output Compression rewrites its captured output before the Tool Result is appended to the Context. Skipped entirely for empty output or in Degraded Mode. Not a Step of its own.
_Avoid_: compressing, summarizing
