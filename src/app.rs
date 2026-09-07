//! The Controller: an async event loop that forwards terminal input/tick
//! events through a channel and, depending on `SessionState`, drives the
//! `Session`/`ModelClient` loop. `Session` itself has no knowledge of the
//! terminal or the channel; the loop's own `display: Session` is a live
//! mirror kept in sync by applying each `TurnEvent` (forwarded as an
//! `AppEvent`) as it arrives — not a snapshot refreshed only once a turn
//! finishes.

use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use eyre::Result;
use ratatui::{DefaultTerminal, Frame};
use tokio::sync::{Mutex, mpsc, oneshot};
use tokio::task::{JoinError, JoinHandle, LocalSet};

use tib::config::Config;
use tib::local_model::LocalModel;
use tib::model_client::{ContextMessage, ModelClient, ToolCallRequest, Usage};
use tib::session::{Session, SessionState, TurnEvent};

use crate::view::{self, ControllerState};

const TICK_RATE: Duration = Duration::from_millis(200);

/// How many lines `PageUp`/`PageDown` scroll the transcript by, vs. one line
/// for `Up`/`Down`.
const SCROLL_PAGE: u16 = 10;

/// Mirrors `TurnEvent` one-for-one (plus the terminal/lifecycle events),
/// forwarded from the spawned turn task so `AppState` can apply each one to
/// `display` as it arrives — see the module doc comment.
enum AppEvent {
    Key(KeyEvent),
    Resize,
    Tick,
    /// The in-progress turn's current step's assistant text, accumulated so
    /// far. Replaces (rather than appends to) `streaming_text`, so a new
    /// step's first update naturally overwrites the previous step's
    /// leftover text instead of concatenating onto it.
    StreamChunk(String),
    StateChanged(SessionState),
    MessageAppended(ContextMessage),
    CountersChanged {
        step_count: u32,
        tool_call_count: u32,
        last_usage: Option<Usage>,
    },
    /// A Tool Call needs the user's y/n Approval (see `CONTEXT.md`) — mirrors
    /// [`TurnEvent::ApprovalNeeded`].
    ApprovalNeeded {
        tool_call: ToolCallRequest,
        reason: String,
        respond: oneshot::Sender<bool>,
    },
    TurnFinished(Result<()>),
    /// The Local Model finished loading (or failed to — see `CONTEXT.md`'s
    /// Degraded Mode), from the background task `spawn_local_model_load`
    /// starts at startup.
    LocalModelLoaded(Option<Arc<LocalModel>>),
}

/// One Tool Call paused at Awaiting Approval, waiting on the user's y/n
/// answer — `respond` resumes the Turn once answered (see
/// [`AppEvent::ApprovalNeeded`]).
struct PendingApproval {
    tool_call: ToolCallRequest,
    reason: String,
    respond: oneshot::Sender<bool>,
}

/// Polls terminal input on a background thread (crossterm's event reading is
/// blocking I/O) and forwards it, plus a steady tick, into `tx`.
fn spawn_input_forwarder(tx: mpsc::UnboundedSender<AppEvent>) {
    thread::spawn(move || {
        let mut last_tick = Instant::now();
        loop {
            let timeout = TICK_RATE.saturating_sub(last_tick.elapsed());
            let forwarded = if event::poll(timeout).unwrap_or(false) {
                match event::read() {
                    Ok(Event::Key(key)) if key.kind == KeyEventKind::Press => {
                        tx.send(AppEvent::Key(key))
                    }
                    Ok(Event::Resize(_, _)) => tx.send(AppEvent::Resize),
                    _ => Ok(()),
                }
            } else {
                Ok(())
            };
            if forwarded.is_err() {
                break;
            }

            if last_tick.elapsed() >= TICK_RATE {
                if tx.send(AppEvent::Tick).is_err() {
                    break;
                }
                last_tick = Instant::now();
            }
        }
    });
}

/// Adjusts a "distance scrolled up from the bottom" (see
/// `view::resolve_scroll`'s doc comment) by `delta` lines, snapping back to
/// `None` (pinned) rather than leaving a `Some(0)` once it reaches the
/// bottom again.
fn scroll_up(offset: Option<u16>, delta: u16) -> u16 {
    offset.unwrap_or(0).saturating_add(delta)
}

fn scroll_down(offset: Option<u16>, delta: u16) -> Option<u16> {
    offset.and_then(|distance| {
        let distance = distance.saturating_sub(delta);
        (distance > 0).then_some(distance)
    })
}

/// Spawns the current Turn (already started via `Session::push_user_message`)
/// as a `LocalSet` task: drives `session` forward via `client`/`config`/
/// `local_model`, forwarding each `TurnEvent` as a matching `AppEvent` on
/// `tx` as it arrives, then reports completion as an `AppEvent::TurnFinished`.
#[allow(clippy::future_not_send)] // see `run`'s doc comment
fn spawn_turn<C>(
    session: Arc<Mutex<Session>>,
    client: Arc<C>,
    config: Arc<Config>,
    local_model: Option<Arc<LocalModel>>,
    tx: mpsc::UnboundedSender<AppEvent>,
) -> JoinHandle<()>
where
    C: ModelClient + Sync + 'static,
{
    tokio::task::spawn_local(async move {
        let event_tx = tx.clone();
        let result = {
            let mut session = session.lock().await;
            session
                .run_turn(
                    client.as_ref(),
                    config.as_ref(),
                    local_model.as_deref(),
                    move |event| {
                        let app_event = match event {
                            TurnEvent::Text(text) => AppEvent::StreamChunk(text),
                            TurnEvent::StateChanged(state) => AppEvent::StateChanged(state),
                            TurnEvent::MessageAppended(message) => {
                                AppEvent::MessageAppended(message)
                            }
                            TurnEvent::CountersChanged {
                                step_count,
                                tool_call_count,
                                last_usage,
                            } => AppEvent::CountersChanged {
                                step_count,
                                tool_call_count,
                                last_usage,
                            },
                            TurnEvent::ApprovalNeeded {
                                tool_call,
                                reason,
                                respond,
                            } => AppEvent::ApprovalNeeded {
                                tool_call,
                                reason,
                                respond,
                            },
                        };
                        let _unused = event_tx.send(app_event);
                    },
                )
                .await
        };
        let _unused = tx.send(AppEvent::TurnFinished(result));
    })
}

/// Spawns the Local Model's load (see `CONTEXT.md`) as a background
/// `LocalSet` task, reporting the outcome as an `AppEvent::LocalModelLoaded`
/// once it's done — `None` on any failure (missing `HOME`, load error, or a
/// failed self-test), which the Controller treats as Degraded Mode rather
/// than a startup failure (see `CONTEXT.md`'s Degraded Mode).
fn spawn_local_model_load(timeout_seconds: u64, tx: mpsc::UnboundedSender<AppEvent>) {
    tokio::task::spawn_local(async move {
        let local_model = match std::env::var("HOME") {
            Ok(home) => LocalModel::load(&home, Duration::from_secs(timeout_seconds))
                .await
                .ok()
                .map(Arc::new),
            Err(_) => None,
        };
        let _unused = tx.send(AppEvent::LocalModelLoaded(local_model));
    });
}

/// Runs the app: connects `client`, shows the three-pane layout, and drives
/// the loop until the user quits with Ctrl+C.
///
/// Runs each turn via [`tokio::task::spawn_local`] on a [`LocalSet`] rather
/// than [`tokio::spawn`], since `ModelClient::complete`'s native `async fn`
/// in a trait doesn't guarantee its future is `Send` (a real network client
/// may hold non-`Send` internals); the whole app already runs on a
/// single-threaded runtime, so this costs nothing.
///
/// # Errors
///
/// Returns an error if drawing to `terminal` fails.
// The returned future is never actually sent across threads: `main` drives
// it on a `current_thread` runtime, and `LocalSet` itself is deliberately
// `!Send` (that's what lets `spawn_local` accept a non-`Send` `ModelClient`
// future below).
#[allow(clippy::future_not_send)]
pub async fn run<C>(
    terminal: &mut DefaultTerminal,
    config: Config,
    client: C,
    context_length: Option<u32>,
) -> Result<()>
where
    C: ModelClient + Sync + 'static,
{
    let local_set = LocalSet::new();
    local_set
        .run_until(run_on_local_set(terminal, config, client, context_length))
        .await
}

/// All of the Controller's mutable state for one run: the shared `Session`
/// plus everything in `view::ControllerState`, plus the point-in-time
/// `display` snapshot and in-flight turn handle.
struct AppState<C> {
    session: Arc<Mutex<Session>>,
    client: Arc<C>,
    config: Arc<Config>,
    tx: mpsc::UnboundedSender<AppEvent>,
    input: String,
    processing: bool,
    error_banner: Option<String>,
    /// Text streamed so far for the in-progress turn's current step, via
    /// `AppEvent::StreamChunk`; cleared once that step's message is folded
    /// into `display` via `AppEvent::MessageAppended` (or the turn ends).
    streaming_text: String,
    /// How many lines the user has manually scrolled the transcript up from
    /// the bottom; `None` means pinned to the bottom (see
    /// `view::resolve_scroll`'s doc comment).
    scroll_offset: Option<u16>,
    /// The Controller's live mirror of `session` for rendering: seeded by
    /// cloning `session` once at startup and after pushing a new user
    /// message, then kept in sync for the rest of the turn purely by
    /// applying each `AppEvent` translated from a `TurnEvent` — never by
    /// re-locking and re-cloning `session`, which the spawned turn task
    /// holds for the turn's whole duration (see `run`'s doc comment).
    display: Session,
    // The in-flight turn's task, tracked so a panic inside it (which would
    // otherwise leave `processing` stuck `true` forever, since the
    // `TurnFinished` event is only sent on normal completion) still resolves
    // the turn via its `JoinError`.
    turn_handle: Option<JoinHandle<()>>,
    /// `None` until `AppEvent::LocalModelLoaded` arrives; after that, `Some`
    /// once loaded, or permanently `None` in Degraded Mode (see
    /// `CONTEXT.md`). Cloned into each spawned turn.
    local_model: Option<Arc<LocalModel>>,
    /// True from startup until `AppEvent::LocalModelLoaded` arrives — shown
    /// as a loading indicator (see `CONTEXT.md`'s design decision not to
    /// block the UI on this).
    local_model_loading: bool,
    /// A Tool Call paused at Awaiting Approval, waiting on the user's y/n
    /// answer (see `CONTEXT.md`).
    pending_approval: Option<PendingApproval>,
}

impl<C> AppState<C>
where
    C: ModelClient + Sync + 'static,
{
    async fn new(config: Config, client: C, tx: mpsc::UnboundedSender<AppEvent>) -> Self {
        let config = Arc::new(config);
        let client = Arc::new(client);
        let session = Arc::new(Mutex::new(Session::new(config.system_prompt.clone())));
        let display = session.lock().await.clone();
        spawn_local_model_load(config.local_model_timeout_seconds, tx.clone());
        Self {
            session,
            client,
            config,
            tx,
            input: String::new(),
            processing: false,
            error_banner: None,
            streaming_text: String::new(),
            scroll_offset: None,
            display,
            turn_handle: None,
            local_model: None,
            local_model_loading: true,
            pending_approval: None,
        }
    }

    fn render(&mut self, frame: &mut Frame, context_length: Option<u32>) {
        let controller = ControllerState {
            input: &self.input,
            processing: self.processing,
            error_banner: self.error_banner.as_deref(),
            streaming_text: (!self.streaming_text.is_empty())
                .then_some(self.streaming_text.as_str()),
            scroll_offset: self.scroll_offset,
            local_model_loading: self.local_model_loading,
            local_model_degraded: !self.local_model_loading && self.local_model.is_none(),
            pending_approval: self
                .pending_approval
                .as_ref()
                .map(|pending| (&pending.tool_call, pending.reason.as_str())),
        };
        let max_scroll = view::render(
            frame,
            &self.display,
            &self.config,
            context_length,
            &controller,
        );
        // Re-clamp against this render's `max_scroll`, so a distance that
        // overshot the top (e.g. `PageUp` on a short transcript) doesn't
        // sit unclamped until enough `Down` presses catch up with it, and
        // so a distance doesn't suddenly map to a different position if the
        // transcript shrinks (e.g. on resize).
        self.scroll_offset = self.scroll_offset.and_then(|distance| {
            let distance = distance.min(max_scroll);
            (distance > 0).then_some(distance)
        });
    }

    /// Handles one key event. Returns `true` if the app should quit.
    fn handle_key(&mut self, key: KeyEvent) -> bool {
        if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
            return true;
        }

        match key.code {
            // Scrolling works regardless of `self.processing`, so the
            // transcript can be read while a turn is streaming in.
            KeyCode::Up => self.scroll_offset = Some(scroll_up(self.scroll_offset, 1)),
            KeyCode::Down => self.scroll_offset = scroll_down(self.scroll_offset, 1),
            KeyCode::PageUp => {
                self.scroll_offset = Some(scroll_up(self.scroll_offset, SCROLL_PAGE));
            }
            KeyCode::PageDown => self.scroll_offset = scroll_down(self.scroll_offset, SCROLL_PAGE),
            _ => {
                // A pending Approval takes over the keyboard until answered,
                // even ahead of the `processing` gate below (a Turn paused at
                // Awaiting Approval is still `processing`) — otherwise y/n
                // could never reach it.
                if let Some(pending) = self.pending_approval.take() {
                    match key.code {
                        KeyCode::Char('y' | 'Y') => {
                            let _unused = pending.respond.send(true);
                        }
                        KeyCode::Char('n' | 'N') => {
                            let _unused = pending.respond.send(false);
                        }
                        _ => self.pending_approval = Some(pending),
                    }
                    return false;
                }
                if self.processing {
                    return false;
                }
                self.error_banner = None;
                match key.code {
                    KeyCode::Enter if !self.input.trim().is_empty() => {
                        let message = std::mem::take(&mut self.input);
                        self.processing = true;
                        self.streaming_text.clear();
                        self.scroll_offset = None;
                        // Push and display the message immediately — it
                        // shouldn't look like it vanished while the turn
                        // it started is still in flight. Session's lock is
                        // free here: no turn is running while `!processing`.
                        if let Ok(mut session) = self.session.try_lock() {
                            session.push_user_message(message);
                            self.display = session.clone();
                        }
                        self.turn_handle = Some(spawn_turn(
                            Arc::clone(&self.session),
                            Arc::clone(&self.client),
                            Arc::clone(&self.config),
                            self.local_model.clone(),
                            self.tx.clone(),
                        ));
                    }
                    KeyCode::Backspace => {
                        self.input.pop();
                    }
                    KeyCode::Char(character) => self.input.push(character),
                    _ => {}
                }
            }
        }
        false
    }

    fn on_turn_finished(&mut self, result: Result<()>) {
        self.processing = false;
        self.streaming_text.clear();
        // Every other change this turn made was already applied live as its
        // `AppEvent` arrived; `run_turn` sets this last transition without
        // going through `on_event`, since by then the turn (and so the
        // event stream) is already over.
        self.display.state = SessionState::AwaitingUserInput;
        if let Err(error) = result {
            self.error_banner = Some(error.to_string());
        }
    }

    fn on_turn_panicked(&mut self, join_error: &JoinError) {
        self.processing = false;
        self.streaming_text.clear();
        self.display.state = SessionState::AwaitingUserInput;
        self.error_banner = Some(format!("turn task panicked: {join_error}"));
    }
}

async fn run_on_local_set<C>(
    terminal: &mut DefaultTerminal,
    config: Config,
    client: C,
    context_length: Option<u32>,
) -> Result<()>
where
    C: ModelClient + Sync + 'static,
{
    let (tx, mut rx) = mpsc::unbounded_channel::<AppEvent>();
    spawn_input_forwarder(tx.clone());
    let mut state = AppState::new(config, client, tx).await;

    loop {
        terminal.draw(|frame| state.render(frame, context_length))?;

        let has_turn = state.turn_handle.is_some();
        let join_turn = async {
            match state.turn_handle.as_mut() {
                Some(handle) => handle.await,
                None => std::future::pending().await,
            }
        };

        tokio::select! {
            biased;
            join_result = join_turn, if has_turn => {
                state.turn_handle = None;
                if let Err(join_error) = &join_result {
                    state.on_turn_panicked(join_error);
                }
            }
            event = rx.recv() => {
                let Some(event) = event else { break; };
                match event {
                    AppEvent::Tick | AppEvent::Resize => {}
                    AppEvent::Key(key) => {
                        if state.handle_key(key) {
                            break;
                        }
                    }
                    AppEvent::StreamChunk(text_so_far) => state.streaming_text = text_so_far,
                    AppEvent::StateChanged(session_state) => state.display.state = session_state,
                    AppEvent::MessageAppended(message) => {
                        state.display.context.push(message);
                        // The step's streamed preview is now this finalized
                        // message in `display.context` instead; showing
                        // both would duplicate it in the transcript.
                        state.streaming_text.clear();
                    }
                    AppEvent::CountersChanged {
                        step_count,
                        tool_call_count,
                        last_usage,
                    } => {
                        state.display.step_count = step_count;
                        state.display.tool_call_count = tool_call_count;
                        state.display.last_usage = last_usage;
                    }
                    AppEvent::ApprovalNeeded {
                        tool_call,
                        reason,
                        respond,
                    } => {
                        state.pending_approval = Some(PendingApproval {
                            tool_call,
                            reason,
                            respond,
                        });
                    }
                    AppEvent::TurnFinished(result) => state.on_turn_finished(result),
                    AppEvent::LocalModelLoaded(local_model) => {
                        state.local_model = local_model;
                        state.local_model_loading = false;
                    }
                }
            }
        }
    }

    Ok(())
}
