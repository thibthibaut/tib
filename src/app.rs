//! The Controller: an async event loop that forwards terminal input/tick
//! events through a channel and, depending on `SessionState`, drives the
//! `Session`/`ModelClient` loop, feeding results back through the same
//! channel. `Session` itself has no knowledge of the terminal or the
//! channel.

use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use eyre::Result;
use ratatui::{DefaultTerminal, Frame};
use tokio::sync::{Mutex, mpsc};
use tokio::task::{JoinError, JoinHandle, LocalSet};

use tib::config::Config;
use tib::model_client::ModelClient;
use tib::session::{Session, SessionState, TurnEvent};

use crate::view::{self, ControllerState};

const TICK_RATE: Duration = Duration::from_millis(200);

/// How many lines `PageUp`/`PageDown` scroll the transcript by, vs. one line
/// for `Up`/`Down`.
const SCROLL_PAGE: u16 = 10;

enum AppEvent {
    Key(KeyEvent),
    Resize,
    Tick,
    /// The in-progress turn's current step's assistant text, accumulated so
    /// far, forwarded from `ModelClient::complete_streaming`'s `on_update`
    /// callback. Replaces (rather than appends to) `streaming_text`, so a
    /// new step's first update naturally overwrites the previous step's
    /// leftover text instead of concatenating onto it.
    StreamChunk(String),
    /// `Session::state` just changed, forwarded from `TurnEvent::StateChanged`
    /// so the Controller can reflect it live instead of only at turn
    /// boundaries.
    StateChanged(SessionState),
    TurnFinished(Result<()>),
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
/// as a `LocalSet` task: drives `session` forward via `client`/`config`,
/// forwarding each `TurnEvent` as a matching `AppEvent` on `tx` as it
/// arrives, then reports completion as an `AppEvent::TurnFinished`.
#[allow(clippy::future_not_send)] // see `run`'s doc comment
fn spawn_turn<C>(
    session: Arc<Mutex<Session>>,
    client: Arc<C>,
    config: Arc<Config>,
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
                .run_turn(client.as_ref(), config.as_ref(), move |event| {
                    let app_event = match event {
                        TurnEvent::Text(text) => AppEvent::StreamChunk(text),
                        TurnEvent::StateChanged(state) => AppEvent::StateChanged(state),
                    };
                    let _unused = event_tx.send(app_event);
                })
                .await
        };
        let _unused = tx.send(AppEvent::TurnFinished(result));
    })
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
    /// Text streamed so far for the in-progress turn, via
    /// `AppEvent::StreamChunk`; cleared once the turn folds its finalized
    /// message into `display`.
    streaming_text: String,
    /// The in-progress turn's live `SessionState`, via
    /// `AppEvent::StateChanged`; cleared once the turn finishes, since
    /// `display.state` (always `AwaitingUserInput` right after a turn) takes
    /// over at that point.
    live_state: Option<SessionState>,
    /// How many lines the user has manually scrolled the transcript up from
    /// the bottom; `None` means pinned to the bottom (see
    /// `view::resolve_scroll`'s doc comment).
    scroll_offset: Option<u16>,
    // A point-in-time copy of `Session` for rendering: refreshed only at
    // turn boundaries (start-up and `TurnFinished`), since the spawned turn
    // task holds `session`'s lock for the whole turn anyway (see `run`'s
    // doc comment) — polling it on every tick would just re-clone the same
    // unchanged snapshot.
    display: Session,
    // The in-flight turn's task, tracked so a panic inside it (which would
    // otherwise leave `processing` stuck `true` forever, since the
    // `TurnFinished` event is only sent on normal completion) still resolves
    // the turn via its `JoinError`.
    turn_handle: Option<JoinHandle<()>>,
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
        Self {
            session,
            client,
            config,
            tx,
            input: String::new(),
            processing: false,
            error_banner: None,
            streaming_text: String::new(),
            live_state: None,
            scroll_offset: None,
            display,
            turn_handle: None,
        }
    }

    fn render(&mut self, frame: &mut Frame, context_length: Option<u32>) {
        let controller = ControllerState {
            input: &self.input,
            processing: self.processing,
            error_banner: self.error_banner.as_deref(),
            streaming_text: (!self.streaming_text.is_empty())
                .then_some(self.streaming_text.as_str()),
            live_state: self.live_state.as_ref(),
            scroll_offset: self.scroll_offset,
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
                        self.live_state = Some(SessionState::CallingModel { step: 1 });
                        self.turn_handle = Some(spawn_turn(
                            Arc::clone(&self.session),
                            Arc::clone(&self.client),
                            Arc::clone(&self.config),
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
        self.live_state = None;
        if let Err(error) = result {
            self.error_banner = Some(error.to_string());
        }
        if let Ok(guard) = self.session.try_lock() {
            self.display = guard.clone();
        }
    }

    fn on_turn_panicked(&mut self, join_error: &JoinError) {
        self.processing = false;
        self.streaming_text.clear();
        self.live_state = None;
        self.error_banner = Some(format!("turn task panicked: {join_error}"));
        if let Ok(guard) = self.session.try_lock() {
            self.display = guard.clone();
        }
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
                    AppEvent::StateChanged(session_state) => state.live_state = Some(session_state),
                    AppEvent::TurnFinished(result) => state.on_turn_finished(result),
                }
            }
        }
    }

    Ok(())
}
