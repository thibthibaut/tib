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
use ratatui::DefaultTerminal;
use tokio::sync::{Mutex, mpsc};
use tokio::task::{JoinHandle, LocalSet};

use tib::config::Config;
use tib::model_client::ModelClient;
use tib::session::Session;

use crate::view;

const TICK_RATE: Duration = Duration::from_millis(200);

enum AppEvent {
    Key(KeyEvent),
    Resize,
    Tick,
    TurnFinished(Result<()>),
}

/// Polls terminal input on a background thread (crossterm's event reading is
/// blocking I/O) and forwards it, plus a steady tick, into `tx`.
fn spawn_input_forwarder(tx: mpsc::Sender<AppEvent>) {
    thread::spawn(move || {
        let mut last_tick = Instant::now();
        loop {
            let timeout = TICK_RATE.saturating_sub(last_tick.elapsed());
            let forwarded = if event::poll(timeout).unwrap_or(false) {
                match event::read() {
                    Ok(Event::Key(key)) if key.kind == KeyEventKind::Press => {
                        tx.blocking_send(AppEvent::Key(key))
                    }
                    Ok(Event::Resize(_, _)) => tx.blocking_send(AppEvent::Resize),
                    _ => Ok(()),
                }
            } else {
                Ok(())
            };
            if forwarded.is_err() {
                break;
            }

            if last_tick.elapsed() >= TICK_RATE {
                if tx.blocking_send(AppEvent::Tick).is_err() {
                    break;
                }
                last_tick = Instant::now();
            }
        }
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
pub async fn run<C>(terminal: &mut DefaultTerminal, config: Config, client: C) -> Result<()>
where
    C: ModelClient + Sync + 'static,
{
    let local_set = LocalSet::new();
    local_set
        .run_until(run_on_local_set(terminal, config, client))
        .await
}

async fn run_on_local_set<C>(
    terminal: &mut DefaultTerminal,
    config: Config,
    client: C,
) -> Result<()>
where
    C: ModelClient + Sync + 'static,
{
    let config = Arc::new(config);
    let client = Arc::new(client);
    let session = Arc::new(Mutex::new(Session::new(config.system_prompt.clone())));

    let (tx, mut rx) = mpsc::channel::<AppEvent>(100);
    spawn_input_forwarder(tx.clone());

    let mut input = String::new();
    let mut processing = false;
    let mut error_banner: Option<String> = None;
    // A point-in-time copy of `Session` for rendering: refreshed only at
    // turn boundaries (start-up and `TurnFinished`), since the spawned turn
    // task holds `session`'s lock for the whole turn anyway (see `run`'s
    // doc comment) — polling it on every tick would just re-clone the same
    // unchanged snapshot.
    let mut display = session.lock().await.clone();
    // The in-flight turn's task, tracked so a panic inside it (which would
    // otherwise leave `processing` stuck `true` forever, since the
    // `TurnFinished` event is only sent on normal completion) still resolves
    // the turn via its `JoinError`.
    let mut turn_handle: Option<JoinHandle<()>> = None;

    loop {
        terminal.draw(|frame| {
            view::render(
                frame,
                &display,
                &config,
                &input,
                processing,
                error_banner.as_deref(),
            );
        })?;

        let has_turn = turn_handle.is_some();
        let join_turn = async {
            match turn_handle.as_mut() {
                Some(handle) => handle.await,
                None => std::future::pending().await,
            }
        };

        tokio::select! {
            biased;
            join_result = join_turn, if has_turn => {
                turn_handle = None;
                if let Err(join_error) = join_result {
                    processing = false;
                    error_banner = Some(format!("turn task panicked: {join_error}"));
                    if let Ok(guard) = session.try_lock() {
                        display = guard.clone();
                    }
                }
            }
            event = rx.recv() => {
                let Some(event) = event else { break; };
                match event {
                    AppEvent::Tick | AppEvent::Resize => {}
                    AppEvent::Key(key) => {
                        if key.code == KeyCode::Char('c')
                            && key.modifiers.contains(KeyModifiers::CONTROL)
                        {
                            break;
                        }
                        if processing {
                            continue;
                        }
                        error_banner = None;
                        match key.code {
                            KeyCode::Enter if !input.trim().is_empty() => {
                                let message = std::mem::take(&mut input);
                                processing = true;

                                let session = Arc::clone(&session);
                                let client = Arc::clone(&client);
                                let config = Arc::clone(&config);
                                let tx = tx.clone();
                                turn_handle = Some(tokio::task::spawn_local(async move {
                                    let result = {
                                        let mut session = session.lock().await;
                                        session
                                            .send_user_message(
                                                message,
                                                client.as_ref(),
                                                config.as_ref(),
                                            )
                                            .await
                                    };
                                    let _unused = tx.send(AppEvent::TurnFinished(result)).await;
                                }));
                            }
                            KeyCode::Backspace => {
                                input.pop();
                            }
                            KeyCode::Char(character) => input.push(character),
                            _ => {}
                        }
                    }
                    AppEvent::TurnFinished(result) => {
                        processing = false;
                        if let Err(error) = result {
                            error_banner = Some(error.to_string());
                        }
                        if let Ok(guard) = session.try_lock() {
                            display = guard.clone();
                        }
                    }
                }
            }
        }
    }

    Ok(())
}
