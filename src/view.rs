//! The View: a pure function from `&Session`/`&Config` (plus small bits of
//! ephemeral Controller state) to one ratatui frame. No mutation, no async,
//! no I/O.

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};

use tib::config::Config;
use tib::model_client::Role;
use tib::session::{Session, SessionState};
use tib::token_heuristic::scaled_message_tokens;

/// The shared role→color mapping used by both the transcript pane and the
/// context dot-grid, per `CONTEXT.md`'s Context Visualization glossary entry.
const fn role_color(role: Role) -> Color {
    match role {
        Role::System => Color::Yellow,
        Role::User => Color::Green,
        Role::Tool => Color::Red,
        Role::Assistant => Color::Blue,
    }
}

fn transcript_lines(session: &Session) -> Vec<Line<'static>> {
    session
        .context
        .messages
        .iter()
        .flat_map(|message| {
            let style = Style::default().fg(role_color(message.role()));
            message
                .display_text()
                .lines()
                .map(|line| Line::styled(line.to_string(), style))
                .collect::<Vec<_>>()
        })
        .collect()
}

fn dot_grid_spans(session: &Session) -> Vec<Span<'static>> {
    let Some(usage) = session.last_usage else {
        return Vec::new();
    };
    scaled_message_tokens(&session.context, usage.total_tokens)
        .into_iter()
        .flat_map(|message_tokens| {
            let dots = message_tokens.tokens.checked_div(1000).unwrap_or(0);
            let style = Style::default().fg(role_color(message_tokens.role));
            std::iter::repeat_n(Span::styled("●", style), usize::try_from(dots).unwrap_or(0))
        })
        .collect()
}

const fn state_label(state: &SessionState) -> &'static str {
    match state {
        SessionState::AwaitingUserInput => "awaiting input",
        SessionState::CallingModel { .. } => "calling model…",
        SessionState::ExecutingTools { .. } => "executing tools…",
    }
}

fn info_lines(session: &Session, config: &Config, processing: bool) -> Vec<Line<'static>> {
    let cost = session
        .last_usage
        .map_or_else(|| "n/a".to_string(), |usage| format!("${:.4}", usage.cost));
    let total_tokens = session.last_usage.map_or(0, |usage| usage.total_tokens);

    let mut lines = vec![
        Line::from(format!("model: {}", config.model)),
        Line::from(format!("step: {}/{}", session.step_count, config.max_steps)),
        Line::from(format!("tool calls: {}", session.tool_call_count)),
        Line::from(format!("last call cost: {cost}")),
        Line::from(if processing {
            // `session` here is a point-in-time snapshot that can't reflect
            // a turn in progress (the loop only refreshes it at turn
            // boundaries), so this deliberately doesn't read `session.state`.
            "state: working…".to_string()
        } else {
            format!("state: {}", state_label(&session.state))
        }),
        Line::from(""),
        Line::from(format!("context (~{total_tokens} tokens):")),
    ];
    lines.push(Line::from(dot_grid_spans(session)));
    lines
}

fn layout(area: Rect) -> (Rect, Rect, Rect) {
    let columns = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(70), Constraint::Percentage(30)])
        .split(area);
    let left_column = columns.first().copied().unwrap_or_default();
    let info_area = columns.get(1).copied().unwrap_or_default();

    let left_rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(1), Constraint::Length(3)])
        .split(left_column);
    let transcript_area = left_rows.first().copied().unwrap_or_default();
    let input_area = left_rows.get(1).copied().unwrap_or_default();

    (transcript_area, input_area, info_area)
}

/// Renders one frame from `session`/`config` and the small amount of
/// ephemeral Controller state (`input` box contents, whether a turn is in
/// flight, and the last turn's error, if any).
pub fn render(
    frame: &mut Frame,
    session: &Session,
    config: &Config,
    input: &str,
    processing: bool,
    error_banner: Option<&str>,
) {
    let (transcript_area, input_area, info_area) = layout(frame.area());

    let lines = transcript_lines(session);
    let line_count = u16::try_from(lines.len()).unwrap_or(u16::MAX);
    let transcript_height = transcript_area.height.saturating_sub(2);
    // Approximates scroll-to-bottom using unwrapped line count (a message
    // that wraps to multiple rows is undercounted here), which is close
    // enough to keep the latest messages in view without depending on
    // ratatui's unstable line-counting API.
    let scroll = line_count.saturating_sub(transcript_height);
    let transcript = Paragraph::new(lines)
        .block(Block::default().borders(Borders::ALL).title("Transcript"))
        .wrap(Wrap { trim: false })
        .scroll((scroll, 0));
    frame.render_widget(transcript, transcript_area);

    let input_title = if processing {
        "Message (sending…)"
    } else {
        "Message"
    };
    let input_style = if error_banner.is_some() {
        Style::default().fg(Color::Red)
    } else {
        Style::default()
    };
    // The input box has no cursor movement (only append/backspace at the
    // end), so the cursor is always at the end of `input`: scrolling to show
    // just the tail that fits keeps it visible instead of clipped off-screen.
    let input_inner_width = usize::from(input_area.width.saturating_sub(2)).max(1);
    let visible_chars = input.chars().count().min(input_inner_width);
    let visible_input: String = input
        .chars()
        .skip(input.chars().count().saturating_sub(visible_chars))
        .collect();
    let input_text = error_banner.map_or_else(|| visible_input, |error| format!("error: {error}"));
    let input_paragraph = Paragraph::new(input_text)
        .style(input_style)
        .block(Block::default().borders(Borders::ALL).title(input_title));
    frame.render_widget(input_paragraph, input_area);

    let info = Paragraph::new(info_lines(session, config, processing))
        .block(Block::default().borders(Borders::ALL).title("Info"))
        .wrap(Wrap { trim: false });
    frame.render_widget(info, info_area);

    if !processing && error_banner.is_none() {
        frame.set_cursor_position((
            input_area
                .x
                .saturating_add(1)
                .saturating_add(u16::try_from(visible_chars).unwrap_or(u16::MAX)),
            input_area.y.saturating_add(1),
        ));
    }
}
