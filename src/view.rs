//! The View: a pure function from `&Session`/`&Config` (plus small bits of
//! ephemeral Controller state) to one ratatui frame. No mutation, no async,
//! no I/O.

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Block, Borders, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState, Wrap,
};

use tib::config::Config;
use tib::model_client::Role;
use tib::session::{Session, SessionState};
use tib::token_heuristic::{free_dots, scaled_message_tokens};

/// A colored marker drawn at the start of every line of a message, so
/// adjacent messages read as distinct sections without recoloring their text
/// (which would hurt legibility on long tool output).
const GUTTER: &str = "▎";

/// The glyph for one dot's worth (~1000 tokens, see `CONTEXT.md`'s Context
/// Visualization entry) of remaining, unused context budget.
const FREE_DOT: &str = "·";

/// The color remaining, unused context budget is drawn in — deliberately not
/// one of `role_color`'s colors, since free dots belong to no role.
const FREE_DOT_COLOR: Color = Color::DarkGray;

/// Ephemeral state the Controller owns that isn't part of `Session`: the
/// input box contents, whether a turn is in flight, the last turn's error
/// (if any), text streamed so far for the in-progress turn (not yet folded
/// into `Session`), and how far the transcript is manually scrolled up from
/// the bottom (`None` means pinned to the bottom).
pub struct ControllerState<'a> {
    pub input: &'a str,
    pub processing: bool,
    pub error_banner: Option<&'a str>,
    pub streaming_text: Option<&'a str>,
    pub scroll_offset: Option<u16>,
}

/// The shared role→color mapping used by both the transcript pane and the
/// Context Visualization, per `CONTEXT.md`'s glossary entry for it.
const fn role_color(role: Role) -> Color {
    match role {
        Role::System => Color::Yellow,
        Role::User => Color::Green,
        Role::Tool => Color::Red,
        Role::Assistant => Color::Blue,
    }
}

/// One message's lines, each prefixed with a role-colored gutter marker; the
/// message text itself stays in the terminal's default color.
fn message_lines(role: Role, text: &str) -> Vec<Line<'static>> {
    let gutter_style = Style::default().fg(role_color(role));
    text.lines()
        .map(|line| {
            Line::from(vec![
                Span::styled(GUTTER, gutter_style),
                Span::raw(line.to_string()),
            ])
        })
        .collect()
}

/// The transcript's lines: one blank separator line between each message
/// (including a trailing in-progress `streaming_text`, if any) so sections
/// are visually distinct.
fn transcript_lines(session: &Session, streaming_text: Option<&str>) -> Vec<Line<'static>> {
    let mut lines: Vec<Line<'static>> = Vec::new();

    for message in &session.context.messages {
        if !lines.is_empty() {
            lines.push(Line::default());
        }
        lines.extend(message_lines(message.role(), &message.display_text()));
    }

    if let Some(text) = streaming_text.filter(|text| !text.is_empty()) {
        if !lines.is_empty() {
            lines.push(Line::default());
        }
        lines.extend(message_lines(Role::Assistant, text));
    }

    lines
}

/// Resolves the transcript's scroll position from `offset` (`None` pins to
/// the bottom; `Some(distance)` is how many lines up from the bottom the
/// user has manually scrolled) against `max_scroll`, the furthest position
/// currently scrollable — clamping so a manual scroll can never go further
/// up than the top, even after the transcript shrinks (e.g. on resize).
fn resolve_scroll(offset: Option<u16>, max_scroll: u16) -> u16 {
    offset.map_or(max_scroll, |distance| {
        max_scroll.saturating_sub(distance.min(max_scroll))
    })
}

fn context_visualization_spans(
    session: &Session,
    context_length: Option<u32>,
) -> Vec<Span<'static>> {
    let Some(usage) = session.last_usage else {
        return Vec::new();
    };
    let mut spans: Vec<Span<'static>> = scaled_message_tokens(&session.context, usage.total_tokens)
        .into_iter()
        .flat_map(|message_tokens| {
            let dots = message_tokens.tokens.checked_div(1000).unwrap_or(0);
            let style = Style::default().fg(role_color(message_tokens.role));
            std::iter::repeat_n(Span::styled("●", style), usize::try_from(dots).unwrap_or(0))
        })
        .collect();

    let free = free_dots(context_length, usage.total_tokens);
    let free_style = Style::default().fg(FREE_DOT_COLOR);
    spans.extend(std::iter::repeat_n(
        Span::styled(FREE_DOT, free_style),
        usize::try_from(free).unwrap_or(0),
    ));

    spans
}

/// The role→color legend shown at the bottom of the info pane, generated
/// directly from `role_color` so it can never drift out of sync with the
/// dots or gutter markers it explains.
fn legend_lines() -> Vec<Line<'static>> {
    [
        (Role::System, "system"),
        (Role::User, "user"),
        (Role::Assistant, "assistant"),
        (Role::Tool, "tool"),
    ]
    .into_iter()
    .map(|(role, label)| {
        Line::from(vec![
            Span::styled("●", Style::default().fg(role_color(role))),
            Span::raw(format!(" {label}")),
        ])
    })
    .chain(std::iter::once(Line::from(vec![
        Span::styled(FREE_DOT, Style::default().fg(FREE_DOT_COLOR)),
        Span::raw(" free"),
    ])))
    .collect()
}

const fn state_label(state: &SessionState) -> &'static str {
    match state {
        SessionState::AwaitingUserInput => "awaiting input",
        SessionState::CallingModel { .. } => "calling model…",
        SessionState::ExecutingTools { .. } => "executing tools…",
    }
}

fn info_lines(
    session: &Session,
    config: &Config,
    processing: bool,
    context_length: Option<u32>,
) -> Vec<Line<'static>> {
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
    lines.push(Line::from(context_visualization_spans(
        session,
        context_length,
    )));
    lines.push(Line::default());
    lines.extend(legend_lines());
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

/// Renders one frame from `session`/`config`, `context_length` (the
/// model's context window, if known), and the ephemeral Controller state in
/// `controller`. Returns the transcript's `max_scroll` (the furthest
/// position currently scrollable) as rendered, so the caller can keep a
/// manual `scroll_offset` clamped to it between renders — otherwise a
/// distance recorded while scrolled far up could overshoot once the
/// transcript shrinks (e.g. after a resize).
pub fn render(
    frame: &mut Frame,
    session: &Session,
    config: &Config,
    context_length: Option<u32>,
    controller: &ControllerState<'_>,
) -> u16 {
    let (transcript_area, input_area, info_area) = layout(frame.area());

    let lines = transcript_lines(session, controller.streaming_text);
    let line_count = u16::try_from(lines.len()).unwrap_or(u16::MAX);
    let transcript_height = transcript_area.height.saturating_sub(2);
    // Approximates scroll-to-bottom using unwrapped line count (a message
    // that wraps to multiple rows is undercounted here), which is close
    // enough to keep the latest messages in view without depending on
    // ratatui's unstable line-counting API.
    let max_scroll = line_count.saturating_sub(transcript_height);
    let scroll = resolve_scroll(controller.scroll_offset, max_scroll);
    let transcript = Paragraph::new(lines)
        .block(Block::default().borders(Borders::ALL).title("Transcript"))
        .wrap(Wrap { trim: false })
        .scroll((scroll, 0));
    frame.render_widget(transcript, transcript_area);

    let mut scrollbar_state = ScrollbarState::new(usize::from(line_count))
        .position(usize::from(scroll))
        .viewport_content_length(usize::from(transcript_height));
    let scrollbar = Scrollbar::new(ScrollbarOrientation::VerticalRight)
        .begin_symbol(None)
        .end_symbol(None);
    frame.render_stateful_widget(scrollbar, transcript_area, &mut scrollbar_state);

    let input_title = if controller.processing {
        "Message (sending…)"
    } else {
        "Message"
    };
    let input_style = if controller.error_banner.is_some() {
        Style::default().fg(Color::Red)
    } else {
        Style::default()
    };
    // The input box has no cursor movement (only append/backspace at the
    // end), so the cursor is always at the end of `input`: scrolling to show
    // just the tail that fits keeps it visible instead of clipped off-screen.
    let input_inner_width = usize::from(input_area.width.saturating_sub(2)).max(1);
    let visible_chars = controller.input.chars().count().min(input_inner_width);
    let visible_input: String = controller
        .input
        .chars()
        .skip(
            controller
                .input
                .chars()
                .count()
                .saturating_sub(visible_chars),
        )
        .collect();
    let input_text = controller
        .error_banner
        .map_or_else(|| visible_input, |error| format!("error: {error}"));
    let input_paragraph = Paragraph::new(input_text)
        .style(input_style)
        .block(Block::default().borders(Borders::ALL).title(input_title));
    frame.render_widget(input_paragraph, input_area);

    let info = Paragraph::new(info_lines(
        session,
        config,
        controller.processing,
        context_length,
    ))
    .block(Block::default().borders(Borders::ALL).title("Info"))
    .wrap(Wrap { trim: false });
    frame.render_widget(info, info_area);

    if !controller.processing && controller.error_banner.is_none() {
        frame.set_cursor_position((
            input_area
                .x
                .saturating_add(1)
                .saturating_add(u16::try_from(visible_chars).unwrap_or(u16::MAX)),
            input_area.y.saturating_add(1),
        ));
    }

    max_scroll
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_offset_pins_to_the_bottom() {
        assert_eq!(resolve_scroll(None, 50), 50);
    }

    #[test]
    fn zero_distance_is_the_same_as_pinned() {
        assert_eq!(resolve_scroll(Some(0), 50), 50);
    }

    #[test]
    fn a_manual_distance_scrolls_up_from_the_bottom() {
        assert_eq!(resolve_scroll(Some(10), 50), 40);
    }

    #[test]
    fn a_distance_past_the_top_clamps_rather_than_underflows() {
        assert_eq!(resolve_scroll(Some(100), 50), 0);
    }
}
