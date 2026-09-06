//! Per-message token estimates for the Context Visualization, scaled to
//! match `OpenRouter`'s last authoritative `usage.total_tokens` (ADR-0004).
//!
//! Each message's raw estimate is `chars / 4`; estimates are then scaled so
//! their sum equals `total_tokens` exactly, with any rounding remainder
//! folded into the last message so the total never drifts.

use crate::model_client::{Context, Role};

/// One message's role and its scaled token estimate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MessageTokens {
    pub role: Role,
    pub tokens: u32,
}

fn char_estimate(text: &str) -> u64 {
    let chars = u64::try_from(text.chars().count()).unwrap_or(u64::MAX);
    chars.checked_div(4).unwrap_or(0)
}

/// Splits `total` evenly across `count` slots: an even distribution is what
/// `scale_to_total` produces when every slot's raw estimate is equal.
fn distribute_evenly(total: u64, count: usize) -> Vec<u64> {
    if count == 0 {
        return Vec::new();
    }
    let count_u64 = u64::try_from(count).unwrap_or(u64::MAX);
    scale_to_total(&vec![1; count], count_u64, total)
}

/// Scales each raw estimate proportionally to `raw`'s share of `raw_sum` out
/// of `total`, folding the rounding remainder into the last slot.
fn scale_to_total(raw: &[u64], raw_sum: u64, total: u64) -> Vec<u64> {
    let mut scaled: Vec<u64> = raw
        .iter()
        .map(|value| {
            value
                .saturating_mul(total)
                .checked_div(raw_sum)
                .unwrap_or(0)
        })
        .collect();
    let assigned: u64 = scaled.iter().sum();
    let remainder = total.saturating_sub(assigned);
    if let Some(last) = scaled.last_mut() {
        *last = last.saturating_add(remainder);
    }
    scaled
}

/// The number of empty dots the Context Visualization should draw for unused capacity.
///
/// Computed as `context_length`'s budget minus `used_tokens` already spent,
/// at the same ~1000-tokens-per-dot scale as the used dots. Returns `0` if
/// `context_length` is unknown or `used_tokens` has already reached it.
#[must_use]
pub fn free_dots(context_length: Option<u32>, used_tokens: u32) -> u32 {
    let Some(context_length) = context_length else {
        return 0;
    };
    context_length
        .saturating_sub(used_tokens)
        .checked_div(1000)
        .unwrap_or(0)
}

/// The percentage of `context_length`'s budget that `used_tokens` has consumed.
///
/// For the Context Visualization's percentage display. Returns `None` if
/// `context_length` is unknown (or reports zero), and clamps to `100` if
/// `used_tokens` has exceeded it — mirroring `free_dots`'s clamp-to-zero for
/// the same case.
#[must_use]
pub fn context_percent_used(context_length: Option<u32>, used_tokens: u32) -> Option<u32> {
    let context_length = context_length?;
    let percent = used_tokens
        .saturating_mul(100)
        .checked_div(context_length)?;
    Some(percent.min(100))
}

/// Estimates a per-message token count for each message in `context`, scaled
/// so the estimates sum to exactly `total_tokens`. Returns an empty `Vec` for
/// an empty context.
#[must_use]
pub fn scaled_message_tokens(context: &Context, total_tokens: u32) -> Vec<MessageTokens> {
    if context.messages.is_empty() {
        return Vec::new();
    }

    let raw: Vec<u64> = context
        .messages
        .iter()
        .map(|message| char_estimate(&message.display_text()))
        .collect();
    let raw_sum: u64 = raw.iter().sum();
    let total = u64::from(total_tokens);

    let scaled = if raw_sum == 0 {
        distribute_evenly(total, raw.len())
    } else {
        scale_to_total(&raw, raw_sum, total)
    };

    context
        .messages
        .iter()
        .zip(scaled)
        .map(|(message, tokens)| MessageTokens {
            role: message.role(),
            tokens: u32::try_from(tokens).unwrap_or(u32::MAX),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_client::ContextMessage;

    fn context_of(messages: Vec<ContextMessage>) -> Context {
        Context { messages }
    }

    #[test]
    fn empty_context_produces_no_message_tokens() {
        let context = context_of(Vec::new());

        assert_eq!(scaled_message_tokens(&context, 100), Vec::new());
    }

    #[test]
    fn single_message_gets_the_entire_total() {
        let context = context_of(vec![ContextMessage::User("hello there".to_string())]);

        let tokens = scaled_message_tokens(&context, 42);

        assert_eq!(
            tokens,
            vec![MessageTokens {
                role: Role::User,
                tokens: 42
            }]
        );
    }

    #[test]
    fn estimates_split_proportionally_to_message_length_and_sum_to_the_total() {
        // "aaaaaaaa" (8 chars) is twice the raw estimate of "aaaa" (4 chars).
        let context = context_of(vec![
            ContextMessage::System("aaaa".to_string()),
            ContextMessage::User("aaaaaaaa".to_string()),
        ]);

        let tokens = scaled_message_tokens(&context, 90);

        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].role, Role::System);
        assert_eq!(tokens[1].role, Role::User);
        assert_eq!(tokens[0].tokens, 30);
        assert_eq!(tokens[1].tokens, 60);
        let sum: u32 = tokens.iter().map(|message| message.tokens).sum();
        assert_eq!(sum, 90);
    }

    #[test]
    fn zero_heuristic_sum_falls_back_to_an_even_split_without_panicking() {
        // Every message estimates to zero raw tokens (fewer than 4 chars
        // each), so the proportional split would divide by zero.
        let context = context_of(vec![
            ContextMessage::System(String::new()),
            ContextMessage::User("ab".to_string()),
            ContextMessage::Assistant {
                text: Some("cd".to_string()),
                tool_calls: Vec::new(),
            },
        ]);

        let tokens = scaled_message_tokens(&context, 10);

        let sum: u32 = tokens.iter().map(|message| message.tokens).sum();
        assert_eq!(sum, 10);
        assert_eq!(tokens.len(), 3);
    }

    #[test]
    fn free_dots_is_zero_when_context_length_is_unknown() {
        assert_eq!(free_dots(None, 500), 0);
    }

    #[test]
    fn free_dots_scales_remaining_capacity_at_1000_tokens_per_dot() {
        assert_eq!(free_dots(Some(10_000), 3_400), 6);
    }

    #[test]
    fn free_dots_is_zero_rather_than_negative_when_used_exceeds_context_length() {
        assert_eq!(free_dots(Some(1_000), 5_000), 0);
    }

    #[test]
    fn context_percent_used_is_none_when_context_length_is_unknown() {
        assert_eq!(context_percent_used(None, 500), None);
    }

    #[test]
    fn context_percent_used_computes_the_percentage() {
        assert_eq!(context_percent_used(Some(1_000), 250), Some(25));
    }

    #[test]
    fn context_percent_used_clamps_to_100_when_used_exceeds_context_length() {
        assert_eq!(context_percent_used(Some(1_000), 2_000), Some(100));
    }

    #[test]
    fn context_percent_used_is_none_rather_than_dividing_by_zero() {
        assert_eq!(context_percent_used(Some(0), 100), None);
    }

    #[test]
    fn zero_total_tokens_produces_all_zero_estimates() {
        let context = context_of(vec![
            ContextMessage::System("aaaa".to_string()),
            ContextMessage::User("aaaaaaaa".to_string()),
        ]);

        let tokens = scaled_message_tokens(&context, 0);

        assert!(tokens.iter().all(|message| message.tokens == 0));
    }
}
