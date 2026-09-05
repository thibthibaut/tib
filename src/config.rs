use eyre::WrapErr;
use serde::Deserialize;
use std::path::{Path, PathBuf};

const REQUIRED_FIELDS: &str =
    "model, max_steps, system_prompt, tool_timeout_seconds, tool_output_truncate_chars";

/// Tib's configuration, parsed from `$HOME/.config/tib/config.toml`.
#[derive(Debug, Deserialize, PartialEq, Eq)]
pub struct Config {
    pub model: String,
    pub max_steps: u32,
    pub system_prompt: String,
    pub tool_timeout_seconds: u64,
    pub tool_output_truncate_chars: usize,
}

fn config_path(home: &str) -> PathBuf {
    Path::new(home)
        .join(".config")
        .join("tib")
        .join("config.toml")
}

fn parse_config(contents: &str, path: &Path) -> eyre::Result<Config> {
    toml::from_str(contents).wrap_err_with(|| {
        format!(
            "Failed to parse config file at {}. Required fields: {REQUIRED_FIELDS}",
            path.display()
        )
    })
}

/// Loads and parses Tib's config file from `$HOME/.config/tib/config.toml`.
///
/// # Errors
///
/// Returns an error naming the expected path and required fields if `HOME`
/// is unset, the file is missing, or the file's contents are not valid TOML
/// matching [`Config`].
pub fn load_config() -> eyre::Result<Config> {
    let home = std::env::var("HOME").wrap_err("HOME environment variable is not set")?;
    let path = config_path(&home);
    let contents = std::fs::read_to_string(&path).wrap_err_with(|| {
        format!(
            "Config file not found at {}. Required fields: {REQUIRED_FIELDS}",
            path.display()
        )
    })?;
    parse_config(&contents, &path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_path_joins_home_and_expected_suffix() {
        let path = config_path("/home/alice");
        assert_eq!(path, PathBuf::from("/home/alice/.config/tib/config.toml"));
    }

    #[test]
    fn parse_config_accepts_valid_toml_with_all_fields() {
        let toml = r#"
            model = "openrouter/some-model"
            max_steps = 20
            system_prompt = "You are a helpful assistant."
            tool_timeout_seconds = 30
            tool_output_truncate_chars = 4000
        "#;

        let config = parse_config(toml, Path::new("/tmp/config.toml")).unwrap();

        assert_eq!(
            config,
            Config {
                model: "openrouter/some-model".to_string(),
                max_steps: 20,
                system_prompt: "You are a helpful assistant.".to_string(),
                tool_timeout_seconds: 30,
                tool_output_truncate_chars: 4000,
            }
        );
    }

    #[test]
    fn parse_config_rejects_toml_missing_a_required_field() {
        let toml = r#"
            model = "openrouter/some-model"
            max_steps = 20
            system_prompt = "You are a helpful assistant."
            tool_timeout_seconds = 30
        "#;

        let error = parse_config(toml, Path::new("/tmp/config.toml")).unwrap_err();

        let message = format!("{error}");
        assert!(message.contains("/tmp/config.toml"));
        assert!(message.contains("tool_output_truncate_chars"));
    }

    #[test]
    fn parse_config_rejects_malformed_toml() {
        let toml = "this is not valid toml [[[";

        let error = parse_config(toml, Path::new("/tmp/config.toml")).unwrap_err();

        let message = format!("{error}");
        assert!(message.contains("/tmp/config.toml"));
    }
}
