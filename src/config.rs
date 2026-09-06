use eyre::WrapErr;
use serde::Deserialize;
use std::path::{Path, PathBuf};

const REQUIRED_FIELDS: &str =
    "model, max_steps, system_prompt, tool_timeout_seconds, tool_output_truncate_chars";

const DEFAULT_SYSTEM_PROMPT: &str = "You are Tib, a terminal assistant with one tool: running \
    bash commands on this machine. Use it to check real state (files, command output, installed \
    tools) instead of guessing, and prefer taking direct action over asking the user to do it \
    themselves. Keep responses concise and focused on the task, explaining your reasoning only \
    when it isn't obvious from the commands you ran. Ask for confirmation before running \
    commands that delete data, overwrite files, or otherwise can't be undone.";

/// Written to `$HOME/.config/tib/config.toml` the first time Tib runs and no
/// config file exists yet, so a first run works without the user having to
/// hand-write one first. Built from `DEFAULT_SYSTEM_PROMPT` rather than
/// duplicating it inline, so the two can't drift apart.
fn default_config_toml() -> String {
    format!(
        "model = \"deepseek/deepseek-v4-flash-0731\"\n\
         max_steps = 20\n\
         system_prompt = \"{DEFAULT_SYSTEM_PROMPT}\"\n\
         tool_timeout_seconds = 30\n\
         tool_output_truncate_chars = 10000\n"
    )
}

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

/// Loads and parses Tib's config at `path`, writing `default_config_toml()`
/// there first if nothing exists yet, so a first run works without the user
/// having to hand-write a config file first.
///
/// # Errors
///
/// Returns an error if creating a missing config's parent directory or
/// writing its default contents fails, or if the (now guaranteed to exist)
/// file's contents aren't valid TOML matching [`Config`].
fn load_config_at(path: &Path) -> eyre::Result<Config> {
    if !path.exists() {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).wrap_err_with(|| {
                format!("failed to create config directory {}", parent.display())
            })?;
        }
        std::fs::write(path, default_config_toml())
            .wrap_err_with(|| format!("failed to write default config to {}", path.display()))?;
    }

    let contents = std::fs::read_to_string(path).wrap_err_with(|| {
        format!(
            "Config file not found at {}. Required fields: {REQUIRED_FIELDS}",
            path.display()
        )
    })?;
    parse_config(&contents, path)
}

/// Loads and parses Tib's config file from `$HOME/.config/tib/config.toml`,
/// creating it with default values first if it doesn't exist yet.
///
/// # Errors
///
/// Returns an error if `HOME` is unset, a missing config file can't be
/// created, or the file's contents are not valid TOML matching [`Config`].
pub fn load_config() -> eyre::Result<Config> {
    let home = std::env::var("HOME").wrap_err("HOME environment variable is not set")?;
    let path = config_path(&home);
    load_config_at(&path)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A path under the real temp directory, unique per call, so tests
    /// never touch the real `$HOME/.config/tib/config.toml`.
    fn temp_config_path(unique: &str) -> PathBuf {
        std::env::temp_dir()
            .join(format!(
                "tib-config-test-{unique}-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_or(0, |duration| duration.as_nanos())
            ))
            .join("config.toml")
    }

    #[test]
    fn load_config_at_creates_the_default_file_when_none_exists() {
        let path = temp_config_path("creates-default");
        assert!(!path.exists());

        let config = load_config_at(&path).unwrap();

        assert_eq!(config.model, "deepseek/deepseek-v4-flash-0731");
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            default_config_toml()
        );

        std::fs::remove_dir_all(path.parent().unwrap()).unwrap();
    }

    #[test]
    fn load_config_at_leaves_an_existing_file_untouched() {
        let path = temp_config_path("leaves-existing");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let custom_toml = r#"
            model = "custom/model"
            max_steps = 1
            system_prompt = "custom prompt"
            tool_timeout_seconds = 1
            tool_output_truncate_chars = 1
        "#;
        std::fs::write(&path, custom_toml).unwrap();

        let config = load_config_at(&path).unwrap();

        assert_eq!(config.model, "custom/model");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), custom_toml);

        std::fs::remove_dir_all(path.parent().unwrap()).unwrap();
    }

    #[test]
    fn config_path_joins_home_and_expected_suffix() {
        let path = config_path("/home/alice");
        assert_eq!(path, PathBuf::from("/home/alice/.config/tib/config.toml"));
    }

    #[test]
    fn default_config_toml_parses_into_the_expected_defaults() {
        let config = parse_config(&default_config_toml(), Path::new("/tmp/config.toml")).unwrap();

        assert_eq!(
            config,
            Config {
                model: "deepseek/deepseek-v4-flash-0731".to_string(),
                max_steps: 20,
                system_prompt: DEFAULT_SYSTEM_PROMPT.to_string(),
                tool_timeout_seconds: 30,
                tool_output_truncate_chars: 10_000,
            }
        );
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
