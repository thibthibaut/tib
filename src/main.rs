mod app;
mod view;

use tib::config;
use tib::openrouter_client::OpenRouterClient;

#[tokio::main(flavor = "current_thread")]
async fn main() -> eyre::Result<()> {
    color_eyre::install()?;
    let config = config::load_config()?;
    let client = OpenRouterClient::new(config.model.clone())?;
    // A degraded (`None`) context length just means the Context
    // Visualization shows no free-capacity dots; it never blocks startup.
    let context_length = client.context_length().await.ok().flatten();

    let mut terminal = ratatui::try_init()?;
    let result = app::run(&mut terminal, config, client, context_length).await;
    ratatui::restore();
    result
}
