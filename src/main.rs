mod app;
mod view;

use tib::config;
use tib::openrouter_client::OpenRouterClient;

#[tokio::main(flavor = "current_thread")]
async fn main() -> eyre::Result<()> {
    color_eyre::install()?;
    let config = config::load_config()?;
    let client = OpenRouterClient::new(config.model.clone())?;

    let mut terminal = ratatui::try_init()?;
    let result = app::run(&mut terminal, config, client).await;
    ratatui::restore();
    result
}
