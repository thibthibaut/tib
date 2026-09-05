use tib::config;

fn main() -> eyre::Result<()> {
    color_eyre::install()?;
    let _config = config::load_config()?;
    Ok(())
}
