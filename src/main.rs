mod agent;
mod bg;
mod context;
mod message;
mod mode;
mod paths;
mod session;
mod session_store;
mod stream;
mod thinking;
mod tool;
mod transcript;
mod tui;

use agent::Agent;
use clap::Parser;
use mode::Mode;
use openai_oxide::client::OpenAI;
use std::sync::{Arc, Mutex};
use thinking::ThinkingLevel;

#[derive(Parser)]
struct Cli {
    #[arg(long, default_value = "http://localhost:11434/v1")]
    api_url: String,

    #[arg(long, default_value = "llama3.2")]
    model: String,

    #[arg(long, default_value_t = 24000)]
    context_window: u64,

    #[arg(long, value_enum, default_value_t = ThinkingLevel::XHigh)]
    thinking: ThinkingLevel,

    #[arg(long, default_value_t = 300)]
    bash_timeout: u64,
}

fn build_agent(
    client: &OpenAI,
    model: &str,
    thinking: ThinkingLevel,
    thinking_cell: Arc<Mutex<ThinkingLevel>>,
    mode_cell: Arc<Mutex<Mode>>,
    context_window: u64,
    bash_timeout: u64,
) -> Agent {
    let base = Some(thinking == ThinkingLevel::Off);
    Agent::new(
        client.clone(),
        model,
        mode_cell,
        context_window,
        std::time::Duration::from_secs(bash_timeout),
    )
    .with_thinking(thinking_cell, base)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    let client =
        OpenAI::with_config(openai_oxide::ClientConfig::new("local").base_url(cli.api_url));
    let model = cli.model.clone();

    let client = client.clone();
    let thinking_cell = Arc::new(Mutex::new(cli.thinking));
    let mode_cell = Arc::new(Mutex::new(Mode::Yolo));
    tui::run(
        {
            let model = model.clone();
            let cell = thinking_cell.clone();
            let mode = mode_cell.clone();
            move || {
                build_agent(
                    &client,
                    &model,
                    cli.thinking,
                    cell.clone(),
                    mode.clone(),
                    cli.context_window,
                    cli.bash_timeout,
                )
            }
        },
        model,
        thinking_cell,
        mode_cell,
    )
    .await
}
