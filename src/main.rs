mod agent;
mod bg;
mod context;
mod message;
mod paths;
mod session;
mod session_store;
mod stream;
mod thinking;
mod transcript;
mod tool;
mod tui;


use agent::Agent;
use std::sync::{Arc, Mutex};
use thinking::ThinkingLevel;
use clap::Parser;
use openai_oxide::client::OpenAI;
const SYSTEM_PROMPT: &str = "You are a coding agent. Use the tools to accomplish tasks. For long-running commands (tests, builds, dev servers), use bg_run instead of bash; its result is reported automatically when the task finishes. Your configuration and session history live in .kite/: previous sessions are stored as JSON transcripts in .kite/sessions/ and background task logs in .kite/tasks/ — read them when the user refers to previous work. Before quoting or summarizing any file's content, re-read it. Never answer from remembered file content — files may have changed since you last saw them.";

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
    context_window: u64,
    bash_timeout: u64,
) -> Agent {
    let base = Some(thinking == ThinkingLevel::Off);
    Agent::new(
        client.clone(),
        model,
        SYSTEM_PROMPT,
        context_window,
        std::time::Duration::from_secs(bash_timeout),
    )
    .with_thinking(thinking_cell, base)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    let client = OpenAI::with_config(openai_oxide::ClientConfig::new("local").base_url(cli.api_url));
    let model = cli.model.clone();

    let client = client.clone();
    let thinking_cell = Arc::new(Mutex::new(cli.thinking));
    tui::run(
        {
            let model = model.clone();
            let cell = thinking_cell.clone();
            move || {
                build_agent(
                    &client,
                    &model,
                    cli.thinking,
                    cell.clone(),
                    cli.context_window,
                    cli.bash_timeout,
                )
            }
        },
        model,
        thinking_cell,
    )
    .await
}
