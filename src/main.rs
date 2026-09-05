mod agent;
mod context;
mod message;
mod render;
mod tool;

use std::io::{IsTerminal, Write};

use agent::Agent;
use clap::{Parser, ValueEnum};
use openai_oxide::client::OpenAI;
use render::Renderer;
use tokio::io::AsyncBufReadExt;

const SYSTEM_PROMPT: &str = "You are a coding agent. Use the tools to accomplish tasks. Before quoting or summarizing any file's content, re-read it. Never answer from remembered file content — files may have changed since you last saw them.";

#[derive(Parser)]
struct Cli {
    #[arg(long, default_value = "http://localhost:11434/v1")]
    api_url: String,

    #[arg(long, default_value = "llama3.2")]
    model: String,

    #[arg(long, default_value_t = 24000)]
    context_window: u64,

    #[arg(long, value_enum, default_value = "auto")]
    thinking: Thinking,

    #[arg(long, default_value_t = 120)]
    bash_timeout: u64,
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum Thinking {
    Auto,
    On,
    Off,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    let client = OpenAI::with_config(openai_oxide::ClientConfig::new("local").base_url(cli.api_url));
    let mut agent = Agent::new(client, cli.model, SYSTEM_PROMPT, cli.context_window);
    if cli.thinking != Thinking::Auto {
        let enabled = matches!(cli.thinking, Thinking::On);
        agent = agent.with_extra_body(serde_json::json!({
            "chat_template_kwargs": { "enable_thinking": enabled }
        }));
    }

    let tty = std::io::stdout().is_terminal();
    let mut stdin = tokio::io::BufReader::new(tokio::io::stdin());
    let mut line = String::new();

    loop {
        print!("> ");
        std::io::stdout().flush()?;
        if stdin.read_line(&mut line).await? == 0 {
            break;
        }
        let input = line.trim().to_string();
        line.clear();
        if input.is_empty() {
            continue;
        }

        let mut renderer = Renderer::new(tty, std::io::stdout(), std::io::stderr());
        match agent.chat(&input, &mut |event| renderer.on_event(event)).await {
            Ok(_) => renderer.finish(),
            Err(error) => eprintln!("error: {error}"),
        }
    }

    println!(
        "session over: {} prompt / {} completion tokens",
        agent.context.total_prompt_tokens,
        agent.context.total_completion_tokens
    );

    Ok(())
}
