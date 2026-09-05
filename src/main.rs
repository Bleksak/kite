mod agent;
mod context;
mod message;
mod render;
mod tool;

use std::io::{IsTerminal, Write};

use agent::Agent;
use openai_oxide::client::OpenAI;
use render::Renderer;
use tokio::io::AsyncBufReadExt;

const SYSTEM_PROMPT: &str = "You are a coding agent. Use the tools to accomplish tasks.";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_url = std::env::var("API_URL")
        .unwrap_or_else(|_| "http://localhost:11434/v1".to_string());
    let model = std::env::var("KITE_MODEL").unwrap_or_else(|_| "llama3.2".to_string());
    let max_tokens = std::env::var("KITE_CONTEXT_WINDOW")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(24000);

    let client = OpenAI::with_config(openai_oxide::ClientConfig::new("local").base_url(api_url));
    let mut agent = Agent::new(client, model, SYSTEM_PROMPT, max_tokens);
    if let Ok(value) = std::env::var("KITE_THINKING") {
        let enabled = value != "0";
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
