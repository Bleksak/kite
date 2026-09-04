mod agent;
mod context;
mod message;
mod tool;

use std::io::{BufRead, Write};

use agent::Agent;
use openai_oxide::client::OpenAI;

const SYSTEM_PROMPT: &str = "You are a coding agent. Use the tools to accomplish tasks.";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_url = std::env::var("API_URL")
        .unwrap_or_else(|_| "http://localhost:11434/v1".to_string());
    let model = std::env::var("KITE_MODEL").unwrap_or_else(|_| "llama3.2".to_string());

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;

    let client = OpenAI::with_config(openai_oxide::ClientConfig::new("local").base_url(api_url));
    let mut agent = Agent::new(client, model, SYSTEM_PROMPT);

    let mut stdin = std::io::stdin().lock();
    let mut stdout = std::io::stdout();
    let mut line = String::new();

    loop {
        print!("> ");
        stdout.flush()?;
        if stdin.read_line(&mut line)? == 0 {
            break;
        }
        let input = line.trim().to_string();
        line.clear();
        if input.is_empty() {
            continue;
        }

        match runtime.block_on(agent.chat(&input)) {
            Ok(text) => println!("{text}"),
            Err(error) => eprintln!("error: {error}"),
        }
    }

    Ok(())
}
